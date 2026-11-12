# M1 MoE decode throughput work

This work starts from `627e1a9b` on the fork's clean master, on
`perf/apple-mps-moe-throughput`. The target remains measured parity with
llama.cpp; this change does not establish that parity.

## Measured bottlenecks and changes

The pinned Qwen3 MoE has hidden width 1024, intermediate width 3072, four
experts, TopK=2, 28 layers, and 128-dimensional attention heads. Benchmarking
complete expert execution at these dimensions exposed approximately 1.6 ms
per layer for single-token expert evaluation. Storage-type/quantization
branches and scalar loads sat inside the reduction loop.

* FP16 expert GEMV now loads four adjacent activation and weight values per
  lane, with FP32 FMA accumulation. INT4 uses a two-byte load for four nibbles,
  reuses their group scale, and rounds each dequantized weight to FP16 before
  accumulation, matching the format's reference semantics. Weights stay packed;
  there is no full-matrix dequantization or per-token packing.
* The vector path requires the existing aligned, depth-divisible-by-eight
  layout. FP32, BF16, odd depths, and unsupported alignment retain the scalar
  path.
* Short FP16/INT4 expert prefills (9 through 63 input rows) now use a
  16x64x32 SIMD-matrix tile with half threadgroup storage and FP32 accumulation.
  This shares the long-prefill kernel's implementation, reduces barriers and
  reuses activations across more output columns than the 16x32x16 fallback.
  Complete 31-row expert evaluation measured 2.18 versus 7.61 ms in the A/B
  diagnostic. Long prefills retain their 64x64x32 tile. Other storage types and
  explicit reference comparisons retain the generic implementation.
* Small GEMV routing plans with at most 128 assignments now generate expert
  offsets, stable permutations, inverse permutations, and indirect grids in
  one dispatch instead of three. Each expert computes its prefix independently
  from GPU routing IDs; it never reads another threadgroup's partial results.
  Larger plans and GEMM retain the general three-dispatch implementation.
* Compact-cache decode attention now distributes QK dot products across SIMD
  lanes. Adjacent lanes read adjacent key dimensions, and query fragments stay
  in registers. FP32 scores and stable softmax/split merging are preserved.
  Barriers separate score production, score consumption, and probability reuse.
  The attention matrix is never materialized.
* Cooperative QK is a separate compiled entry point for compact caches with
  head dimension 64 through 256. Standard-layout attention, including the shared
  Whisper/Marian hook, retains its previous compiled kernel. Unsupported shapes
  retain the existing fallback behavior.

Expert outputs are still accumulated in ascending expert order with the
original dtype rounding boundaries. Router normalization, sampling, cache
layout, quantization format, and model conversion are unchanged.
Vectorization changes the FP32 reduction order, so logits are compared to the
independent source reference rather than required to be bit-identical.

## Reproduction and interpretation

Use the commands in [the benchmark guide](../tools/benchmark/llm/README.md).
Timing requires a native arm64 Release build with `ENABLE_PROFILING=OFF`, an
idle GPU, and separate model processes. Use the saved prompt IDs, greedy
sampling, seed 42, 32 output tokens, and two warm repeats after the first
request at each shape. Do not include `--validate-logits` in timing runs.

The local `mps-llm-artifacts/throughput-*` files retain raw intervals, prompt IDs,
runtime controls, loaded-library SHA256, reference comparisons, and test logs.
The source snapshot is
`f85830ed241787c392660f3e064050ccacac9602` of
`DavidAU/Qwen3-MOE-4x0.6B-2.4B-Writing-Thunder-V1.2` (approximately 1.39B actual
weights). Hardware is the 7-core GPU M1 MacBook Air with 8 GB unified memory.

TTFT includes first-token decoding, sampling, and callback delivery; prompt
tokens divided by TTFT is not isolated prefill throughput. Process RSS is not
total peak unified memory. Kernel microbenchmarks are diagnostics and cannot
substitute for end-to-end measurements.

Diagnostic A/B switches, recorded in benchmark metadata:

```bash
CT2_MPS_DISABLE_VECTOR_MOE=1
CT2_MPS_DISABLE_SMALL_MOE_PLAN=1
CT2_MPS_DISABLE_COOPERATIVE_QK=1
CT2_MPS_DISABLE_SHORT_MOE_GEMM=1
```

The internal `force_generic_plan` argument supports exact plan/expert regression
comparisons. The real-model regressions clear the diagnostic switches so that
they always qualify the default path. The `benchmark_mps moe-experts` target
measures planning, gate/up, SwiGLU, down, and aggregation at the real geometry;
`benchmark_mps flash-decode` measures attention at 512/2048/8192 keys.

## Rejected experiments

Direct expert dispatch from IDs removed four workspaces and all three planning
dispatches, but issued unused expert/slot threadgroups and provided no reliable
end-to-end improvement. It was removed.

Using FP16 SIMD-matrix operands with FP32 accumulators in the existing wide
prefill tile did not improve measured expert latency on this M1. Reducing the
tile from 64x64x32 to 32x64x32 was slower: approximately 59.8 versus 50.8 ms for
2048-token expert evaluation. Both experiments were removed; the existing
prefill path is preserved.

## Remaining work

Long-context prefill still needs better GEMM/attention scheduling and utilization.
INT4 prefill remains slower than FP16; its dense projections still use small
16x32x16 tiles. Changing expert tiles alone did not solve this.

Expert execution still encodes two projections per expert, including empty
experts' no-op indirect grids. A GPU-selected, grouped expert implementation
would reduce host encoding and dispatch cost, particularly for architectures
with many more than four experts. It needs model-owned resource tables and
careful asynchronous lifetime/concurrency validation.

Further opportunities include fusing norm/RoPE/QKV preparation, sharing GQA KV
reads, improving the large FP16 vocabulary projection, and reusing workspaces
within a decoder replica. INT4 vocabulary weights, a matching llama.cpp Q4
comparison, broad corpus quality evaluation, GPU utilization, and true peak
unified-memory measurements remain unqualified.
