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

## End-to-end results

These are warm medians of two requests after the first request at each shape,
with 32 output tokens. Every one of the 12 FP16 requests matches both the
baseline and llama.cpp's complete generated token sequence. All 12 INT4
requests match the prior INT4 baseline. No competing model/build ran during
timing. Results are specific to this model and this fanless M1; frequencies
were not pinned, and small timing differences are not treated as improvements.

| Prompt tokens | Baseline FP16 decode tok/s | Current FP16 decode tok/s | llama.cpp FP16 decode tok/s |
|---:|---:|---:|---:|
| 32 | 16.02 | 25.89 | 29.69 |
| 128 | 15.58 | 24.79 | 31.38 |
| 512 | 14.07 | 22.85 | 30.46 |
| 2048 | 11.64 | 18.93 | 27.05 |

FP16 decode improves approximately 1.59–1.63x. Matching llama.cpp still requires
approximately 15%, 27%, 33%, and 43% more decode throughput, respectively.
The final 2048-token warm samples span 18.53–19.33 tok/s; the preceding decode
pass reached 19.76–19.77 with the same long-context execution path. Report the
final series above, rather than selecting the fastest earlier run.

| Prompt tokens | Baseline FP16 TTFT seconds | Current FP16 TTFT seconds | llama.cpp FP16 TTFT seconds |
|---:|---:|---:|---:|
| 32 | 0.389 | 0.144 | 0.070 |
| 128 | 0.294 | 0.270 | 0.144 |
| 512 | 0.851 | 0.814 | 0.499 |
| 2048 | 4.392 | 4.384 | 2.307 |

The short-prefill optimization reduces 32-token TTFT by about 2.70x.
There is no qualified long-prefill speedup in this change: at 2048 tokens,
TTFT remains approximately 1.90x llama.cpp's time.

| Prompt tokens | Baseline INT4 decode tok/s | Current INT4 decode tok/s | Current INT4 TTFT seconds |
|---:|---:|---:|---:|
| 32 | 19.95 | 31.36 | 0.318 |
| 128 | 18.82 | 30.42 | 0.542 |
| 512 | 17.30 | 27.63 | 1.636 |
| 2048 | 13.47 | 22.76 | 7.000 |

INT4 decode improves approximately 1.57–1.69x against the same CT2 INT4 model.
This is not a matching llama.cpp Q4 comparison: the CT2 model retains FP16
embeddings/vocabulary weights and uses its own group-64 quantization format.

Code: `e3ff93af` (decode pass: `ab9a23bc`); baseline: `627e1a9b`.
Loaded optimized library SHA256:
`103ebfef0d114458e4b174cd2625ef54daedc675728d01a2e93332fa8df0dd39`.
Baseline library SHA256:
`dcdfc6132be39bd7b696b86773043f4fff40d9161d231acb19a098f2e250a2a3`.
llama.cpp build `b9481-bfb4308b0` uses the same FP16 snapshot, all layers on
Metal, F16 K/V, four CPU threads, one slot, context 4096, batch 2048, microbatch
512, and automatic FlashAttention. Prompt reuse and the RAM prompt cache are
disabled. Sampling is greedy TopK=1, TopP=1, temperature=1, seed=42.
Prompt-ID SHA256 is
`bd5e66e4e53d5304966f89eafffe19be6bc11b11df9a12f0612634b9eee9579c`.

Raw series: `throughput-baseline-{f16,int4}.json`,
`throughput-release-{f16,int4}.json`, `throughput-llama-f16.json`, and the
asserted comparison `throughput-comparison.json`. First-request TTFT, individual
decode intervals, median interval latency, warm ranges, load time, and process
RSS remain in the raw artifacts; cold results are not combined into warm medians.
Logical compact KV storage is 112 KiB per token. Capacity rounding gives
approximately 56/56/112/280 MiB at the four request sizes including generation;
these are geometry estimates, not total unified-memory measurements.

## Correctness and component validation

* 234 native Metal tests pass, including FP16/BF16 attention, odd head dimensions
  65/129/255, GQA/MQA, all-masked rows, windows, poisoned cache tails, split
  merging, and multi-token query order. Small routing plans are checked at
  the 128-assignment boundary and with empty/uneven experts. Vector expert paths
  retain CPU reference bounds and exact packed-versus-unpacked comparisons.
* 118 tiny decoder/spec tests pass, including INT4/shared experts, routing
  normalization, and explicit sliding-window schedules. Eight real dense
  regressions also pass. The Marian smoke retains all eight beam/batch/length
  hypotheses; Whisper retains its previous token hash (6.50 s smoke).
* Independent CPU source references at lengths 7/33 have maximum logit error
  0.17673 for FP16 and 0.08717 for INT4, maximum RMSE 0.01034/0.01197, and 100%
  argmax agreement. All generated reference tokens match. Limits remain
  max-absolute 0.2 and RMSE 0.025. Cached/full-sequence checks retain matching
  greedy choices. The final short tile's full logits are bit-identical to the
  preceding implementation at both checked lengths, for both weight formats.
* Long FP16 source comparisons at 32/128/512/2048/2051 tokens have maximum error
  0.11044, maximum RMSE 0.01024, and exact greedy choices. The new short tile
  leaves that long-prefill path unchanged; final cold/warm 32-token generation
  checks pass at every benchmark length.
* Same-build attention A/B at 2048 keys measures 0.79145 to 0.37530 ms; at 8192,
  3.09165 to 1.36895 ms. These averages include encoding/execution, not isolated
  GPU timestamp samples. Whole-process diagnostics for two 32-prompt/4-output
  requests reduce compute dispatches from 8926 to 8478. Both record zero CPU
  fallbacks, GPU sampling TopK, and 65 synchronizations including model setup.
  These totals are not per-token counts; synchronized scope profiling was off.

Logs and parity reports use `throughput-short-gemm-*`, `throughput-f16-long-*`,
`throughput-final-tests-mps.log`, `throughput-dense-tests.log`,
`throughput-existing-models.log`, `throughput-release-attention*`, and
`throughput-release-dispatch.*`. The repeated short-prompt perplexity smoke is
unchanged (33 tokens: FP16 197.80, INT4 224.33); broad corpus quality and a
matching Q4 comparison remain outstanding.

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
long-prefill path is preserved.

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
