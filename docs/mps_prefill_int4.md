# Metal decoder prefill, INT4, shared experts, and window attention

This extends the decoder implementation in [MPS MoE](mps_moe_routing.md).
Correctness and timing qualification are separate: primitive parity does not
establish model quality or a speed advantage over llama.cpp. The measurements
below use a MacBook Air M1, seven GPU cores, 8 GB unified memory, macOS 27.0.1,
a native arm64 Release build, and one generator worker with four CPU threads.

## Measured bottleneck and implementation

The real model is
`DavidAU/Qwen3-MOE-4x0.6B-2.4B-Writing-Thunder-V1.2`, pinned to
`f85830ed241787c392660f3e064050ccacac9602`. Its actual matrices have 28 layers,
residual width 1024, expert intermediate width 3072, four experts, selected TopK
2, 16 query heads, eight KV heads, and head depth 128. The source contains
approximately 1.39 billion weights; the repository name is not a parameter count.

A synchronized attribution run at 2048 prompt tokens put 6.707 seconds in expert
gate/up, 3.601 seconds in expert down, and 3.931 seconds in attention MatMul.
These are profiler attribution numbers, not ordinary end-to-end timings: the
profiler synchronizes scopes. `ENABLE_PROFILING` is off for reportable timing.

`moe_prefill_metal.h` adds a 64x64x32 FP16 tile with FP32 SIMD-matrix accumulators.
Gate/up share gathered activation loads; down has a separate specialization so
it does not allocate unused up accumulators. Padded threadgroup leading dimensions,
aligned vector loads, and scalar guarded tails cover real widths and odd shapes.
Routing counts, grouping, inverse mapping, and indirect grids remain on the GPU.
Up to eight rows retain GEMV; smaller or unsupported tiles keep the old paths.
The wide tile is selected for FP16 batches with at least 64 input rows.

An expert-only intermediate run reduced warm 2048-token TTFT from 15.106 to
7.007 seconds, with all 32 generated IDs unchanged at 32/128/512/2048 tokens.
Synchronized gate/up and down attribution fell to 1.210 and 0.670 seconds.
Attention then became the largest remaining prefill component.

`flash_prefill_metal.h` adds native 32-query by 32-key tiled attention. FP32 QK,
maximum-rescaled online softmax, denominators, and PV accumulation avoid the
full Q-by-K attention matrix. Depth capacities 64/128/256 avoid runtime-sized
accumulator arrays. It supports compact GQA/MQA, causal or noncausal attention,
past offsets, per-row length bounds, sliding windows, empty masked rows, and
odd tails. Shared-state updates have explicit threadgroup barriers.

The compact FlashAttention layer selects this kernel at 64 or more query rows.
Short prefills retain the established materialized GPU path. Attention-weight
requests and unsupported configurations retain their generic implementation.
`CT2_MPS_DISABLE_FLASH_PREFILL=1` disables the new path for comparison. The old
`CT2_MPS_USE_STREAMING_ATTENTION` experiment is unrelated and remains unqualified.
Whisper/Marian retain their existing encoder/prefill selection and previously
qualified incremental attention hooks.

## Short FP16 cache logits

A short real-model check exposed FP16 vocabulary storage rounding distinct
winning logits to equal values. Decoder-only MPS FP16 projections now keep their
FP32 accumulator through the vocabulary logits. Activations, attention, and KV
storage remain FP16. Sampling, beam search, forward logits, scoring, and returned
snapshots use the decoder's separate `logits_type()` contract. Encoder-decoder
models retain their existing vocabulary projection selection.

`mixed_gemm_metal.h` supplies mixed FP16-input/weight, FP32-output GEMV/GEMM.
Independent double-precision tests cover multiple rows, odd widths, bias, and a
constructed pair that is distinct in FP32 but tied after FP16 rounding. This
removes one source of artificial ties; FP16 hidden-state arithmetic can still
change sufficiently close decisions. Exact greedy parity remains a model-level
gate, with known qualification gaps documented rather than broader tolerances.

## Native group-64 INT4 format

`--quantization int4_float16` is available for decoder-only Transformer conversion.
It requires unquantized floating-point source matrices. Existing AWQ/GPTQ/INT8
packed matrices are rejected rather than treated as ordinary weights. Drama3,
Whisper, and Marian conversion do not silently accept this new mode.

Each linear output row stores consecutive low/high nibbles, with signed value
`nibble - 8`, symmetric values -7 through 7, a positive FP32 scale per 64 input
columns, and the original input depth. Odd final nibbles are padded. Zero blocks
have scale 1 and encoded zero. Conversion uses nearest-even rounding and rejects
nonfinite weights. Embeddings, vocabulary projection, routers, and scalar shared
expert gates remain FP16; this is mixed precision, not an assertion that every
model weight occupies four bits.

Packed matrices and scales are immutable model-owned storage. They are packed
once during conversion; there is no pointer-keyed global cache, per-token
repacking, or complete MPS dequantized weight allocation. Dense and expert GEMV
load dequantized values into registers. GEMM dequantizes only the current tile
into threadgroup memory. Weight values round to FP16 before FP32 accumulation,
matching the independent CPU reference. The wide expert tile supports the same
format, including paired gate/up and quantized shared experts through Dense.

The loader validates packed shape, original depth, scales, device, and compute
type. MPS requires FP16 activations. CPU expands weights once at load and uses
its ordinary reference path. Unsupported GPU/compute modes fail explicitly;
there is no per-token host fallback. Packed buffers and asynchronous dispatch
resources follow existing model/Metal allocator lifetimes.

A memory-bounded conversion tool also accepts an existing CT2 FP16 decoder:

```bash
export PYTHONPATH="$PWD/python"
python tools/benchmark/llm/pack_int4.py /path/to/ct2-f16 /path/to/new-ct2-int4
```

It streams one linear matrix at a time and copies other records in bounded
blocks. It requires a new directory outside the source and removes its own
partial output on failure. Aliased linear matrices need source-model conversion;
ordinary embedding/head aliases remain intact. The pinned real MoE checkpoint
packs 392 matrices: 2,777,820,161 bytes become 1,005,259,273 bytes.

## Shared experts and sliding windows

Qwen2 MoE conversion supports the routed expert sum plus
`sigmoid(shared_gate(x)) * shared_expert(x)`. The shared expert is ordinary
SwiGLU and can use INT4 Dense projections. A fused GPU row-gate/add kernel
preserves dtype rounding without broadcasting a gate into a full temporary.

`MoeTopK(k, false)` retains probabilities from the full expert softmax; the
default still normalizes the selected experts. Selected IDs retain the existing
stable descending-logit/ascending-index order. Full-softmax denominators use
compensated FP32 summation on MPS. CPU and MPS support this mode; CUDA/HIP rejects
it until implemented, preserving the previous default routing behavior.

Mixtral and Qwen MoE window configurations use explicit per-layer absolute
window metadata. Decoder-wide chunking is disabled for those converted models,
so full-sequence APIs return every input row, including prompts longer than the
window. Generic CPU/MPS attention and native compact attention apply the same
per-query lower/upper key bounds. Mixed full/window layer schedules are retained.

Cache prefixes are currently retained; window masking is correct but memory is
not bounded by the window. Eviction requires tracking a cache base position and
is separate future work. Legacy Qwen2 MoE configurations without `layer_types`
are rejected when their eager/Flash schedules disagree; supply an explicit
compatible schedule. Unsupported schedules are never changed to full attention.

## Validation and reproducibility

Focused C++ tests cover double-reference attention (FP16/BF16, GQA/MQA,
causal/noncausal masks, all-masked rows, window/past offsets, long contexts,
large scores, and poisoned padding), INT4 multi-group/tail GEMV/GEMM, wide/small/
GEMV expert dispatch, empty experts, full-softmax weights, and precise logits.
Tiny complete-decoder tests cover CPU FP32, MPS FP16, and native INT4, generic and
compact attention, Qwen3 dense/MoE, Mixtral, shared Qwen2 MoE, routing TopK 1/2/4,
window lengths beyond the cache window, scoring, cached/full logits, exact greedy
IDs, repeated requests, beam reordering, and seeded TopK/TopP/temperature filtering.

`reference_moe.py --int4` independently quantizes source Transformers layer
weights with torch and runs a memory-bounded CPU FP32 reference. It does not call
the converter, native kernels, or CT2 CPU dequantization. Compare exports with
`validate.py` before timing. Its prompt perplexity is a regression smoke metric,
not a held-out corpus quality qualification. A broader corpus and matching
quantization in llama.cpp remain necessary for an INT4 quality/speed claim.

The initial source INT4 reference has cached/full maximum logit errors below
0.00011 at 7/33 tokens, exact four-token argmax agreement, and prompt perplexities
959.94/224.62. Compare these only with the same source prompts/settings.

Reportable end-to-end runs must have profiling off, no competing CPU/GPU model,
identical saved prompt IDs, tokenizer, output length, quantization, sampling,
and cache configuration. RSS is not peak total unified memory; GPU utilization
and synchronization cost are not inferred from wall-clock throughput. Historical
FP16 llama.cpp baselines in the MoE document remain faster than the prior CT2
implementation. This change does not claim to outperform llama.cpp.

## Last complete end-to-end benchmark

One first request plus two warm repeats per shape; 32 generated tokens, greedy,
TopP/temperature 1, seed 42, identical saved prompt IDs and tokenizer. Timings
exclude model/tokenizer loading and include callback delivery. The matching
llama.cpp run uses the same source F16 weights, F16 KV cache, routing, hardware,
and output length (commit `bfb4308b058b334c6e68085c661ec9eb7e3d59f4`).

| Prompt | Prior CT2 F16 TTFT s | New CT2 F16 TTFT s | llama.cpp F16 TTFT s | New CT2 F16 decode tok/s | llama.cpp F16 decode tok/s |
|---:|---:|---:|---:|---:|---:|
| 32 | 0.385 | 0.410 | 0.155 | 15.55 | 27.31 |
| 128 | 0.846 | 0.297 | 0.147 | 15.49 | 29.39 |
| 512 | 3.105 | 0.852 | 0.534 | 14.16 | 25.31 |
| 2048 | 15.106 | 4.147 | 2.396 | 11.51 | 25.90 |

Long-prefill TTFT improved 3.64x over the prior implementation. Short TTFT and
decode regressed in this intermediate run; this is explicitly not an across-
the-board speedup. An attempted compile-time FP16/INT4 specialization recovered decode throughput
but failed the real-model cold/warm token-identity check at 512 tokens, despite
passing primitive tests. It was reverted before promotion to master. The slower
validated implementation remains the baseline; no decode recovery is claimed.

| Prompt | Native INT4 TTFT s | Native INT4 decode tok/s |
|---:|---:|---:|
| 32 | 0.359 | 19.91 |
| 128 | 0.561 | 19.04 |
| 512 | 1.670 | 17.52 |
| 2048 | 7.078 | 13.56 |

These INT4 figures are an internal mixed-precision result, not a matching-Q4
comparison with llama.cpp. Its quantization scheme is different. INT4 improves
storage and decode relative to the measured new FP16 path but has slower prefill.
Peak process RSS was 1.020/1.436 GB for F16/INT4; this does not measure total
unified/GPU memory and must not be used as an INT4 memory-footprint comparison.
Logical compact FP16 KV storage is 112 KiB/token, rounded to 512-token capacity
blocks (280 MiB at capacity 2560); window prefixes are still retained.

Artifacts: `moe-prefill-final-timing.*`, `moe-int4-final-timing.*`,
`moe-prefill-final-long-parity.*`, `moe-int4-final-parity.*`, and
`llama-moe-timing.*` under the external `mps-llm-artifacts` directory. The complete
new timing run loaded library SHA256
`a2b863603548ebd98d39ec02b6ef9d1d9aeffca7cdfc4ded2054c2465384bffe`.

Release validation before publication: 252 Metal C++ tests and 369 CPU/shared
C++ tests passed; 74 complete tiny decoder tests and 46 specification tests
passed. Six dense real-model repeated-prefill cases and both formerly failing
mixed-length cache cases passed; the near-tie expected-failure allowance is
removed. The real F16 MoE matches all four independent CPU greedy IDs at
32/128/512/2048/2051 tokens, with maximum cached-logit error 0.1298 and RMSE below
0.0094. Real INT4 7/33-token full/cached logits match source-quantized CPU argmax
and greedy IDs, maximum error 0.1128 and RMSE below 0.0145. Its 33-token prompt
perplexity is 224.33 versus 197.80 for F16, a 13.4% increase on this small smoke
sample. This is not broad corpus quality qualification.

The clean master port keeps the fork's CTranslate2 4.8.1 version and is based
on `9206f367cb920947ddab5715abfac5b2ee8ab30b`, with the new implementation
collected into one commit directly above that master revision. Its isolated
Release build passed 233 Metal C++ tests, 209 CPU/shared tests (one existing
Conv1D dilation skip), and 118 Python tests (74 complete decoder and 44
specification cases). Real F16 MoE full/cached checks at 7/33 tokens and cached
checks at 32/128/512/2048/2051 tokens pass the independent CPU references. Real
INT4 full/cached checks at 7/33 tokens also pass, with exports identical to the
qualified implementation above. The fresh installed library SHA256 is
`ddcdfc6132be39bd7b696b86773043f4fff40d9161d231acb19a098f2e250a2a3`.
All eight Marian reference hypotheses match exactly. Whisper retains the prior
output hash `699bbdcdc1b00dd7b31d6e3bdb86d5b081e2b859e86f824f19f716bc816b85ce`.
The timing tables above retain their original library/build scope; these fresh
master-port checks establish correctness, without claiming a new timing result.

Remaining performance work: restore decode throughput while retaining precise
logits, tune INT4 dense/QKV prefill tiles, reduce per-expert dispatch counts,
reuse routing workspaces, improve long-context attention scheduling and KV
bandwidth, implement position-aware bounded window-cache eviction, and collect
GPU counters/isolated timings plus broader model/context/quantization quality
results. The standard Metal kernels remain behind llama.cpp on this machine.
