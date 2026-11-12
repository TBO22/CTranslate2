# Native Metal attention: prefill and incremental decoding

Native tiled prefill extends the incremental implementation. See
[prefill, INT4, shared experts, and windows](mps_prefill_int4.md) for the current
selection rules, model validation, and performance qualification. Historical
measurements below retain their original scope.

## Execution paths

`Generator(..., device="mps", compute_type="float16", flash_attention=True)`
selects compact `[batch, capacity, KV heads, depth]` caches, with 512-token growth
blocks. New K/V rows are appended in place. Valid prefix rows are copied only on
capacity growth. GQA/MQA heads are read directly without retaining replicated
KV caches. Beam reordering uses the existing GPU Gather operation.

A native split-K kernel computes QK, stable softmax and the value product in
256-key tiles. FP32 maxima, denominators and unnormalized value numerators are
merged with maximum rescaling. The workspace is
`B * Hq * Q * ceil(K / 256) * (D + 2)` floats. No full attention matrix is written.
The implementation supports FP16/BF16 storage, causal and length masks,
sliding-window masking, empty masked rows, GQA/MQA, and head depths through 256.
Invalid/padded KV rows are never read, even when they contain NaNs.

Compact prefills of at least 64 query rows use native tiled online-softmax
attention for supported shapes. Short prefills and unsupported native shapes
use the established GPU MatMul/SoftMax/MatMul path. Requested attention weights also require materialization. ALiBi,
relative position bias, merged encoder attention, and tensor parallelism are
not supported by the new compact-cache layer; use `flash_attention=False` for
these configurations. RoPE and optional Q/K/V normalization run before attention.
The BTHD RoPE time indexing is covered by a new regression.

The shared `FusedAttention` hook also selects native single-query decoding in
the existing BHTD layout. This covers ordinary Whisper and Marian decoder
self-attention without changing their model APIs or cache layouts. Encoder
prefill remains on the existing path. Attention-weight requests, relative
biases and unsupported shapes keep their generic implementation. Disable this
shared hook for comparison with `CT2_MPS_DISABLE_FLASH_DECODE=1`.

Cross-attention with interleaved beam queries and shared encoder KV is
implemented and primitive-tested, but opt-in through
`CT2_MPS_USE_FLASH_CROSS_ATTENTION=1`. The first real Whisper comparison was
slower, so it is not enabled by default. This setting is experimental.

The older `CT2_MPS_USE_STREAMING_ATTENTION=prefill` experiment remains separate
and unqualified. Do not enable it when reproducing the default measurements.

## Validation and measurement

The tests in `tests/mps_flash_attention_test.cc` compare against an independent
double-precision attention calculation using exact representable FP16/BF16
inputs. They cover odd lengths, split boundaries, depths 13/64/128/256,
GQA/MQA, batch two, causal/window/empty/negative/oversized masks, large scores,
poisoned unused cache capacity, in-place append, growth, beam reordering,
interleaved beam queries and generic fallback selection.

Real-model timing must be collected without logit capture, profiling or a
competing model process. `tools/benchmark/llm/benchmark.py --flash-attention`
records the loaded library hash, exact prompt IDs, sampling controls, individual
token intervals, TTFT, RSS and estimated KV capacity. Prompt tokens divided by
TTFT includes first-token decoding and sampling; it is not isolated prefill
throughput. RSS is not total peak unified/GPU memory. Microbenchmark wall time
includes enqueue/dispatch and amortized synchronization, not isolated GPU time.

### Local M1 results (2026-11-11)

MacBook Air M1, 7-core GPU, 8 GB RAM, macOS 27.0.1, arm64 Release. Qwen3-0.6B
snapshot `c1899de289a04d12100db370d81485cdf75e47ca`, FP16 weights and KV,
32 output tokens, greedy TopK=1, TopP=1, temperature=1, seed=42. Baseline is the
NaN-fixed `e4907f0a` library, SHA256
`52f41e6b8ad9d1eef87d5cc7b014c7ffd32e6f10568b9f97f92c90a52354f28e`.
All exact prompt IDs are stored in the benchmark's prompt manifest (SHA256
`bd5e66e4e53d5304966f89eafffe19be6bc11b11df9a12f0612634b9eee9579c`).
Median warm results, three baseline warm repetitions and four candidate warm
repetitions, one cold/first-per-shape request excluded:

| Prompt | Baseline decode tok/s | Compact Flash decode tok/s | Baseline TTFT s | Compact Flash TTFT s | llama.cpp decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 24.46 | 35.37 | 0.0919 | 0.0771 | 45.18 |
| 128 | 21.34 | 32.87 | 0.1949 | 0.1795 | 44.85 |
| 512 | 16.86 | 27.94 | 0.7798 | 0.7605 | 43.94 |
| 2048 | 7.38 | 18.88 | 5.7387 | 5.8295 | 39.07 |

Candidate library SHA256:
`40c928da416cc7294b712e2dd131c4f0915cedf282f77a74608a25df930d968d`.
Process peak RSS increased from 1.20 GB to 1.57 GB in these runs; this is not
reported as an overall memory reduction. Logical compact KV is 112 KiB/token,
versus 224 KiB/token for the replicated cache; allocated capacity includes the
512-token rounding, and temporary/workspace allocations remain to be profiled.

All 32 generated IDs match CPU at all four lengths, including cold and repeated
requests. This is 1.45–2.56x faster decoding than the fixed baseline in these
runs. Long-prompt TTFT did not improve. llama.cpp remains substantially faster;
its 2048-prompt TTFT was 1.5575 s. The comparison uses the same pinned snapshot
converted to F16 GGUF, exact prompt token IDs, F16 KV and sampling, llama.cpp
`bfb4308b058b334c6e68085c661ec9eb7e3d59f4`, all layers on GPU, context 4096,
one slot, no prompt reuse. HTTP/SSE and Python callback overhead differ.

Local Whisper smoke: beam five, 12.515-second Pashto audio, one cold and three
warm requests. Shared decoder self-attention reduced warm median from 6.882 to
6.529 s (about 5%). Every token matched. Enabling both native self and
interleaved-beam cross-attention took 8.467 s, so the cross path stays opt-in.
Local Marian smoke: source lengths 32/128, batch 1/4, beam 1/4, output cap 32,
one cold and three warm requests per case. Decoder self-attention reduced warm
medians by approximately 1–10%; all tested hypotheses matched the established
MPS path. These small local smoke tests do not replace corpus WER/BLEU or
qualification on other Apple GPU generations.

Artifacts are under the workspace's `mps-llm-artifacts/` directory:
`flash-fixed-baseline-timing.json`, `flash-final-timing.json`,
`llama-f16-timing.json`, `whisper-flash-{disabled,self-only,enabled}.json`, and
`marian-flash-{disabled,self,cpu}.json/.npz`. Model assets are not committed.

### Numerical qualification still outstanding

The new primitive passes independent FP16/BF16 reference tests, but this does
not establish exact greedy agreement for every FP16 model/prompt/batch. A mixed
33/65-token Qwen batch changes its second sequence at generated token four:
CPU logits favor token 11504 over 1376 by 0.02926, while compact Flash FP16
logits round both to 18.15625, selecting the lower token index 1376. Separate
single-sequence requests match CPU. This is a measured near-tie and a failed
exact greedy qualification case, not accepted as full batched model parity.
Use `flash_attention=False` for this configuration while it is investigated.
The real-model pytest suite compares this mixed-length batch with and without
logit/score capture. It recognizes only the exact previously observed token
disagreement as an expected failure; other disagreements still fail.
Repeated results are deterministic and finite; no softmax/shared-memory
corruption has been found in this case.

Short full/cached Qwen logits at lengths 7/33/65 match CPU within maximum
absolute error 0.113 (full) and 0.063 (cached), with exact generated four-token
sequences. A length-257 full-vocabulary outlier already exists in the established
MPS prefill path; compact Flash prefill preserves those full logits bit-for-bit.
The older experimental tiled prefill remains disabled. No tolerances were
widened to qualify these outliers. Seeded temperature 0.7/TopK 10/TopP 0.9
replays match in fresh processes; resetting the existing API seed does not
rewind the MPS random counter within a process.


Final validation: MPS C++ 239 passed / 15 skipped; CPU/shared C++ 369 passed /
16 skipped; Python real-model, storage and translation 78 passed / 5 skipped.
The CPU-only build also succeeds. Long generated-logit checks at lengths
33/1025/2048/2051, cold plus one warm repeat, are finite with exact CPU argmax
agreement; maximum absolute errors are 0.051/0.063/0.065/0.067 and RMSE below
0.012. The primitive benchmark's enqueue/amortized wall times at K=512/2048/8192
were 0.247/0.800/2.985 ms for native compact attention versus
0.359/0.838/3.248 ms for materialized attention; it is not an isolated GPU timer
and its KV layout/replication differs intentionally.

Marian raw-logit comparison at batch four matched the established MPS argmax
at every captured position (maximum difference 0.125, RMSE up to 0.068). The
single-item `return_logits_vocab` diagnostic initially produced unreliable early
snapshots on both established and candidate MPS paths. The subsequent MoE work
pinpointed and fixed asynchronous deep copies during final hypothesis sorting;
see [the result-handoff diagnosis](mps_llm_nan_diagnosis.md). New tiny-model
tests cover every returned logit. Repeating the real Marian single-item capture
at lengths 32/128 now matches CPU argmax at all four steps, with maximum finite
logit error 0.119/0.131 after excluding padded/suppressed entries.
Token timings above omit logit capture and retain identical hypotheses. CPU
vocabulary padding differs (64171 vs MPS 64176), so references must exclude the
padded tail and suppressed-token sentinels.

The subsequent [M1 throughput work](mps_throughput_optimization.md) adds a
separate cooperative QK entry point for compact-cache decode. The standard-layout
Whisper/Marian kernel remains the established implementation.
