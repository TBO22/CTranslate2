# Decoder MoE on CPU and Metal

The standard decoder supports SwiGLU MoE blocks, including Qwen3 MoE, Mixtral,
and shared-expert Qwen2 MoE checkpoints. The current
[prefill/INT4/shared/window implementation](mps_prefill_int4.md) extends the
original normalized FP16 implementation documented below. This includes conversion,
router projection, expert selection, token grouping, expert execution, weighted
aggregation, residuals, prefill, and cached/beam decoding. Dense layers and MoE
layers can coexist in the same decoder. The separate Drama sparse layer is not
used by these models.

## Execution and lifetime

`layers::MixtureOfExperts` is selected when a standard FFN has a router weight.
The converter writes a router, TopK/normalization attributes, and independent
expert gate/up/down projections. CPU FP32 uses existing Dense/Gather operators
and a grouped reference implementation. MPS uses native compute kernels; it
never reads routing IDs/counts back to the CPU.

Conversion supports both per-expert modules (Transformers 4.57.6) and stacked
gate/up/down parameters (repository-pinned 5.9.0). Tiny complete-decoder tests
pass all 20 cases under both versions. A full 5.9 conversion of the pinned
checkpoint produces exactly the same 2,777,820,161-byte model, configuration and
vocabulary as 4.57.6: model SHA256
`c7e2eca19888248bca647a9d2c3ac79956730a7ffd5c8057e7e7b42ca5848928`.
The second serialization was hashed as a stream to avoid storing another
multi-GB model. Evidence: `moe-transformers59-conversion-parity.json` and
`tests-moe-transformers{457,59}.log`. Upstream layout references:
[Qwen3 MoE 5.9](https://github.com/huggingface/transformers/blob/v5.9.0/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py),
[Mixtral 5.9](https://github.com/huggingface/transformers/blob/v5.9.0/src/transformers/models/mixtral/modeling_mixtral.py).

1. `moe_router` reads FP16/BF16/FP32 activations directly, retaining FP32 router
   weights, accumulation, and logits. The FP32 router copy is cached once while
   loading the layer. An initial implementation converted activations through
   `primitives<MPS>::convert`, which synchronizes and runs a host loop. That
   per-layer/per-token conversion has been removed.
2. `moe_topk_f32` selects unique experts and softmax-normalizes selected logits.
3. GPU count/prefix/group kernels generate expert offsets, a stable token/slot
   permutation, its inverse, and indirect dispatch grids. Counts remain on GPU.
4. Indirect expert projections fuse gate/up GEMMs and SwiGLU. Prefill uses a
   64x64x32 tile for qualifying FP16 batches, with the prior 16x32x16 tile
   as a fallback. Small batches (up to eight tokens)
   use a SIMD-reduction GEMV. Devices without Apple7 matrix support retain that
   generic GEMV path. Down projections also use the indirect grids.
5. GPU aggregation restores original token order, applying weights in ascending
   expert order with model-dtype rounding at the same operation boundaries as
   the source implementation. It uses no floating-point atomics. Empty experts
   receive no-op dispatches and never read their input/weights.

Expert weights remain in model storage, referenced for the model lifetime;
there is no per-token repacking, full-weight dequantization, or weight copy.
FP32 router copies and expert references belong to the layer. Intermediate
buffers use the existing asynchronous Metal allocator and command-buffer
retention rules. All buffers are registered before encoding. Workspace index
spans are checked before allocation. GPU grouping is stable and the inverse
mapping guarantees exactly one result for each selected token/expert slot.

The host encodes two indirect projection dispatches per expert. This avoids CPU
routing/synchronization but still has dispatch overhead, especially for models
with many empty experts. Large expert counts have not received end-to-end
performance qualification. Grouping and workspaces are currently rebuilt for
each layer invocation; further reuse/fusion should follow measured profiles.

## Routing semantics and compatibility

TopK is sorted by descending logit, then ascending expert index. NaNs rank below
numbers and receive zero weight. Selected positive infinities split weight
uniformly; all-negative-infinity/NaN rows receive zero weight. These cases match
the existing CPU operator. Ties can differ from source libraries whose TopK tie
order is unspecified, so source fixture export checks tie compatibility.

Stable maximum subtraction, precise exp/divide, Kahan summation, and compensated
denominator division preserve the existing 1e-7 weight gate. Kernel-local
reassociation is disabled so the compiler cannot erase compensation. Other
kernels retain their original math options.

Supported: normalized selected-softmax routing, SiLU/SwiGLU, unbiased projections,
CPU FP32, and native MPS FP32/FP16/BF16 expert primitives. Complete model tests
exercise CPU FP32 and MPS FP16 with generic and compact Flash attention.

Current extensions support full-expert-softmax routing weights, gated shared
experts, native INT4 matrices, and per-layer sliding-window semantics. Prefixes
remain resident for absolute window positions; bounded window cache eviction is
not implemented. Legacy ambiguous Qwen2 window schedules require explicit
`layer_types`. See the extension document for format and converter restrictions.

Still rejected: corrected/group-limited routing, biased routed expert projections,
prequantized AWQ/GPTQ expert formats, CUDA decoder MoE, and tensor parallel MoE.
Native INT4 requires FP16 activations; floating-point expert primitives retain
FP32/FP16/BF16 support. The following correctness/performance results describe the
original FP16 revision unless an extension result is explicitly identified.

## Correctness tests

`python/tests/test_moe_decoder.py` builds tiny deterministic source models locally
without downloads. Its 20 passing tests cover Qwen3 MoE TopK 1/2/4, mixed dense
and sparse layers, Mixtral, and dense Qwen3 regression. CPU and both MPS attention
paths compare full logits and four generated/cached steps against independent
Transformers FP32 outputs at odd lengths 7/33. Repeated calls validate cache reset
and returned logit snapshots. Mixed-length batches and four hypotheses exercise
beam cache expansion/reordering and preserve each hypothesis relative to single
requests. Unsupported routing and attention schedules fail explicitly.

Metal C++ tests compare mixed-storage router projection against independent
FP64 accumulation at odd and actual hidden widths (13/1024). Complete expert
execution is checked against independent FP64 projections with explicit
FP16/BF16 rounding, across 1/7/9/33/65 tokens, TopK 1/2/4, tail dimensions 13/17,
GEMM and GEMV paths, empty experts with NaN-poisoned weights, and uneven loads. Dispatch tests verify a
bijection over all token/slot assignments, stable grouping and inverse mapping.
Existing TopK tests cover widths 1/17/256/513, k=1/8/all, ties, nonfinite scores,
aliases, invalid arguments, empty rows, reused buffers and row permutations.

```bash
CT2_TEST_MPS=1 PYTHONPATH="$PWD/python" python -m pytest -q \
  python/tests/test_moe_decoder.py
build-mps/tests/ctranslate2_test tests/data \
  --gtest_filter='MPS/OpDeviceTest.MoeTopK*:MPSBackendTest.Moe*'
```

A separate result-handoff bug was fixed in `src/decoding.cc`: final hypothesis
sorting deep-copies GPU logit snapshots on the worker after the sampler's last
wait. Returning before those copies complete exposed zeros/stale data on another
thread. Only requests returning vocabulary logits wait at that final handoff;
ordinary generation gains no additional wait. The tiny model tests reproduce
this case on generic and Flash attention. See the NaN diagnosis report for its
distinction from the earlier QKV GEMM NaNs.

## Real model qualification and reproduction

Pinned checkpoint: `DavidAU/Qwen3-MOE-4x0.6B-2.4B-Writing-Thunder-V1.2`, snapshot
`f85830ed241787c392660f3e064050ccacac9602`. It has 28 layers, hidden width 1024,
expert width 3072, four experts, TopK two with normalization, 16 Q/8 KV heads,
head depth 128, and vocabulary 151936. Use measured tensor counts rather than
inferring memory from the repository's marketed parameter count.

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m ctranslate2.converters.transformers \
  --model /path/to/pinned/snapshot --output_dir /path/to/moe-ct2-f16 \
  --quantization float16 --low_cpu_mem_usage
python tools/benchmark/llm/reference_moe.py --snapshot /path/to/pinned/snapshot \
  --lengths 7 33 --output-tokens 4 --output /path/to/source.json
python tools/benchmark/llm/validate.py export --model /path/to/moe-ct2-f16 \
  --tokenizer /path/to/pinned/snapshot --device mps --compute-type float16 \
  --flash-attention --lengths 7 33 --output-tokens 4 --output /path/to/mps.json
python tools/benchmark/llm/validate.py compare /path/to/source.npz /path/to/mps.npz
```

`reference_moe.py` loads original Transformers decoder layers one at a time in
CPU FP32, independently of CT2 conversion/expert kernels. This keeps the
reference usable on the 8 GB M1; it is not a deployment fallback or timed CPU
baseline. Its output is itself tested against a complete tiny source model.
For large contexts use `--last-logits-only` and candidate `--cached-only` to avoid
retaining every vocabulary row. The CPU reference still checks incremental
logits against a separate full-sequence pass.

At 7/33 tokens, the final native-router implementation matches every source full
logit argmax and all four greedy generated tokens. FP16 full-logit maximum errors
are 0.0576/0.1763 with RMSE 0.0102/0.0081; cached errors are 0.0295/0.0698 with
RMSE 0.0060/0.0093. These pass the unchanged 0.2 maximum / 0.025 RMSE gates.
FP16 GPU cached-vs-GPU-full argmax agrees 3/4 at length 7 and 4/4 at length 33.
The length-7 third prediction is a near tie: GPU full selects 15578, whereas CPU
and GPU cached select 15878. This is a remaining exact-greedy qualification gap,
not a claim of complete numerical equivalence. Short synthetic-prompt perplexity
is 2244.1/198.47 on source FP32 versus 2277.35/197.66 on FP16; this is not a corpus
quality evaluation or an INT4 result.

Actual first-layer router fixtures (32/128/512/2048 tokens) also match all source
expert indices, with maximum normalized-weight error 5.960464e-8. At 2048 tokens
expert loads are 3/2046/2047/0, including an empty expert. Export with
`export_moe_router.py`; validate/timing with `benchmark_mps moe-router-file`.
These fixtures test selection on captured FP32 logits. Router projection is
tested separately against independent accumulation and complete model logits.
The fixture format is little-endian `CT2MOE1\0`, three uint32 dimensions, logits,
INT32 IDs and FP32 weights. The exporter manifest records prompts and hashes;
the benchmark rejects truncated/invalid fixtures and parity failures before timing.
Warm 200-sample selection wall latencies are 0.0164/0.0218/0.0670/0.0997 ms.
These include enqueue/amortized synchronization and exclude router projection,
grouping, expert execution, attention and sampling.

## End-to-end measurements, M1, 2026-11-11

FP16 weights/KV, same source/tokenizer/persisted prompt IDs, output length 32,
greedy TopK 1, TopP 1, temperature 1, seed 42, one worker, four CPU threads.
Model/tokenizer loading is excluded. Each shape has a first request and two warm
requests; table values are warm medians. Runs are sequential on the same machine.
TTFT includes prefill, first-token decode, sampling and callback delivery. Dividing
prompt length by TTFT is not an isolated prefill throughput measurement.

| Prompt | Router host-conversion baseline decode tok/s | Native router decode tok/s | Baseline TTFT s | Native TTFT s |
|---:|---:|---:|---:|---:|
| 32 | 17.04 | 20.89 | 0.4080 | 0.3846 |
| 128 | 15.96 | 21.01 | 0.9198 | 0.8461 |
| 512 | 14.96 | 19.14 | 3.6845 | 3.1054 |
| 2048 | 11.64 | 14.42 | 16.5099 | 15.1063 |

All 32 IDs are unchanged between the baseline and candidate, and between each
shape's cold/warm requests. The native router improves decode by 23–32% in this
final comparison. An earlier run measured 16–32%, with 512-token TTFT essentially
unchanged (3.701 s); this illustrates run-to-run variation. First-request TTFT is
1.103/0.486 s, but prior runs warmed persistent Metal compilation caches, so this
is not an isolated cold-start improvement claim. Peak process RSS is 732/980 MB; shared Metal mappings
make this unsuitable as a claim of total peak unified memory. Logical compact
FP16 KV costs 112 KiB/token, with capacity rounded to multiples of 512 tokens
(280 MiB at capacity 2560). Scratch storage and cached/aliased weights add memory.

Artifacts remain outside Git under `mps-llm-artifacts/`: source/model snapshots,
converted model, router fixtures, `moe-source-reference.*`,
`moe-native-router-short*`, `tests-moe-complete-decoders.log`,
`moe-router-baseline-timing.*`, and `moe-native-router-timing.*`. Baseline library
SHA256 is `c51a08ae62514dabeb1a923e830fefd05beb25960a7c0cd59a47494ea45c9e18`;
candidate is `f12678e91de8905302130399376834d678f05ef9bdfdff6121e88ac0d8d14b66`.
No speed advantage over llama.cpp is claimed from this before/after comparison.

Long-context generated-logit validation at 32/128/512/2048/2051 tokens matches
all four source CPU greedy IDs at every length, with finite logits throughout.
Maximum absolute cached-logit errors are 0.0647/0.0795/0.0630/0.0403/0.0512;
RMSE stays below 0.010. The independent CPU cached-vs-full passes have maximum
error below 0.00018 and identical argmax at all positions. Artifacts:
`moe-source-long.*`, `moe-native-router-long.*`, and
`moe-native-router-long-parity.json`. This checks cache growth and odd tails;
it does not resolve the separately recorded short FP16 near tie.

### Matched llama.cpp comparison

llama.cpp commit `bfb4308b058b334c6e68085c661ec9eb7e3d59f4` (build 9481),
the same source snapshot converted to F16 GGUF, F16 KV, all layers on Metal,
Flash attention enabled, one slot, context 4096 and four CPU threads. Expert
count/TopK/normalization match the source. The replay sends the exact persisted
CT2 prompt IDs and sampling settings, disables prompt caching and repeat/min-p
filters, and checks evaluated prompt lengths. Both backends produce identical
32-token sequences at all four lengths, across cold/warm requests.

| Prompt | CT2 decode tok/s | llama.cpp decode tok/s | CT2 TTFT s | llama.cpp TTFT s |
|---:|---:|---:|---:|---:|
| 32 | 20.89 | 27.31 | 0.3846 | 0.1547 |
| 128 | 21.01 | 29.39 | 0.8461 | 0.1470 |
| 512 | 19.14 | 25.31 | 3.1054 | 0.5339 |
| 2048 | 14.42 | 25.90 | 15.1063 | 2.3963 |

llama.cpp is faster here, especially at prefill. Its isolated server prompt
throughput is 289/879/977/868 tokens/s. CT2's table reports TTFT, not isolated
prefill. HTTP/SSE delivery is included in llama.cpp TTFT; CT2 includes Python
callbacks. Neither first request is assumed free of compilation/page-in costs.
The native-router speedup above is an internal CT2 improvement, not a win over
llama.cpp. Prefill scheduling, materialized attention, expert tiling and dispatch
count remain optimization targets. Dedicated per-kernel GPU latency/utilization
and total peak unified-memory instrumentation are still needed; cumulative
profile counters are not substitutes for those measurements.

Reproducible files: `convert-moe-gguf.log`, `llama-moe-server.log`,
`llama-moe-timing.*` and `moe-native-router-timing.*`. Models/builds are outside
Git, and the local benchmark server is stopped after measurement.

### Final regression and synchronization checks

The MPS C++ suite passes 246 tests (12 unrelated skips); CPU/shared passes 369
(16 skips). The CPU-only build succeeds. Python complete-model, sampling,
storage/translation and metric checks pass 94 tests (five skips). The pinned
dense-model repeated-prefill suite passes six additional tests; the two existing
mixed-batch Flash near-tie cases remain explicit expected failures. Seeded MoE
sampling is repeated in fresh processes with temperature 0.6, TopK 10 and TopP
0.7; selected tokens also satisfy independently computed filtering semantics.

A warm real-model eight-token trace reports no host conversions, no tensor CPU
fallbacks, GPU router/TopK/grouping/expert execution, and only eight four-byte
sampled-ID readbacks. Initial loading still performs 28 router conversions once;
first use builds 56 per-layer RoPE tables. These are not per-token expert
fallbacks. The profile counts and command-buffer GPU-time sums are diagnostic,
not isolated routing/expert latency measurements. Reproduction trace:
`moe-native-router-warm-profile.log`.

The private real Whisper beam-five smoke retains the exact token hash over four
runs (warm median 6.512 s versus 6.529 s before MoE integration). All 32 Marian
beam/batch/length/repeat combinations retain their previous MPS hypotheses.
Returned Marian single-item raw-logit snapshots now agree with CPU argmax at all
four checked steps; suppressed/padded entries are excluded from numerical error
comparisons. Files: `whisper-moe-regression.*`, `marian-moe-regression.*`.
These smoke tests do not replace corpus WER/BLEU qualification.
