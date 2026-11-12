# Decoder-only MPS validation and benchmarks

Use a native arm64 Release build, an idle Mac, and separate processes for CPU,
MPS, Transformers, and llama.cpp. Never time competing model processes. Preserve
the source snapshot revision and conversion settings. Do not use debug GEMM or
synchronization logging for reportable timing; collect a separate profile.

The default backend bypasses large FP16 MPSMatrix dense projections using the
native SIMD-group GEMM (or generic tiled Metal for unsupported native layouts).
This fixes the reproduced repeated long-prefill NaNs without tuning switches.
See [the investigation and validation](../../../docs/mps_llm_nan_diagnosis.md).
The opt-in real-model regression deliberately clears experimental GEMM/attention
switches so they cannot hide a broken default path.

Native split-K incremental attention and compact GQA/MQA KV caches are available
with `Generator(..., flash_attention=True)` or the tools' `--flash-attention` flag.
The shared standard-layout decoder hook also serves Whisper and Marian. See
[execution paths, validation and limitations](../../../docs/mps_flash_attention.md).
Supported compact prefills now use native tiled attention; see
[prefill/INT4/shared/window details](../../../docs/mps_prefill_int4.md). The older
`CT2_MPS_USE_STREAMING_ATTENTION=prefill` experiment is not qualified: it failed
the chosen model-logit maximum-error gate. Native INT4 and complete Qwen MoE execution are available with the restrictions
and validation described in that document.

`benchmark.py` writes prompt token IDs, effective compute type, sampling options,
module path, runtime controls, cold/first-per-shape and warm results, TTFT,
individual decode intervals, and process peak RSS. Its prompt-tokens/TTFT metric
includes the first decode and sampling; it is **not isolated prefill throughput**.
Process RSS is **not total peak unified memory**. KV memory is a geometry estimate;
the generic layer replicates GQA heads, while `--flash-attention` uses compact
KV heads with capacity rounded to 512-token blocks. No GPU utilization or
per-kernel timing is inferred from wall time.

Use `--validate-logits` for repeated-prefill correctness runs. This captures every
generated vocabulary logit, rejects nonfinite values and cold/warm greedy token
differences, and saves the failing run before exiting. Its extra GPU copies make
its timing unsuitable for comparison with ordinary timing runs. The opt-in model
regression runs short and 2048-token prompts without downloading models:

```bash
CT2_TEST_MPS=1 CT2_MPS_LLM_MODEL=/path/to/ct2-model \
CT2_MPS_LLM_SOURCE=/path/to/hf-snapshot \
python -m pytest -q python/tests/test_mps_llm_real.py
```

```bash
export PYTHONPATH="$PWD/python"
PYTHON=/path/to/python
MODEL=/path/to/ct2-qwen3-f16
SOURCE=/path/to/pinned/huggingface/snapshot

# Run logits/greedy/cache validation before timing; each export exits before the next.
$PYTHON tools/benchmark/llm/validate.py export --model "$MODEL" \
  --tokenizer "$SOURCE" --device cpu --compute-type float32 --output cpu.json
$PYTHON tools/benchmark/llm/validate.py export --model "$MODEL" \
  --tokenizer "$SOURCE" --device mps --compute-type float16 --output mps.json
$PYTHON tools/benchmark/llm/validate.py compare cpu.npz mps.npz --output parity.json

# Transformers checks the converter independently. It exports full-sequence logits.
$PYTHON tools/benchmark/llm/validate.py export --backend transformers \
  --model "$SOURCE" --tokenizer "$SOURCE" --output transformers.json

$PYTHON tools/benchmark/llm/benchmark.py --model "$MODEL" --tokenizer "$SOURCE" \
  --device mps --compute-type float16 --flash-attention --lengths 32 128 512 2048 \
  --output-tokens 32 --repeats 3 --output mps-timing.json

# Use a fresh process and a short request for shape/synchronization diagnostics.
CT2_MPS_PROFILE=1 CT2_MPS_LOG_GEMM=1 CT2_MPS_LOG_SYNC=1 \
$PYTHON tools/benchmark/llm/benchmark.py --model "$MODEL" --tokenizer "$SOURCE" \
  --device mps --lengths 32 --output-tokens 4 --repeats 1 \
  --output mps-profile.json 2>mps-profile.log
```

`validate.py` exports unscaled full-vocabulary logits, next-token perplexity on
the deterministic input, generated greedy IDs, and cached-step/full-sequence logit
differences for odd lengths. These short prompts are a regression smoke check,
not a corpus quality evaluation. Select tolerances before reviewing candidates;
defaults are max absolute logit error 0.2 and RMSE 0.025. Every value must be finite.
Greedy IDs must match exactly when comparing CPU and MPS exports. Inspect near-ties
if this fails; do not hide it by widening tolerances. Sampling must also be checked
with the existing primitive and decoding tests. Cross-backend seeded RNG identity
is not assumed.

For llama.cpp, convert the **same snapshot** to F16 GGUF, use F16 K/V, one slot,
and all layers on Metal. Do not compare AWQ to GGUF Q4 as if their quantization
schemes were identical. Retain build commit, server log, configuration and GGUF
conversion log. Start the server only after CT2 has exited:

```bash
llama-server -m qwen3-f16.gguf -ngl 99 -c 4096 -np 1 -ctk f16 -ctv f16 \
  --host 127.0.0.1 --port 18081 --no-warmup
$PYTHON tools/benchmark/llm/llama_server.py --prompts mps-timing.prompts.json \
  --output-tokens 32 --repeats 3 --output llama-timing.json
```

The replay disables prompt reuse, min-p and repetition penalties and sends the
exact CT2 token IDs. It rejects implicit prompt length changes and incomplete
outputs. TTFT includes localhost HTTP/SSE overhead; CT2 includes Python callback
overhead. llama.cpp server prompt throughput is recorded separately in the server
timings. Inspect output IDs before interpreting timing differences. Increasing
lengths beyond 2048 requires checking available unified memory first.

Complete normalized Qwen3 MoE/full-attention Mixtral conversion and execution,
the native GPU router/dispatch/expert kernels, and real source-reference checks
are described in [MPS decoder MoE](../../../docs/mps_moe_routing.md).
`reference_moe.py` provides an independent memory-bounded CPU FP32 source
reference. For long contexts use its `--last-logits-only` option and
`validate.py export --cached-only` for the candidate, then compare the NPZ files.
INT4 experts, shared Qwen2 experts, and explicit sliding-window MoE schedules
are covered by the extension tests. Window cache eviction and broad INT4 quality
qualification remain outstanding.
