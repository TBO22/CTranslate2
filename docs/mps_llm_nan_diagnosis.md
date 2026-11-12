# Repeated long-prefill NaN investigation

Status, 2026-11-11: **the repeated-prefill NaN reproduction is fixed in the default
path** by bypassing the affected large FP16 MPSMatrix projections. The underlying
defect inside that execution path is not established; this report does not claim
an Apple driver bug or a proven buffer race.

## Implemented fix

`mps::gemm` automatically uses the existing native SIMD-group kernel for
FP16 `A @ B.T` projections with M, N and K at least 1,024 and beta zero. The
kernel accumulates in FP32, explicitly initializes its accumulators and pads
invalid tile lanes with zero. This removes the failing MPSMatrix operation from
long dense prefill without CPU waits, whole-weight conversions or a global
queue-serialization requirement. No environment switch is required.

Native selection checks Apple GPU family support, 16-byte input/row alignment
and complete 32-bit index spans. Unsupported layouts use the generic tiled
Metal GEMM rather than re-entering the affected MPSMatrix route. Small matrices,
decoding GEMV, attention matmuls and non-FP16 configurations retain their existing
selection. Automatic selection does not fuse bias or activation epilogues.

Validation on the local M1:

- The new real-model regression fails against the original `7cadad48` library
  with nonfinite logits, then passes with the fixed default library.
- Six consecutive 2,048-token requests generating 32 tokens have entirely finite
  logits and bit-for-bit identical logits between cold and repeated requests.
- Compared with the original library's finite first request, the fixed logits
  have maximum absolute difference 0.109375, RMSE 0.00955, and 100% argmax agreement.
- MPS C++: 232 passed, 15 skipped. CPU/shared C++: 369 passed, 16 skipped.
- Python real-model, storage and translation tests: 75 passed, 5 skipped.
- Native and generic-fallback tests cover odd M/N/K, padded leading dimensions,
  misaligned input views, poisoned output/padding, and output-boundary canaries.

These are correctness results, not a claim of improved speed or qualification
on other Apple GPU generations. The original diagnostic history follows.

## Confirmed failure

Qwen3-0.6B, pinned snapshot
`c1899de289a04d12100db370d81485cdf75e47ca`, converted to CT2 FP16, on an M1
7-core GPU with 8 GB unified memory and macOS 27.0.1:

1. Load one generator, one worker, with no experimental MPS flags.
2. Generate 32 greedy tokens from the exact same 2,048-token prompt three times.
3. The first request is finite. Subsequent requests produce nonfinite logits
   and token ID 0 throughout.

This reproduces with the untouched final-PR library at `7cadad48`. The new
streaming-attention kernel is disabled. The initial prefill processes 2,047
tokens, followed by an incremental step for the last prompt token.

The first observed corruption is in an ordinary dense QKV projection:

```
C[2047,4096] = A[2047,1024] @ B[4096,1024].T
dtype = FP16; alpha = 1; beta = 0
lda = ldb = 1024; ldc = 4096
```

The call chain is `Dense::operator()` -> `ops::Gemm` -> `mps::gemm` ->
`MPSMatrixMultiplication::encodeToCommandBuffer`. The concrete encode is in
`src/mps/kernels.mm`. This projection precedes the layer's RoPE, QK product,
softmax, and attention/value product.

## GPU-side capture

Temporary diagnostics scan FP16 exponent bits on the GPU immediately before
and after each matching QKV GEMM. They copy the first failing A/B/C to separate
buffers on the GPU. The host reads those copies only at an existing stream wait;
there is no added CPU wait before GEMM. These probes change dispatch boundaries,
so the exact layer/row of the first failure can change.

The captured failure in `mps-llm-artifacts/qkv-diag` shows:

- QKV invocation 54: second request, zero-based layer 26.
- A and B contain no nonfinite values before GEMM or in the captured copies.
- A and B are bit-for-bit identical to the earlier successful capture.
- A, B and C use distinct MTLBuffers, all at offset zero. There is no direct
  overlap between these three operand ranges in this capture.
- C contains 2,097,152 NaNs: every even column of rows 0 through 1,023.
  The remaining rows are finite. Earlier, lighter probes first detected a bad
  element at row 1,536 instead. The position is not a fixed-layer invariant.
- CPU FP32 replay of those exact A/B values has maximum absolute output
  106.5778. The maximum sum of absolute products is 502.1133, below the FP16
  finite limit of 65,504. Ordinary dot-product overflow is excluded for these
  operands.

`qkv-diag.log`, `qkv-diag/flags.bin`, `qkv-diag/pointers.txt`, and the three
operand binaries preserve the evidence. They are local artifacts outside Git.
The temporary instrumented sources are in `qkv-diag-source`; the branch's
production sources have been restored after capture.

## Controlled exclusions

| Experiment | Result | What it establishes |
| --- | --- | --- |
| Untouched final-PR library, experimental attention disabled | Requests 2 and 3 fail | The new online softmax is not responsible for this reproduction |
| Replay earlier good captured operands, 140 queued GEMMs with reused outputs | All finite | A standalone GEMM with these operands does not reproduce the model failure |
| Replay those operands, 140 GEMMs with fresh output allocations | All finite | Simple output allocation churn alone is insufficient |
| Replay the exact operands captured at the failing model GEMM, 140 GEMMs | All finite | Failure does not follow the numerical operand data |
| GPU-zero the model QKV output before MPS GEMM | Still fails; first failure at layer 24, row 1,024 | Zeroing caller-owned C is insufficient; this is not just untouched NaNs in C |
| Disable MPS internal tiling for matching QKV GEMMs | Still fails | That option is insufficient; it does not prove the internal implementation is blameless |

Previous diagnostic runs also found that restricting the queue to one in-flight
command buffer, or padding large dense row counts to full 512-row blocks,
suppressed the failure in the tested repetitions. Those interventions alter
scheduling, allocation and/or kernel selection together, so they are workarounds,
not root-cause proof. The opt-in SIMD-group GEMM route remained finite for the
tested real-model runs. That evidence motivated the default routing fix above;
the wider medium-M tuning range remains opt-in.

## Memory and remaining uncertainty

Backend buffers use `MTLResourceStorageModeShared`: CPU and GPU share their
backing allocations. That fact alone is not evidence of an invalid CPU access.
The backend records buffer use, retains resources through command completion,
and defers destruction; the relevant code is `src/mps/utils.mm` and
`src/mps/allocator.mm`. No violating access has been demonstrated yet.

The remaining distinction is between an execution/lifetime error in CT2's MPS
integration and state/scratch behavior within the MPS GEMM implementation under
the full-model command history. Identifying a failing encode is not enough to
choose between them. The default routing fix has a real-model regression that
fails before and passes after. It avoids the failing operation, rather than
claiming to repair its unproven internal cause.

The experimental attention path's maximum-logit tolerance failure is a separate
numerical-validation issue. It must not be conflated with this default-path NaN
failure or described as a validated speedup.

## Reproduction artifacts

From the local clean clone, using its fresh Python extension and the pinned
model, the saved helper reproduces the original failure:

```bash
PYTHONPATH="$PWD/python" \
DYLD_LIBRARY_PATH="$PWD/../mps-llm-artifacts/baseline" \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/path/to/ct2-python ../mps-llm-artifacts/repro_long.py \
  mps float16 pinpoint-baseline 3
```

`mps-llm-artifacts/isolate_qkv_gemm.cc` is the standalone CT2 GEMM replay. It
reads `input.bin` and `weight.bin`, submits 28 projections per round, waits, and
scans every output for nonfinite values. `isolate-qkv-failed-input.log` records
five passing rounds with the exact failed-model operands.

## Independent returned-logit snapshot issue, fixed 2026-11-11

Tiny decoder tests exposed zero/stale early `return_logits_vocab` snapshots on
MPS, even though full forward logits and generated token IDs were correct. This
also explains the unreliable Marian single-request raw-logit diagnostic noted
in the FlashAttention report. This issue is separate from the nonfinite QKV
GEMM reproduction above: it involved the lifetime/handoff of copied results,
not router arithmetic, shared threadgroup memory, or online softmax.

`sort_hypotheses` in `src/decoding.cc` calls `index_vector` on vocabulary logits.
That deep-copies StorageViews on the worker's thread-local Metal stream, after
the sampler's final GPU wait. The future could return before those final copies
were submitted/completed. A caller's CPU readback waits on its own stream and
cannot complete the worker's unsubmitted copies.

The fix completes that worker stream after final hypothesis logit reordering,
only when vocabulary logits are returned on MPS. Generation without logit
capture has no additional synchronization. CPU/CUDA behavior is unchanged.
`python/tests/test_moe_decoder.py` compares every full/cached vocabulary logit
against independent tiny Transformers models on CPU, generic MPS and compact
Flash MPS, with repeated requests and beams. Before the fix early snapshots were
all zero (maximum error about 0.493); afterwards typical tiny FP16 errors were
below 0.0005. This is a regression for result delivery as well as model math.
