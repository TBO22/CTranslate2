# T5Gemma Intel throughput investigation

Diagnostic tooling for OpenNMT/CTranslate2 #2175, not a proposed runtime change.
Compare the exact Linux wheel from run 45992420951 (commit 40f8e6a) with 4.9.1
on one Intel host. Both processes use one FP32 model converted by 4.9.1 and the
same saved token IDs from the release gate's pinned SQuAD selection. This isolates
runtime changes from converter changes. The first run uses eight examples.
The manual inputs can select all 50 gate examples, one precision, packed MKL
alone and natural stopping alone for a focused follow-up. Defaults retain the
complete eight-example backend and fixed-length matrix.

The compared Python wheels both report version 4.9.1. Their native-library hashes
and the exact candidate Actions run identify them; do not use the version string
to conclude that the binaries are identical.

The two persistent processes alternate in measurement order and use two CPU
threads each. Both `int8_float32` and FP32 controls run. Modes are ordinary EOS stopping and exactly 64 decoder outputs
with EOS disabled. Backends are packed MKL, unpacked MKL and oneDNN, selected in
fresh processes because CT2 caches backend settings. Each result includes full
tokens, returned counts, request time, encoder time, decoder time, decoder call
count, active row steps and maximum decoder position. Fixed-work counters must
match between base and candidate or the run fails.

The Linux ELF timing hooks interpose existing encoder and decoder functions and,
when requested, Dense, attention MatMul, softmax, mask, Concat and Tile. Their exact symbols
are verified against both wheel libraries before use. Only entry/exit timing and
input dimensions are recorded; numerical operations are unchanged. Missing hooks
or overlapping times fail the run. Uninstrumented controls (fixed-length when
requested, otherwise natural stopping) must
produce identical token sequences and retain their own timing records. These
controls are not mixed with phase timings. The hook is Linux x86-64 diagnostic
code, not a portable CT2 API or a production profiling implementation.

Operator probes also record process minor and major faults around each operator.
Only one inference request runs in each worker process. Counts include its native
threads and describe cumulative faults, not peak memory. Encoder and decoder
timers include the rest of their operations; the probes check for missing hooks
and preserve full token outputs for the uninstrumented comparison.

The `heap_reuse` input selects diagnostic allocator controls. `true` fixes glibc
mmap and trim thresholds to 256 MiB only in worker environments. It does not
guarantee all allocations avoid mmap. `legacy-allocation` restores an unused
full-coverage length buffer's data allocation, fill and lifetime in the candidate
without changing corrected scores or softmax. This is an ablation, not proposed
production code. It does not reconstruct the old shape-vector allocation.
`candidate-ablation` uses the candidate binary for both labels and toggles only
that buffer allocation on the label called candidate. Its records explicitly
identify this comparison and require identical token output between conditions.

## Current execution status

Linux timing, operator and page-fault probes have completed on Intel. Six paired
packed-MKL fixed-work measurements reproduced an 11.02% throughput drop; the new
mask itself took only 4.196 ms for the complete request. A further copy breakdown
localized most extra time to Tile and Concat. The default page-fault probe
records about 16.3 million minor faults in baseline versus 19.4 million in PR.
The exact allocation-history mechanism is still under investigation. These
results do not clear the originally reported 30 to 38 percent Intel regression.
The standalone 50-example natural-stopping run and focused profiles completed;
the initial full backend matrices were interrupted deliberately to save partial
results before timeout and must not be reported as completed CI qualification.

The Intel hardware check fails early on an AMD runner. Explicitly selecting
`require_intel=false` permits a labeled same-host backend control on another CPU;
it can help isolate MKL dependence but does not qualify the Intel issue. The
workflow is manually dispatched and supports the Ubuntu 22.04 and 24.04 pools.
A successful AMD run is not evidence that the Intel issue is fixed. All raw data, wheel provenance,
dependency versions and hardware information are uploaded. No PR is merged or
closed by this workflow.
