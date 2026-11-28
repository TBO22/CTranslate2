# T5Gemma Intel throughput investigation

Diagnostic tooling for OpenNMT/CTranslate2 #2175, not a proposed runtime change.
Compare the exact Linux wheel from run 45992420951 (commit 40f8e6a) with 4.9.1
on one Intel host. Both processes use one FP32 model converted by 4.9.1 and the
same saved token IDs from the release gate's pinned SQuAD selection. This isolates
runtime changes from converter changes. The first run uses eight examples.

The two persistent processes alternate in measurement order and use two CPU
threads each. Modes are ordinary EOS stopping and exactly 64 decoder outputs
with EOS disabled. Backends are packed MKL, unpacked MKL and oneDNN, selected in
fresh processes because CT2 caches backend settings. Each result includes full
tokens, returned counts, request time, encoder time, decoder time, decoder call
count, active row steps and maximum decoder position. Fixed-work counters must
match between base and candidate or the run fails.

The Linux ELF timing hooks interpose two existing functions. Their exact symbols
are verified against both wheel libraries before use. Only entry/exit timing and
input dimensions are recorded; numerical operations are unchanged. Missing hooks
or overlapping times fail the run. Uninstrumented fixed-length controls must
produce identical token sequences and retain their own timing records. These
controls are not mixed with phase timings. The hook is Linux x86-64 diagnostic
code, not a portable CT2 API or a production profiling implementation.

The Intel hardware check fails early on an AMD runner. A successful AMD run is
not evidence that the Intel issue is fixed. All raw data, wheel provenance,
dependency versions and hardware information are uploaded. No PR is merged or
closed by this workflow.
