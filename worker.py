"""Persistent single-worker CPU benchmark; stdout is a JSON protocol."""

import argparse
import ctypes
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

import ctranslate2


class Counters(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in (
        "encoder_ns", "decoder_ns", "encoder_calls", "decoder_calls",
        "decoder_row_steps", "max_step_plus_one",
    )]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--compute-type", default="int8_float32")
    parser.add_argument("--timers")
    parser.add_argument("--operators", action="store_true")
    parser.add_argument("--length", type=int, default=64)
    args = parser.parse_args()
    timers = ctypes.CDLL(args.timers) if args.timers else None
    if timers:
        timers.diag_read.argtypes = [ctypes.POINTER(Counters)]
        timers.diag_read.restype = None
        timers.diag_reset.argtypes = []
        timers.diag_reset.restype = None
        if args.operators:
            timers.diag_read_operators.argtypes = [ctypes.POINTER(ctypes.c_uint64)]
            timers.diag_read_operators.restype = None
    inputs = json.loads(Path(args.inputs).read_text())["tokens"]
    translator = ctranslate2.Translator(
        args.model, device="cpu", compute_type=args.compute_type,
        intra_threads=2, inter_threads=1,
    )
    if translator.compute_type != args.compute_type:
        raise RuntimeError(f"Unexpected compute type: {translator.compute_type}")
    worker_environment = {k: os.environ.get(k) for k in (
                          "CT2_USE_MKL", "CT2_PACKED_GEMM", "OMP_NUM_THREADS",
                          "MKL_NUM_THREADS", "CT2_FORCE_CPU_ISA",
                          "MALLOC_MMAP_THRESHOLD_", "MALLOC_TRIM_THRESHOLD_",
                          "CT2_DIAG_LEGACY_ALLOCATION", "CT2_DIAG_LIBRARY",
                          "DIAG_CANDIDATE_RUN")}
    print(json.dumps({"ready": True, "version": ctranslate2.__version__,
                      "compute_type": translator.compute_type,
                      "environment": worker_environment}), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        if request.get("stop"):
            break
        fixed = request["mode"] == "fixed"
        options = dict(beam_size=2, sampling_topk=1, max_batch_size=8,
                       max_decoding_length=args.length, return_scores=False)
        if fixed:
            options.update(min_decoding_length=args.length, end_token=[])
        if timers:
            timers.diag_reset()
        usage_before = resource.getrusage(resource.RUSAGE_SELF)
        start = time.perf_counter_ns()
        outputs = translator.translate_batch(inputs, **options)
        wall_ns = time.perf_counter_ns() - start
        usage_after = resource.getrusage(resource.RUSAGE_SELF)
        counts = Counters()
        if timers:
            timers.diag_read(ctypes.byref(counts))
            if not counts.encoder_calls or not counts.decoder_calls:
                raise RuntimeError("Timing hooks did not intercept CT2; refusing empty timings")
        hypotheses = [result.hypotheses[0] for result in outputs]
        lengths = list(map(len, hypotheses))
        if fixed and lengths != [args.length] * len(inputs):
            raise RuntimeError(f"Fixed-length contract violated: {lengths}")
        tokens = sum(lengths)
        record = dict(request, environment=worker_environment, wall_ns=wall_ns, output_tokens=tokens,
                      tokens_per_second=tokens * 1e9 / wall_ns,
                      output_lengths=lengths, hypotheses=hypotheses,
                      scores=[r.scores for r in outputs],
                      output_sha256=hashlib.sha256(json.dumps(hypotheses).encode()).hexdigest(),
                      timers_enabled=bool(timers))
        record["resource_usage"] = dict(
            minor_faults=usage_after.ru_minflt - usage_before.ru_minflt,
            major_faults=usage_after.ru_majflt - usage_before.ru_majflt,
            user_cpu_seconds=usage_after.ru_utime - usage_before.ru_utime,
            system_cpu_seconds=usage_after.ru_stime - usage_before.ru_stime,
            voluntary_switches=usage_after.ru_nvcsw - usage_before.ru_nvcsw,
            involuntary_switches=usage_after.ru_nivcsw - usage_before.ru_nivcsw)
        record.update({name: getattr(counts, name) for name, _ in Counters._fields_})
        if args.operators:
            values = (ctypes.c_uint64 * 48)()
            timers.diag_read_operators(values)
            names = [f"{phase}_{op}_{unit}" for unit in ("ns", "calls", "minor_faults", "major_faults")
                     for phase in ("encoder", "decoder")
                     for op in ("dense", "matmul", "softmax", "mask", "concat", "tile")]
            record["operators"] = dict(zip(names, values))
            for op in ("dense", "matmul", "softmax"):
                if not record["operators"][f"decoder_{op}_calls"]:
                    raise RuntimeError(f"The decoder {op} hook did not intercept CT2")
        if timers:
            record["other_ns"] = wall_ns - counts.encoder_ns - counts.decoder_ns
            if record["other_ns"] < 0:
                raise RuntimeError("Overlapping phase timers are invalid for this benchmark")
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
