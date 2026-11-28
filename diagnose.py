"""Controlled base/PR measurements with one model and Linux ELF phase hooks."""

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

MODEL = "jordimas/t5gemma-2-270m-270m"
REVISION = "dfb6d0d619d8e6d42f279a6197abb90dec0137c1"
SQUAD = "7b6d24c440a36b6815f21b70d25016731768db1f"
HERE = Path(__file__).resolve().parent


def prepare(args):
    import transformers
    from datasets import load_dataset
    from ctranslate2.converters.transformers import TransformersConverter
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = load_dataset("rajpurkar/squad", split="validation", revision=SQUAD)
    data = data.shuffle(seed=42)[:args.limit]
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    ids = [tokenizer.encode(f"Context: {context} Question: {question} Answer:",
                            truncation=True, max_length=512)
           for context, question in zip(data["context"], data["question"])]
    payload = {"model": MODEL, "revision": REVISION, "squad_revision": SQUAD,
               "ids": ids, "tokens": [tokenizer.convert_ids_to_tokens(row) for row in ids]}
    (out / "inputs.json").write_text(json.dumps(payload) + "\n")
    TransformersConverter(MODEL, revision=REVISION).convert(
        str(out / "model"), force=True, quantization="float32")
    with open(out / "model/model.bin", "rb") as model_file:
        model_sha256 = hashlib.file_digest(model_file, "sha256").hexdigest()
    manifest = {"model_sha256": model_sha256,
                "inputs_sha256": hashlib.sha256((out / "inputs.json").read_bytes()).hexdigest(),
                "prompt_lengths": list(map(len, ids)), "conversion_version": __import__("ctranslate2").__version__}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def library(python):
    script = ("import ctranslate2,pathlib; p=pathlib.Path(ctranslate2.__file__).resolve().parent.parent; "
              "x=list(p.glob('ctranslate2.libs/libctranslate2*.so*')); "
              "assert len(x)==1,x; print(x[0])")
    return subprocess.check_output([python, "-c", script], text=True).strip()


def symbols(path):
    raw = subprocess.check_output(["nm", "-D", "--defined-only", path], text=True)
    names = [line.split()[-1] for line in raw.splitlines() if len(line.split()) >= 3]
    decoded = subprocess.check_output(["c++filt"], input="\n".join(names), text=True).splitlines()
    prefixes = {"ENCODER_SYMBOL": "ctranslate2::layers::TransformerEncoder::operator()(std::vector<",
                "DECODER_SYMBOL": "ctranslate2::layers::TransformerDecoder::decode(",
                "DENSE_SYMBOL": "ctranslate2::layers::Dense::operator()(",
                "MATMUL_SYMBOL": "ctranslate2::ops::MatMul::operator()(",
                "CONCAT_SYMBOL": "ctranslate2::ops::Concat::operator()(",
                "TILE_SYMBOL": "ctranslate2::ops::Tile::operator()(ctranslate2::StorageView const&,",
                "MERGED_SYMBOL": "ctranslate2::layers::MultiHeadAttention::forward_merged(",
                "ALLOCATOR_SYMBOL": "ctranslate2::get_allocator(ctranslate2::Device)",
                "SOFTMAX_SYMBOL": "ctranslate2::ops::SoftMax::operator()(ctranslate2::StorageView const&, ctranslate2::StorageView const*,",
                "MASK_SYMBOL": "void ctranslate2::primitives<(ctranslate2::Device)0>::mask_merged_attention<float>("}
    result = {}
    for key, prefix in prefixes.items():
        matches = [name for name, display in zip(names, decoded) if display.startswith(prefix)]
        if key == "MASK_SYMBOL" and not matches:
            continue  # The released baseline predates this primitive.
        if len(matches) != 1:
            raise RuntimeError(f"Expected one {key} in {path}, got {matches}")
        result[key] = matches[0]
    return result


def inspect(args):
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    metadata = {}
    for label, python in (("baseline", args.baseline_python), ("candidate", args.candidate_python)):
        lib = library(python)
        script = '''import ctypes, json, sys, ctranslate2
library = ctypes.CDLL(sys.argv[1])
result = {"ctranslate2_version": ctranslate2.__version__, "mkl_version": None}
for name in ("MKL_Get_Version_String", "mkl_get_version_string"):
    try:
        call = getattr(library, name)
    except AttributeError:
        continue
    call.argtypes = [ctypes.c_char_p, ctypes.c_int]
    call.restype = None
    buffer = ctypes.create_string_buffer(512)
    call(buffer, len(buffer))
    result["mkl_version"] = buffer.value.decode()
    break
print(json.dumps(result))
'''
        record = json.loads(subprocess.check_output([python, "-c", script, lib], text=True))
        record["dependencies"] = subprocess.check_output(["ldd", lib], text=True)
        compiler = subprocess.run(["readelf", "-p", ".comment", lib], text=True, capture_output=True)
        record["compiler_comments"] = compiler.stdout
        with open(lib, "rb") as stream:
            record["library_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
        metadata[label] = record
    (out / "native-provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


def build(args):
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    libs = {"baseline": library(args.baseline_python), "candidate": library(args.candidate_python)}
    base, candidate = symbols(libs["baseline"]), symbols(libs["candidate"])
    if any(candidate.get(key) != value for key, value in base.items()):
        raise RuntimeError("Phase-hook ABI differs between these wheels")
    if "MASK_SYMBOL" not in candidate:
        raise RuntimeError("Candidate CPU mask symbol is missing")
    (out / "symbols.h").write_text("".join(f'#define {key} "{value}"\n' for key, value in candidate.items()))
    hook = out / "timing_hooks.so"
    subprocess.check_call(["g++", "-O2", "-std=c++17", "-fPIC", "-shared", "-I", str(out),
                           "-I", str(Path(args.headers).resolve() / "include"),
                           str(HERE / "timing_hooks.cc"), "-ldl", "-o", str(hook)])
    exported = subprocess.check_output(["nm", "-D", "--defined-only", str(hook)], text=True)
    for key, value in candidate.items():
        if key == "ALLOCATOR_SYMBOL":
            continue  # Looked up in CT2; deliberately not interposed by the probe.
        if value not in exported:
            raise RuntimeError(f"Hook does not export the exact {key}")
    (out / "libraries.json").write_text(json.dumps(libs, indent=2) + "\n")


class Worker:
    def __init__(self, python, label, backend, args, hooks=True, compute="int8_float32"):
        out = Path(args.out).resolve()
        libs = json.loads((out / "libraries.json").read_text())
        env = dict(os.environ, CT2_USE_MKL="0" if backend == "dnnl" else "1",
                   CT2_PACKED_GEMM="1" if backend == "mkl-packed" else "0",
                   CT2_VERBOSE="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                   OPENBLAS_NUM_THREADS="2", TOKENIZERS_PARALLELISM="false")
        if args.heap_reuse:
            env.update(MALLOC_MMAP_THRESHOLD_="268435456", MALLOC_TRIM_THRESHOLD_="268435456")
        command = [python, "-u", str(HERE / "worker.py"), "--model", str(out / "model"),
                   "--inputs", str(out / "inputs.json"), "--length", str(args.length),
                   "--compute-type", compute]
        if hooks:
            library_label = "candidate" if args.candidate_ablation else label
            env.update(LD_PRELOAD=str(out / "timing_hooks.so"), CT2_DIAG_LIBRARY=libs[library_label])
            command += ["--timers", str(out / "timing_hooks.so")]
            if args.operators:
                env["CT2_DIAG_OPERATORS"] = "1"
                command += ["--operators"]
            if (args.legacy_allocation or args.candidate_ablation) and label == "candidate":
                env["CT2_DIAG_LEGACY_ALLOCATION"] = "1"
        self.log_path = out / f"{backend}-{compute}-{label}-hooks{int(hooks)}.log"
        self.log = open(self.log_path, "w")
        self.process = subprocess.Popen(command, env=env, stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=self.log, text=True)
        self.ready = self.read()
        if not self.ready.get("ready"):
            raise RuntimeError(self.ready)
        expected = ("DNNL (packed: false)" if backend == "dnnl" else
                    f"MKL (packed: {'true' if backend == 'mkl-packed' else 'false'})")
        if f"SGEMM backend: {expected}" not in self.log_path.read_text():
            self.process.terminate()
            self.process.wait(timeout=30)
            raise RuntimeError(f"Expected backend {expected}; see {self.log_path}")
        self.label = label

    def read(self):
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"Worker exited {self.process.poll()}; see worker log")
        return json.loads(line)

    def measure(self, mode, repeat):
        self.process.stdin.write(json.dumps({"mode": mode, "repeat": repeat}) + "\n")
        self.process.stdin.flush()
        return dict(self.read(), label=self.label, compute_type=self.ready["compute_type"])

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.write('{"stop": true}\n')
            self.process.stdin.flush()
            self.process.wait(timeout=30)
        self.log.close()


def measure_configuration(args, backend, compute, records, controls):
    out = Path(args.out).resolve()
    workers = {}
    baseline_python = args.candidate_python if args.candidate_ablation else args.baseline_python
    runtimes = (("baseline", baseline_python), ("candidate", args.candidate_python))
    try:
        for label, python in runtimes:
            workers[label] = Worker(python, label, backend, args, compute=compute)
            for mode in args.modes:
                workers[label].measure(mode, -1)
        for repeat in range(args.repeats):
            for mode in args.modes:
                order = ("baseline", "candidate") if repeat % 2 == 0 else ("candidate", "baseline")
                pair = []
                for label in order:
                    record = dict(workers[label].measure(mode, repeat), backend=backend,
                                  comparison="candidate allocation on/off" if args.candidate_ablation else "4.9.1 vs reported PR")
                    records.append(record)
                    pair.append(record)
                    if args.operators and label == "candidate" and not record["operators"]["decoder_mask_calls"]:
                        raise RuntimeError("Candidate mask hook did not intercept the decoder")
                    print(json.dumps({k: v for k, v in record.items() if k not in ("hypotheses", "scores")}), flush=True)
                    (out / "measurements.json").write_text(json.dumps(records, indent=2) + "\n")
                if mode == "fixed":
                    for field in ("output_tokens", "encoder_calls", "decoder_calls", "decoder_row_steps", "max_step_plus_one"):
                        if pair[0][field] != pair[1][field]:
                            raise RuntimeError(f"Unequal fixed work for {field}: {pair}")
                if args.candidate_ablation and pair[0]["output_sha256"] != pair[1]["output_sha256"]:
                    raise RuntimeError("Allocation-only ablation changed token outputs")
    finally:
        for worker in workers.values():
            worker.close()
    # Separate uninstrumented controls check that the hook preserves tokens
    # and expose its timing overhead; do not mix these with phase timings.
    control_mode = "fixed" if "fixed" in args.modes else "natural"
    for label, python in runtimes:
        worker = Worker(python, label, backend, args, hooks=False, compute=compute)
        try:
            worker.measure(control_mode, -1)
            control = dict(worker.measure(control_mode, 0), backend=backend)
            reference = next(r for r in records if r["backend"] == backend and
                             r["compute_type"] == compute and r["label"] == label and r["mode"] == control_mode)
            if control["output_sha256"] != reference["output_sha256"]:
                raise RuntimeError("Phase hooks changed token output")
            controls.append(control)
            (out / "uninstrumented-controls.json").write_text(json.dumps(controls, indent=2) + "\n")
        finally:
            worker.close()


def run(args):
    out = Path(args.out).resolve()
    records, controls = [], []
    for compute in args.compute_types:
        for backend in args.backends:
            measure_configuration(args, backend, compute, records, controls)
    summary = []
    groups = sorted({(r["compute_type"], r["backend"], r["mode"], r["label"]) for r in records})
    for compute, backend, mode, label in groups:
        rows = [r for r in records if (r["compute_type"], r["backend"], r["mode"], r["label"]) == (compute, backend, mode, label)]
        summary.append(dict(compute_type=compute, backend=backend, mode=mode, label=label, **{
            key: statistics.median(row[key] for row in rows) for key in (
                "wall_ns", "output_tokens", "tokens_per_second", "encoder_ns", "decoder_ns",
                "other_ns", "decoder_calls", "decoder_row_steps")}))
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "build", "run", "inspect"])
    parser.add_argument("--out", required=True)
    parser.add_argument("--baseline-python")
    parser.add_argument("--candidate-python")
    parser.add_argument("--headers")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--length", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--operators", action="store_true")
    parser.add_argument("--heap-reuse", action="store_true",
                        help="Diagnostic-only fixed glibc thresholds for reusable heap buffers")
    parser.add_argument("--legacy-allocation", action="store_true",
                        help="Candidate-only ablation restoring the removed length-buffer lifetime")
    parser.add_argument("--candidate-ablation", action="store_true",
                        help="Use the candidate binary for both labels; toggle only the legacy allocation")
    parser.add_argument("--backends", nargs="+", choices=["mkl-packed", "mkl-unpacked", "dnnl"],
                        default=["mkl-packed", "mkl-unpacked", "dnnl"])
    parser.add_argument("--modes", nargs="+", choices=["natural", "fixed"],
                        default=["natural", "fixed"])
    parser.add_argument("--compute-types", nargs="+", choices=["int8_float32", "float32"],
                        default=["int8_float32", "float32"])
    args = parser.parse_args()
    {"prepare": prepare, "build": build, "run": run, "inspect": inspect}[args.action](args)


if __name__ == "__main__":
    main()
