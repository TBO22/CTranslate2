[![CI](https://github.com/OpenNMT/CTranslate2/workflows/CI/badge.svg)](https://github.com/OpenNMT/CTranslate2/actions?query=workflow%3ACI) [![PyPI version](https://badge.fury.io/py/ctranslate2.svg)](https://badge.fury.io/py/ctranslate2) [![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://opennmt.net/CTranslate2/) [![Gitter](https://badges.gitter.im/OpenNMT/CTranslate2.svg)](https://gitter.im/OpenNMT/CTranslate2?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge) [![Forum](https://img.shields.io/discourse/status?server=https%3A%2F%2Fforum.opennmt.net%2F)](https://forum.opennmt.net/)

# CTranslate2

CTranslate2 is a C++ and Python library for efficient inference with Transformer models.

The project implements a custom runtime that applies many performance optimization techniques such as weights quantization, layers fusion, batch reordering, etc., to [accelerate and reduce the memory usage](#benchmarks) of Transformer models on CPU and GPU.

> [!NOTE]
> This fork is the home of **MetalTranslate**, a native Apple Metal/MPS backend
> for CTranslate2 built for the Codex hackathon. It enables correct, private,
> on-device Transformer inference on Apple Silicon and reached **1.64x the
> performance of the optimized CPU backend** on the correctness-validated
> Marian translation workload described below. Jump to the
> [Apple Silicon setup guide](#apple-silicon-mps-backend-experimental) to try it.

The following model types are currently supported:

* Encoder-decoder models: Transformer base/big, M2M-100, NLLB, BART, mBART, Pegasus, T5, Whisper, T5Gemma, T5Gemma2, MADLAD-400
* Decoder-only models: GPT-2, GPT-J, GPT-NeoX, OPT, BLOOM, MPT, Llama, Mistral, Gemma, CodeGen, GPTBigCode, Falcon, Qwen2
* Encoder-only models: BERT, DistilBERT, XLM-RoBERTa

Compatible models should be first converted into an optimized model format. The library includes converters for multiple frameworks:

* [OpenNMT-py](https://opennmt.net/CTranslate2/guides/opennmt_py.html)
* [OpenNMT-tf](https://opennmt.net/CTranslate2/guides/opennmt_tf.html)
* [Fairseq](https://opennmt.net/CTranslate2/guides/fairseq.html)
* [Marian](https://opennmt.net/CTranslate2/guides/marian.html)
* [OPUS-MT](https://opennmt.net/CTranslate2/guides/opus_mt.html)
* [Transformers](https://opennmt.net/CTranslate2/guides/transformers.html)

The project is production-oriented and comes with [backward compatibility guarantees](https://opennmt.net/CTranslate2/versioning.html), but it also includes experimental features related to model compression and inference acceleration.

## Key features

* **Fast and efficient execution on CPU and GPU**<br/>The execution [is significantly faster and requires less resources](#benchmarks) than general-purpose deep learning frameworks on supported models and tasks thanks to many advanced optimizations: layer fusion, padding removal, batch reordering, in-place operations, caching mechanism, etc.
* **Quantization and reduced precision**<br/>The model serialization and computation support weights with [reduced precision](https://opennmt.net/CTranslate2/quantization.html): 16-bit floating points (FP16), 16-bit brain floating points (BF16), 16-bit integers (INT16), 8-bit integers (INT8) and AWQ quantization (INT4).
* **Multiple CPU architectures support**<br/>The project supports x86-64 and AArch64/ARM64 processors and integrates multiple backends that are optimized for these platforms: [Intel MKL](https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onemkl.html), [oneDNN](https://github.com/oneapi-src/oneDNN), [OpenBLAS](https://www.openblas.net/), [Ruy](https://github.com/google/ruy), and [Apple Accelerate](https://developer.apple.com/documentation/accelerate).
* **Automatic CPU detection and code dispatch**<br/>One binary can include multiple backends (e.g. Intel MKL and oneDNN) and instruction set architectures (e.g. AVX, AVX2) that are automatically selected at runtime based on the CPU information.
* **Parallel and asynchronous execution**<br/>Multiple batches can be processed in parallel and asynchronously using multiple GPUs or CPU cores.
* **Dynamic memory usage**<br/>The memory usage changes dynamically depending on the request size while still meeting performance requirements thanks to caching allocators on both CPU and GPU.
* **Lightweight on disk**<br/>Quantization can make the models 4 times smaller on disk with minimal accuracy loss.
* **Simple integration**<br/>The project has few dependencies and exposes simple APIs in [Python](https://opennmt.net/CTranslate2/python/overview.html) and C++ to cover most integration needs.
* **Configurable and interactive decoding**<br/>[Advanced decoding features](https://opennmt.net/CTranslate2/decoding.html) allow autocompleting a partial sequence and returning alternatives at a specific location in the sequence.
* **Support tensor parallelism for distributed inference**<br/>Very large model can be split into multiple GPUs. Following this [documentation](docs/parallel.md#model-and-tensor-parallelism) to set up the required environment.

Some of these features are difficult to achieve with standard deep learning frameworks and are the motivation for this project.

## Installation and usage

CTranslate2 can be installed with pip:

```bash
pip install ctranslate2
```

The Python module is used to convert models and can translate or generate text with few lines of code:

```python
translator = ctranslate2.Translator(translation_model_path)
translator.translate_batch(tokens)

generator = ctranslate2.Generator(generation_model_path)
generator.generate_batch(start_tokens)
```

See the [documentation](https://opennmt.net/CTranslate2) for more information and examples.

If you have an AMD ROCm GPU, we provide specific Python wheels on the [releases page](https://github.com/OpenNMT/CTranslate2/releases/).

### Apple Silicon MPS backend (experimental)

MetalTranslate is implemented directly inside the CTranslate2 runtime. It is
not a wrapper around PyTorch: model layers dispatch through CTranslate2's C++
operator system into Objective-C++ and Metal kernels. The backend includes a
persistent asynchronous command stream, decode-specific FP16 GEMV, tiled and
batched GEMM, GPU search and sampling, reductions, layout operations, and
quantized execution.

The MPS build is currently source-only and requires:

* An Apple Silicon Mac running macOS 11 or newer
* Xcode Command Line Tools (`xcode-select --install`)
* CMake and a C++17 compiler
* Python 3.9 or newer for the optional Python API

#### 1. Clone and create a Python environment

```bash
git clone --recursive https://github.com/TBO22/CTranslate2.git
cd CTranslate2

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

> [!IMPORTANT]
> CTranslate2 uses Git submodules for build dependencies such as `cxxopts`,
> `googletest`, and `spdlog`. GitHub's **Download ZIP** archive does not include
> their contents. Use the recursive clone command above for a source build.

If the repository was already cloned without `--recursive`, initialize the
missing dependencies before running CMake:

```bash
git submodule update --init --recursive
```

If CMake reports that `third_party/googletest` has no `CMakeLists.txt` or that
`cxxopts` is missing, the submodules were not initialized; the command above
fixes both errors.

#### 2. Build and install the C++ library

```bash
cmake -S . -B build-mps \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTS=ON \
  -DWITH_MPS=ON \
  -DWITH_ACCELERATE=ON \
  -DWITH_MKL=OFF \
  -DOPENMP_RUNTIME=NONE
cmake --build build-mps -j 4
cmake --install build-mps --prefix "$PWD/install-mps"
```

`WITH_MPS` cannot be combined with CUDA or HIP in the same build.

#### 3. Install the Python extension

The `CTRANSLATE2_ROOT` value makes the extension compile and link against the
MPS-enabled library that was just installed. `ARCHFLAGS` prevents an invalid
x86_64 slice from being added to the native Apple Silicon extension.

```bash
cd python
python -m pip install -r install_requirements.txt

export CTRANSLATE2_ROOT="$(cd ../install-mps && pwd)"
export ARCHFLAGS="-arch arm64"
python -m pip install -e .
cd ..
```

Verify the installation:

```bash
python -c 'import ctranslate2; print(ctranslate2.get_mps_device_count()); print(ctranslate2.get_supported_compute_types("mps"))'
```

On a supported Mac, the first value should be at least `1`. The compute-type
list should include `float32`, `float16`, `bfloat16`, and the INT8 hybrid modes.

#### 4. Run inference

```python
import ctranslate2

translator = ctranslate2.Translator(
    "path/to/converted/model",
    device="mps",
    compute_type="float16",
)

results = translator.translate_batch([["▁Hello", "▁world", "!"]])
print(results[0].hypotheses[0])
```

FP16 is the recommended compute type and is selected by `compute_type="auto"`
on MPS. INT8 reduces model weight size, but is not necessarily faster on Apple
GPUs, particularly during batch-size-1 decoding.

#### 5. Run the MPS correctness tests

Metal API validation is useful during development because it catches invalid
resource use and encoder mistakes:

```bash
MTL_DEBUG_LAYER=1 CT2_MPS_MAX_OPS=16 \
  ./build-mps/tests/ctranslate2_test tests/data \
  --gtest_filter='MPS/*:MPSBackendTest.*:TranslatorTest.MPS*'
```

The final validation run passed **175 of 175 MPS tests**, including
CPU-versus-MPS translation, FP32/FP16/BF16 operators, INT8 quantization,
batched GEMM, TopK, sampling, and quantized grouped Conv1D. A separate
CPU-only build passed 190 tests with 2 expected skips.

#### 6. Benchmark and profile

Build-time benchmarks cover decode GEMM, prefill GEMM, argmax, and copy-heavy
operations:

```bash
./build-mps/tests/benchmark_mps all
```

Set `CT2_MPS_PROFILE=1` to print command-buffer, synchronization, dispatch,
copy, GEMM-path, TopK, allocation, and buffer-lookup counters. See
[environment variables](docs/environment_variables.md) for all tuning and
diagnostic options.

#### Correctness-validated result

This representative measurement used a Release build, an Apple M1 MacBook Air
with a 7-core GPU, a real FP16 Marian Roman Pashto translation model, batch
size 1, and greedy decoding:

| Backend | Inference time | Throughput |
| --- | ---: | ---: |
| Optimized CTranslate2 CPU | 306.15 ms | 114.32 tokens/s |
| MetalTranslate FP16 | 186.96 ms | 187.21 tokens/s |

The result is a **1.64x speedup** and approximately **39% lower latency**.
Absolute performance varies by model, sequence length, search configuration,
and Apple chip. Earlier results produced a larger number while generating
incorrect tokens; those measurements were rejected rather than reported as a
speedup.

#### Supported precision and operations

The backend supports FP32, FP16, BF16, and the `int8_float32`,
`int8_float16`, and `int8_bfloat16` hybrid compute types. BF16 values are
stored in BF16 while GEMM and reduction accumulation use FP32. The INT8 path
uses signed INT8 matrices, INT32 accumulation, per-row activation scales, and
a fused dequantization/output kernel.

The backend keeps common decoding operations such as small TopK/argmax, TopP
masking, multinomial/Gumbel sampling, ALiBi, median filtering, and quantized or
dilated Conv1D on the GPU.

Current MPS limitations include FlashAttention, AWQ INT4, INT16 GEMM,
distributed collectives, and packed/shifted-u8 INT8 GEMM. GPU TopP currently
supports up to 1024 classes, and the optimized small TopK path supports
`k = 1, 2, 4, 8`. Metal kernels are compiled on first use, so performance
measurements should include warmup runs. See [hardware support](docs/hardware_support.md),
[quantization](docs/quantization.md), and [installation](docs/installation.md)
for additional details.

### My research, and where Codex and GPT-5.6 helped

I started this port about six months before the hackathon, and most of the
research behind it was work I had already done myself. I spent months reading
Apple's MLX Metal backend, especially its matmul, matvec, command submission,
tiling, and SIMD-group code. I compared that with the approaches used by
GGML/llama.cpp and PyTorch MPS, then compared all of them with CTranslate2's
CUDA backend and its own operator and tensor conventions.

Those implementations solve different problems. MLX was designed around
Apple hardware from the beginning. GGML is heavily shaped by quantized LLM
decoding and its own weight layouts. PyTorch MPS has to provide broad framework
coverage. CUDA has mature libraries, streams, and a more explicit device-memory
model. I could not just copy one of them into CTranslate2. I had to understand
which ideas made sense for CTranslate2's `StorageView`, primitives, model
loading, weight layouts, and autoregressive search.

I also spent a lot of time understanding unified memory. Apple Silicon lets
the CPU and GPU use the same physical memory, but that does not make execution
automatically synchronized. The CPU can still read data before the GPU has
finished writing it. That distinction shaped the allocator, buffer registry,
persistent command stream, and every place where the host genuinely needs a
result back from Metal.

By the time I brought Codex and GPT-5.6 into the project, I was not starting
from a blank prompt. I already had the backend direction, early kernels, and a
real Roman Pashto model exposing the problems. What I needed was help turning
that research into a complete integration without spending another six months
moving through a large C++ codebase one file at a time.

I used Codex directly inside the repository as a pair programmer. I would give
it a specific failure, a trace, or a benchmark result. It helped follow the
call path across C++, CUDA, Objective-C++, and Metal, implement the next piece,
compile it, add tests, and run the model again. This was especially useful for
finishing missing operations such as Gather, extending the test matrix to odd
shapes and batch strides, rebuilding the Python extension, resolving the
31-commit upstream gap, and cleaning up CI and documentation.

The optimization decisions still came from research and measurement on my
machine. I tested command-buffer limits of 16, 32, 64, and 128. I compared the
custom GEMV with the general matrix path. I found that INT8 worked but was
slower than FP16 on my M1, so I kept FP16 as the automatic default. When an
early run appeared more than twice as fast but produced repeated, corrupted
words, I rejected that number and kept debugging until the translation was
stable.

The fairest description is that I researched the architecture, chose the
direction, and validated the result on my own hardware and models. Codex and
GPT-5.6 helped me implement, debug, test, and finish that work much faster.
This project came from my Metal research; Codex helped me finally turn it into
a working CTranslate2 backend.

## Web Server

[ctranslate2-web-server](https://github.com/jordimas/ctranslate2-web-server) is a web server built on top of CTranslate2 that exposes an OpenAI-compatible REST API, making it easy to integrate CTranslate2 models into applications that already support the OpenAI API.

## Benchmarks

We translate the En->De test set *newstest2014* with multiple models:

* [OpenNMT-tf WMT14](https://opennmt.net/Models-tf/#translation): a base Transformer trained with OpenNMT-tf on the WMT14 dataset (4.5M lines)
* [OpenNMT-py WMT14](https://opennmt.net/Models-py/#translation): a base Transformer trained with OpenNMT-py on the WMT14 dataset (4.5M lines)
* [OPUS-MT](https://github.com/Helsinki-NLP/OPUS-MT-train/tree/master/models/en-de#opus-2020-02-26zip): a base Transformer trained with Marian on all OPUS data available on 2020-02-26 (81.9M lines)

The benchmark reports the number of target tokens generated per second (higher is better). The results are aggregated over multiple runs. See the [benchmark scripts](tools/benchmark) for more details and reproduce these numbers.

**Please note that the results presented below are only valid for the configuration used during this benchmark: absolute and relative performance may change with different settings.**

#### CPU

| | Tokens per second | Max. memory | BLEU |
| --- | --- | --- | --- |
| **OpenNMT-tf WMT14 model** | | | |
| OpenNMT-tf 2.31.0 (with TensorFlow 2.11.0) | 209.2 | 2653MB | 26.93 |
| **OpenNMT-py WMT14 model** | | | |
| OpenNMT-py 3.0.4 (with PyTorch 1.13.1) | 275.8 | 2012MB | 26.77 |
| - int8 | 323.3 | 1359MB | 26.72 |
| CTranslate2 3.6.0 | 658.8 | 849MB | 26.77 |
| - int16 | 733.0 | 672MB | 26.82 |
| - int8 | 860.2 | 529MB | 26.78 |
| - int8 + vmap | 1126.2 | 598MB | 26.64 |
| **OPUS-MT model** | | | |
| Transformers 4.26.1 (with PyTorch 1.13.1) | 147.3 | 2332MB | 27.90 |
| Marian 1.11.0 | 344.5 | 7605MB | 27.93 |
| - int16 | 330.2 | 5901MB | 27.65 |
| - int8 | 355.8 | 4763MB | 27.27 |
| CTranslate2 3.6.0 | 525.0 | 721MB | 27.92 |
| - int16 | 596.1 | 660MB | 27.53 |
| - int8 | 696.1 | 516MB | 27.65 |

Executed with 4 threads on a [*c5.2xlarge*](https://aws.amazon.com/ec2/instance-types/c5/) Amazon EC2 instance equipped with an Intel(R) Xeon(R) Platinum 8275CL CPU.

#### GPU

| | Tokens per second | Max. GPU memory | Max. CPU memory | BLEU |
| --- | --- | --- | --- | --- |
| **OpenNMT-tf WMT14 model** | | | | |
| OpenNMT-tf 2.31.0 (with TensorFlow 2.11.0) | 1483.5 | 3031MB | 3122MB | 26.94 |
| **OpenNMT-py WMT14 model** | | | | |
| OpenNMT-py 3.0.4 (with PyTorch 1.13.1) | 1795.2 | 2973MB | 3099MB | 26.77 |
| FasterTransformer 5.3 | 6979.0 | 2402MB | 1131MB | 26.77 |
| - float16 | 8592.5 | 1360MB | 1135MB | 26.80 |
| CTranslate2 3.6.0 | 6634.7 | 1261MB | 953MB | 26.77 |
| - int8 | 8567.2 | 1005MB | 807MB | 26.85 |
| - float16 | 10990.7 | 941MB | 807MB | 26.77 |
| - int8 + float16 | 8725.4 | 813MB | 800MB | 26.83 |
| **OPUS-MT model** | | | | |
| Transformers 4.26.1 (with PyTorch 1.13.1) | 1022.9 | 4097MB | 2109MB | 27.90 |
| Marian 1.11.0 | 3241.0 | 3381MB | 2156MB | 27.92 |
| - float16 | 3962.4 | 3239MB | 1976MB | 27.94 |
| CTranslate2 3.6.0 | 5876.4 | 1197MB | 754MB | 27.92 |
| - int8 | 7521.9 | 1005MB | 792MB | 27.79 |
| - float16 | 9296.7 | 909MB | 814MB | 27.90 |
| - int8 + float16 | 8362.7 | 813MB | 766MB | 27.90 |

Executed with CUDA 11 on a [*g5.xlarge*](https://aws.amazon.com/ec2/instance-types/g5/) Amazon EC2 instance equipped with a NVIDIA A10G GPU (driver version: 510.47.03).

## Contributing

CTranslate2 is a community-driven project. We welcome contributions of all kinds:
* **New Model Support:** Help us implement more Transformer architectures.
* **Performance:** Propose optimizations for CPU or GPU kernels.
* **Bug Reports:** Open an issue if you find something not working as expected.
* **Documentation:** Improve our guides or add new examples.

Check out our [Contributing Guide](CONTRIBUTING.md) to learn how to set up your development environment.

## Additional resources

* [Documentation](https://opennmt.net/CTranslate2)
* [Forum](https://forum.opennmt.net)
* [Gitter](https://gitter.im/OpenNMT/CTranslate2)
