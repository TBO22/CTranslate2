# Hardware support

```{attention}
The information below is only valid for the prebuilt binaries. If you compiled the project from the sources, the supported hardware will depend on the selected backend and compilation flags.
```

## CPU

* x86-64 processors supporting at least SSE 4.1
* AArch64/ARM64 processors

On x86-64, prebuilt binaries are configured to automatically select the best backend and instruction set architecture for the platform (AVX, AVX2, or AVX512). In particular, they are compiled with both [Intel MKL](https://software.intel.com/en-us/mkl) and [oneDNN](https://github.com/oneapi-src/oneDNN) so that Intel MKL is only used on Intel processors where it performs best, whereas oneDNN is used on other x86-64 processors such as AMD.

```{tip}
See the [environment variables](environment_variables.md) `CT2_USE_MKL` and `CT2_FORCE_CPU_ISA` to control this behavior.
```

## GPU

### NVIDIA (CUDA)

* NVIDIA GPUs with a Compute Capability greater or equal to 3.5

The driver requirement depends on the CUDA version. See the [CUDA Compatibility guide](https://docs.nvidia.com/deploy/cuda-compatibility/index.html) for more information.

### Apple Silicon (MPS)

* Apple M1/M2/M3 (and later) Macs with Metal Performance Shaders (MPS)

MPS support is **optional** and must be enabled at build time. Configure with
``-DWITH_MPS=ON`` when building on macOS (MPS is not available on Linux/Windows).
MPS and CUDA cannot be enabled together; use one GPU backend. Build the project
from source on an Apple Silicon Mac, then use ``device="mps"`` or ``device="auto"``
(in which case MPS is preferred over CPU when available) in the Python API.
