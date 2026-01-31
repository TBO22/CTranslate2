#include "ctranslate2/primitives.h"
#include "ctranslate2/types.h"
#include <stdexcept>
#include <vector>
#include <cmath>

namespace ctranslate2 {

  // Helper macro to throw errors for missing CPU FP16/BF16 support
  #define NO_CPU_IMPL(NAME) \
    throw std::runtime_error("CTranslate2 MPS Port: CPU fallback for " #NAME " with FP16/BF16 is not implemented. Ensure you are running on Device::MPS.");

  // ----------------------------------------------------------------------
  // Unary Math Operations
  // Your primitives.cc only has these for float. We need stubs for FP16/BF16.
  // ----------------------------------------------------------------------
  #define STUB_UNARY(NAME) \
    template<> template<> \
    void primitives<Device::CPU>::NAME<float16_t>(const float16_t*, float16_t*, dim_t) { \
      NO_CPU_IMPL(NAME) \
    } \
    template<> template<> \
    void primitives<Device::CPU>::NAME<bfloat16_t>(const bfloat16_t*, bfloat16_t*, dim_t) { \
      NO_CPU_IMPL(NAME) \
    }

  STUB_UNARY(exp)
  STUB_UNARY(log)
  STUB_UNARY(sin)
  STUB_UNARY(cos)
  STUB_UNARY(tanh)
  STUB_UNARY(relu)
  STUB_UNARY(gelu)
  STUB_UNARY(gelu_tanh)
  STUB_UNARY(gelu_sigmoid)
  STUB_UNARY(sigmoid)
  STUB_UNARY(swish)

  // ----------------------------------------------------------------------
  // Reductions
  // ----------------------------------------------------------------------
  
  // LogSumExp (returns float, implemented only for float in primitives.cc)
  template<> template<>
  float primitives<Device::CPU>::logsumexp<float16_t>(const float16_t*, dim_t) { NO_CPU_IMPL(logsumexp) return 0.f; }
  template<> template<>
  float primitives<Device::CPU>::logsumexp<bfloat16_t>(const bfloat16_t*, dim_t) { NO_CPU_IMPL(logsumexp) return 0.f; }

  // ----------------------------------------------------------------------
  // GEMM & Matrix Operations
  // Implemented only for float/int8/int16 in primitives.cc
  // ----------------------------------------------------------------------

  // Standard GEMM
  template<> template<>
  void primitives<Device::CPU>::gemm<float16_t, float16_t>(
      bool, bool, bool, bool, dim_t, dim_t, dim_t, float,
      const float16_t*, dim_t, const float16_t*, dim_t, float, float16_t*, dim_t, const float16_t*) {
      NO_CPU_IMPL(gemm)
  }
  template<> template<>
  void primitives<Device::CPU>::gemm<bfloat16_t, bfloat16_t>(
      bool, bool, bool, bool, dim_t, dim_t, dim_t, float,
      const bfloat16_t*, dim_t, const bfloat16_t*, dim_t, float, bfloat16_t*, dim_t, const bfloat16_t*) {
      NO_CPU_IMPL(gemm)
  }

  // GEMM Batch Strided
  template<> template<>
  void primitives<Device::CPU>::gemm_batch_strided<float16_t, float16_t>(
      bool, bool, dim_t, dim_t, dim_t, float,
      const float16_t*, dim_t, dim_t, const float16_t*, dim_t, dim_t, float, float16_t*, dim_t, dim_t, dim_t) {
      NO_CPU_IMPL(gemm_batch_strided)
  }
  template<> template<>
  void primitives<Device::CPU>::gemm_batch_strided<bfloat16_t, bfloat16_t>(
      bool, bool, dim_t, dim_t, dim_t, float,
      const bfloat16_t*, dim_t, dim_t, const bfloat16_t*, dim_t, dim_t, float, bfloat16_t*, dim_t, dim_t, dim_t) {
      NO_CPU_IMPL(gemm_batch_strided)
  }
  
  // GEMM Pack B
  template<> template<>
  dim_t primitives<Device::CPU>::gemm_pack_b<float16_t>(const float16_t*, bool, dim_t, dim_t, float, float16_t*) {
    NO_CPU_IMPL(gemm_pack_b) return 0;
  }
  template<> template<>
  dim_t primitives<Device::CPU>::gemm_pack_b<bfloat16_t>(const bfloat16_t*, bool, dim_t, dim_t, float, bfloat16_t*) {
    NO_CPU_IMPL(gemm_pack_b) return 0;
  }

} // namespace ctranslate2