#ifdef __APPLE__

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <cstring>
#include <vector>

#include "ctranslate2/primitives.h"
#include "ctranslate2/types.h"
#include "mps/utils.h"
#include "type_dispatch.h"

namespace ctranslate2 {

  // MPS uses shared memory on Apple Silicon - we can run CPU primitives on MPS pointers.

  template<>
  template <typename T>
  T primitives<Device::MPS>::at(const T* x, dim_t index) {
    return x[index];
  }

  template<>
  template <typename T>
  void primitives<Device::MPS>::fill(T* x, T a, dim_t size) {
    primitives<Device::CPU>::fill(x, a, size);
  }

  template<>
  template <typename T>
  void primitives<Device::MPS>::strided_fill(T* x, T a, dim_t inc_x, dim_t size) {
    primitives<Device::CPU>::strided_fill(x, a, inc_x, size);
  }

  template<>
  template <typename T>
  void primitives<Device::MPS>::indexed_fill(T* x, T a, const int32_t* indices, dim_t num_indices) {
    primitives<Device::CPU>::indexed_fill(x, a, indices, num_indices);
  }

  template<>
  template <typename T>
  void primitives<Device::MPS>::copy(const T* x, T* y, dim_t size) {
    std::memcpy(y, x, size * sizeof(T));
  }

  template<>
  template <typename U, typename V>
  void primitives<Device::MPS>::convert(const U* x, V* y, dim_t size) {
    primitives<Device::CPU>::convert(x, y, size);
  }

  template void primitives<Device::MPS>::convert(const float*, float16_t*, dim_t);
  template void primitives<Device::MPS>::convert(const float16_t*, float*, dim_t);
  template void primitives<Device::MPS>::convert(const float*, bfloat16_t*, dim_t);
  template void primitives<Device::MPS>::convert(const bfloat16_t*, float*, dim_t);
  template void primitives<Device::MPS>::convert(const float16_t*, bfloat16_t*, dim_t);
  template void primitives<Device::MPS>::convert(const bfloat16_t*, float16_t*, dim_t);

  template<>
  template <typename T>
  T primitives<Device::MPS>::sum(const T* array, dim_t size) {
    return primitives<Device::CPU>::sum(array, size);
  }

  template<>
  template <typename T>
  dim_t primitives<Device::MPS>::max_element(const T* array, dim_t size) {
    return primitives<Device::CPU>::max_element(array, size);
  }

  template<>
  template <typename T>
  T primitives<Device::MPS>::max(const T* array, dim_t size) {
    return primitives<Device::CPU>::max(array, size);
  }

  template<>
  template <typename T>
  T primitives<Device::MPS>::amax(const T* array, dim_t size) {
    return primitives<Device::CPU>::amax(array, size);
  }

  template<>
  template <typename T>
  void primitives<Device::MPS>::add(T a, const T* x, T* y, dim_t size) {
    primitives<Device::CPU>::add(a, x, y, size);
  }

  template<>
  template <typename T>
  void primitives<Device::MPS>::add(const T* a, const T* b, T* c, dim_t size) {
    primitives<Device::CPU>::add(a, b, c, size);
  }

  template<>
  template <typename T>
  void primitives<Device::MPS>::add_batch_broadcast(const T* a, const T* b, T* c,
                                                     dim_t a_size, dim_t b_size) {
    primitives<Device::CPU>::add_batch_broadcast(a, b, c, a_size, b_size);
  }

  template<>
  template <typename T>
  void primitives<Device::MPS>::add_depth_broadcast(const T* a, const T* b, T* c,
                                                     dim_t a_size, dim_t b_size) {
    primitives<Device::CPU>::add_depth_broadcast(a, b, c, a_size, b_size);
  }

  template<>
  template <typename T>
  void primitives<Device::MPS>::add_block_broadcast(const T* a, const T* b, T* c,
                                                     dim_t block, dim_t a_size, dim_t b_size) {
    primitives<Device::CPU>::add_block_broadcast(a, b, c, block, a_size, b_size);
  }

  template<>
  template <typename T>
  void primitives<Device::MPS>::sub(const T* a, const T* b, T* c, dim_t size) {
    primitives<Device::CPU>::sub(a, b, c, size);
  }

  template<>
  template <typename T>
  void primitives<Device::MPS>::max(T a, const T* x, T* y, dim_t size) {
    primitives<Device::CPU>::max(a, x, y, size);
  }

  template<>
  template <typename T>
  void primitives<Device::MPS>::max(const T* a, const T* b, T* c, dim_t size) {
    primitives<Device::CPU>::max(a, b, c, size);
  }

  template<>
  template <typename T>
  void primitives<Device::MPS>::min(T a, const T* x, T* y, dim_t size) {
    primitives<Device::CPU>::min(a, x, y, size);
  }

  template<>
  template <typename T>
  void primitives<Device::MPS>::min(const T* a, const T* b, T* c, dim_t size) {
    primitives<Device::CPU>::min(a, b, c, size);
  }

  template<>
  template <typename T>
  void primitives<Device::MPS>::mul(T a, const T* x, T* y, dim_t size) {
    primitives<Device::CPU>::mul(a, x, y, size);
  }

  template<>
  template <typename T>
  void primitives<Device::MPS>::mul(const T* a, const T* b, T* c, dim_t size) {
    primitives<Device::CPU>::mul(a, b, c, size);
  }

  template<>
  template <typename T>
  void primitives<Device::MPS>::mul_batch_broadcast(const T* a, const T* b, T* c,
                                                     dim_t a_size, dim_t b_size) {
    primitives<Device::CPU>::mul_batch_broadcast(a, b, c, a_size, b_size);
  }

  template<>
  template <typename T>
  void primitives<Device::MPS>::penalize_previous_tokens(T* scores,
                                                          const T* previous_scores,
                                                          const int32_t* previous_ids,
                                                          T penalty,
                                                          dim_t batch_size,
                                                          dim_t length,
                                                          dim_t vocabulary_size) {
    primitives<Device::CPU>::penalize_previous_tokens(
        scores, previous_scores, previous_ids, penalty,
        batch_size, length, vocabulary_size);
  }

  template<>
  void primitives<Device::MPS>::prepare_length_mask(const int32_t* lengths,
                                                     dim_t batch_size,
                                                     dim_t num_heads,
                                                     dim_t num_queries,
                                                     bool mask_future,
                                                     bool multi_query,
                                                     int32_t* mask) {
    primitives<Device::CPU>::prepare_length_mask(
        lengths, batch_size, num_heads, num_queries, mask_future, multi_query, mask);
  }

  template<>
  template <typename T>
  void primitives<Device::MPS>::transpose_2d(const T* a, const dim_t* dims, T* b) {
    primitives<Device::CPU>::transpose_2d(a, dims, b);
  }

  template<>
  template <typename T>
  void primitives<Device::MPS>::transpose_3d(const T* a, const dim_t* dims,
                                              const dim_t* perm, T* b) {
    primitives<Device::CPU>::transpose_3d(a, dims, perm, b);
  }

  template<>
  template <typename T>
  void primitives<Device::MPS>::transpose_4d(const T* a, const dim_t* dims,
                                              const dim_t* perm, T* b) {
    primitives<Device::CPU>::transpose_4d(a, dims, perm, b);
  }

  template<>
  template <typename T>
  float primitives<Device::MPS>::logsumexp(const T* x, dim_t size) {
    return primitives<Device::CPU>::logsumexp(x, size);
  }

  template<>
  template <typename T>
  void primitives<Device::MPS>::exp(const T* x, T* y, dim_t size) {
    primitives<Device::CPU>::exp(x, y, size);
  }

  template<>
  template <typename T>
  void primitives<Device::MPS>::log(const T* x, T* y, dim_t size) {
    primitives<Device::CPU>::log(x, y, size);
  }

  template<>
  template <typename T>
  void primitives<Device::MPS>::cos(const T* x, T* y, dim_t size) {
    primitives<Device::CPU>::cos(x, y, size);
  }

  template<>
  template <typename T>
  void primitives<Device::MPS>::sin(const T* x, T* y, dim_t size) {
    primitives<Device::CPU>::sin(x, y, size);
  }

  template<>
  template <typename T>
  void primitives<Device::MPS>::tanh(const T* x, T* y, dim_t size) {
    primitives<Device::CPU>::tanh(x, y, size);
  }

  template<>
  template <typename T>
  void primitives<Device::MPS>::relu(const T* x, T* y, dim_t size) {
    primitives<Device::CPU>::relu(x, y, size);
  }

  template<>
  template <typename T>
  void primitives<Device::MPS>::gelu(const T* x, T* y, dim_t size) {
    primitives<Device::CPU>::gelu(x, y, size);
  }

  template<>
  template <typename T>
  void primitives<Device::MPS>::gelu_tanh(const T* x, T* y, dim_t size) {
    primitives<Device::CPU>::gelu_tanh(x, y, size);
  }

  template<>
  template <typename T>
  void primitives<Device::MPS>::gelu_sigmoid(const T* x, T* y, dim_t size) {
    primitives<Device::CPU>::gelu_sigmoid(x, y, size);
  }

  template<>
  template <typename T>
  void primitives<Device::MPS>::sigmoid(const T* x, T* y, dim_t size) {
    primitives<Device::CPU>::sigmoid(x, y, size);
  }

  template<>
  template <typename T>
  void primitives<Device::MPS>::swish(const T* x, T* y, dim_t size) {
    primitives<Device::CPU>::swish(x, y, size);
  }

  template<>
  void primitives<Device::MPS>::compute_u8_compensation(const int8_t* b,
                                                        bool transpose_b,
                                                        dim_t k, dim_t n,
                                                        float alpha,
                                                        int32_t* compensation) {
    primitives<Device::CPU>::compute_u8_compensation(b, transpose_b, k, n, alpha, compensation);
  }

  template<>
  template <typename T>
  dim_t primitives<Device::MPS>::gemm_pack_b(const T* b, const bool, const dim_t, const dim_t,
                                              const float, T*) {
    return 0;
  }

  // MPS GEMM: C = alpha * A * B + beta * C. Uses temp MTLBuffers (data pointers are buffer contents).
  // Fallback to CPU when transposition is required (MPS path supports transposes via copy layout).
  static void mps_gemm_float(bool transpose_a, bool transpose_b,
                             dim_t m, dim_t n, dim_t k,
                             float alpha,
                             const float* a, dim_t lda,
                             const float* b, dim_t ldb,
                             float beta,
                             float* c, dim_t ldc) {
    if (transpose_a || transpose_b) {
      primitives<Device::CPU>::gemm(false, false, transpose_a, transpose_b,
                                    m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, nullptr);
      return;
    }
    id<MTLDevice> device = (__bridge id<MTLDevice>)mps::get_device();
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)mps::get_command_queue();
    if (!device || !queue)
      throw std::runtime_error("MPS device not available");

    dim_t a_rows = transpose_a ? k : m;
    dim_t a_cols = transpose_a ? m : k;
    dim_t b_rows = transpose_b ? n : k;
    dim_t b_cols = transpose_b ? k : n;

    size_t rowBytesA = [MPSMatrixDescriptor rowBytesFromColumns:(NSUInteger)a_cols dataType:MPSDataTypeFloat32];
    size_t rowBytesB = [MPSMatrixDescriptor rowBytesFromColumns:(NSUInteger)b_cols dataType:MPSDataTypeFloat32];
    size_t rowBytesC = [MPSMatrixDescriptor rowBytesFromColumns:(NSUInteger)n dataType:MPSDataTypeFloat32];
    size_t a_bytes = rowBytesA * a_rows;
    size_t b_bytes = rowBytesB * b_rows;
    size_t c_bytes = rowBytesC * m;

    void* bufA = mps::allocate_buffer(a_bytes);
    void* bufB = mps::allocate_buffer(b_bytes);
    void* bufC = mps::allocate_buffer(c_bytes);
    id<MTLBuffer> mtlA = (__bridge id<MTLBuffer>)bufA;
    id<MTLBuffer> mtlB = (__bridge id<MTLBuffer>)bufB;
    id<MTLBuffer> mtlC = (__bridge id<MTLBuffer>)bufC;

    for (dim_t i = 0; i < a_rows; ++i)
      std::memcpy((char*)[mtlA contents] + i * rowBytesA, a + i * lda, a_cols * sizeof(float));
    for (dim_t i = 0; i < b_rows; ++i)
      std::memcpy((char*)[mtlB contents] + i * rowBytesB, b + i * ldb, b_cols * sizeof(float));
    std::memset([mtlC contents], 0, c_bytes);

    MPSMatrixDescriptor* descA = [MPSMatrixDescriptor matrixDescriptorWithDimensions:(NSUInteger)a_rows
                                                                           columns:(NSUInteger)a_cols
                                                                            rowBytes:rowBytesA
                                                                          dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor* descB = [MPSMatrixDescriptor matrixDescriptorWithDimensions:(NSUInteger)b_rows
                                                                           columns:(NSUInteger)b_cols
                                                                            rowBytes:rowBytesB
                                                                          dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor* descC = [MPSMatrixDescriptor matrixDescriptorWithDimensions:(NSUInteger)m
                                                                           columns:(NSUInteger)n
                                                                            rowBytes:rowBytesC
                                                                          dataType:MPSDataTypeFloat32];

    MPSMatrix* matA = [[MPSMatrix alloc] initWithBuffer:mtlA offset:0 descriptor:descA];
    MPSMatrix* matB = [[MPSMatrix alloc] initWithBuffer:mtlB offset:0 descriptor:descB];
    MPSMatrix* matC = [[MPSMatrix alloc] initWithBuffer:mtlC offset:0 descriptor:descC];

    MPSMatrixMultiplication* matmul = [[MPSMatrixMultiplication alloc] initWithDevice:device
                                                                        resultRows:(NSUInteger)m
                                                                     resultColumns:(NSUInteger)n
                                                                   interiorColumns:(NSUInteger)k];

    id<MTLCommandBuffer> cb = [queue commandBuffer];
    [matmul encodeToCommandBuffer:cb
                       leftMatrix:matA
                      rightMatrix:matB
                     resultMatrix:matC];

    [cb commit];
    [cb waitUntilCompleted];

    const char* r = (const char*)[mtlC contents];
    if (beta != 0) {
      std::vector<float> tmp(m * n);
      for (dim_t i = 0; i < m; ++i)
        for (dim_t j = 0; j < n; ++j)
          tmp[i * n + j] = beta * c[i * ldc + j] + alpha * ((const float*)(r + i * rowBytesC))[j];
      for (dim_t i = 0; i < m; ++i)
        std::memcpy(c + i * ldc, tmp.data() + i * n, n * sizeof(float));
    } else {
      for (dim_t i = 0; i < m; ++i)
        std::memcpy(c + i * ldc, r + i * rowBytesC, n * sizeof(float));
      if (alpha != 1.0f)
        primitives<Device::CPU>::mul(alpha, c, c, m * n);
    }

    mps::free_buffer(bufA);
    mps::free_buffer(bufB);
    mps::free_buffer(bufC);
  }

  template<>
  template<>
  void primitives<Device::MPS>::gemm(bool, bool,
                                      bool transpose_a, bool transpose_b,
                                      dim_t m, dim_t n, dim_t k,
                                      float alpha,
                                      const float* a, dim_t lda,
                                      const float* b, dim_t ldb,
                                      float beta,
                                      float* c, dim_t ldc,
                                      const float*) {
    mps_gemm_float(transpose_a, transpose_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  }

  template<typename T>
  static void gemm_fp16_via_float(bool transpose_a, bool transpose_b,
                                   dim_t m, dim_t n, dim_t k,
                                   float alpha,
                                   const T* a, dim_t lda,
                                   const T* b, dim_t ldb,
                                   float beta,
                                   T* c, dim_t ldc) {
    dim_t a_rows = transpose_a ? k : m;
    dim_t a_cols = transpose_a ? m : k;
    dim_t b_rows = transpose_b ? n : k;
    dim_t b_cols = transpose_b ? k : n;
    std::vector<float> af(a_rows * a_cols), bf(b_rows * b_cols), cf(m * n);
    for (dim_t i = 0; i < a_rows; ++i)
      for (dim_t j = 0; j < a_cols; ++j)
        af[i * a_cols + j] = float(a[i * lda + j]);
    for (dim_t i = 0; i < b_rows; ++i)
      for (dim_t j = 0; j < b_cols; ++j)
        bf[i * b_cols + j] = float(b[i * ldb + j]);
    for (dim_t i = 0; i < m; ++i)
      for (dim_t j = 0; j < n; ++j)
        cf[i * n + j] = float(c[i * ldc + j]);
    primitives<Device::MPS>::gemm(false, false, transpose_a, transpose_b,
                                  m, n, k, alpha,
                                  af.data(), a_cols, bf.data(), b_cols,
                                  beta, cf.data(), n, nullptr);
    for (dim_t i = 0; i < m; ++i)
      for (dim_t j = 0; j < n; ++j)
        c[i * ldc + j] = T(cf[i * n + j]);
  }

  template<>
  template<>
  void primitives<Device::MPS>::gemm(bool, bool,
                                      bool transpose_a, bool transpose_b,
                                      dim_t m, dim_t n, dim_t k,
                                      float alpha,
                                      const float16_t* a, dim_t lda,
                                      const float16_t* b, dim_t ldb,
                                      float beta,
                                      float16_t* c, dim_t ldc,
                                      const float16_t*) {
    gemm_fp16_via_float(transpose_a, transpose_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  }

  template<>
  template<>
  void primitives<Device::MPS>::gemm(bool, bool,
                                      bool transpose_a, bool transpose_b,
                                      dim_t m, dim_t n, dim_t k,
                                      float alpha,
                                      const bfloat16_t* a, dim_t lda,
                                      const bfloat16_t* b, dim_t ldb,
                                      float beta,
                                      bfloat16_t* c, dim_t ldc,
                                      const bfloat16_t*) {
    gemm_fp16_via_float(transpose_a, transpose_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  }

  template<>
  template <>
  void primitives<Device::MPS>::gemm_batch_strided(bool transpose_a, bool transpose_b,
                                                    dim_t m, dim_t n, dim_t k,
                                                    float alpha,
                                                    const float* a, dim_t lda, dim_t stridea,
                                                    const float* b, dim_t ldb, dim_t strideb,
                                                    float beta,
                                                    float* c, dim_t ldc, dim_t stridec,
                                                    dim_t batch_size) {
    for (dim_t i = 0; i < batch_size; ++i)
      primitives<Device::MPS>::gemm(false, false, transpose_a, transpose_b,
                                    m, n, k, alpha,
                                    a + i * stridea, lda,
                                    b + i * strideb, ldb,
                                    beta,
                                    c + i * stridec, ldc,
                                    nullptr);
  }

  template<>
  template <>
  void primitives<Device::MPS>::gemm_batch_strided(bool transpose_a, bool transpose_b,
                                                    dim_t m, dim_t n, dim_t k,
                                                    float alpha,
                                                    const float16_t* a, dim_t lda, dim_t stridea,
                                                    const float16_t* b, dim_t ldb, dim_t strideb,
                                                    float beta,
                                                    float16_t* c, dim_t ldc, dim_t stridec,
                                                    dim_t batch_size) {
    for (dim_t i = 0; i < batch_size; ++i)
      primitives<Device::MPS>::gemm(false, false, transpose_a, transpose_b,
                                    m, n, k, alpha,
                                    a + i * stridea, lda,
                                    b + i * strideb, ldb,
                                    beta,
                                    c + i * stridec, ldc,
                                    nullptr);
  }

  template<>
  template <>
  void primitives<Device::MPS>::gemm_batch_strided(bool transpose_a, bool transpose_b,
                                                    dim_t m, dim_t n, dim_t k,
                                                    float alpha,
                                                    const bfloat16_t* a, dim_t lda, dim_t stridea,
                                                    const bfloat16_t* b, dim_t ldb, dim_t strideb,
                                                    float beta,
                                                    bfloat16_t* c, dim_t ldc, dim_t stridec,
                                                    dim_t batch_size) {
    for (dim_t i = 0; i < batch_size; ++i)
      primitives<Device::MPS>::gemm(false, false, transpose_a, transpose_b,
                                    m, n, k, alpha,
                                    a + i * stridea, lda,
                                    b + i * strideb, ldb,
                                    beta,
                                    c + i * stridec, ldc,
                                    nullptr);
  }

  template<>
  template <typename T>
  void cross_device_primitives<Device::CPU, Device::MPS>::copy(const T* x, T* y, dim_t size) {
    std::memcpy(y, x, size * sizeof(T));
  }

  template<>
  template <typename T>
  void cross_device_primitives<Device::MPS, Device::CPU>::copy(const T* x, T* y, dim_t size) {
    std::memcpy(y, x, size * sizeof(T));
  }

#define MPS_DECLARE_IMPL(T)                                              \
  template T primitives<Device::MPS>::at(const T* x, dim_t index);       \
  template void primitives<Device::MPS>::fill(T* x, T a, dim_t size);    \
  template void primitives<Device::MPS>::copy<T>(const T* x, T* y, dim_t size); \
  template T primitives<Device::MPS>::sum(const T* array, dim_t size);   \
  template dim_t primitives<Device::MPS>::max_element(const T* array, dim_t size); \
  template T primitives<Device::MPS>::max(const T* array, dim_t size);   \
  template void primitives<Device::MPS>::add(T a, const T* x, T* y, dim_t size); \
  template void primitives<Device::MPS>::add(const T* a, const T* b, T* c, dim_t size); \
  template void primitives<Device::MPS>::sub(const T* a, const T* b, T* c, dim_t size); \
  template void primitives<Device::MPS>::mul(T a, const T* x, T* y, dim_t size); \
  template void primitives<Device::MPS>::mul(const T* a, const T* b, T* c, dim_t size); \
  template void cross_device_primitives<Device::CPU, Device::MPS>::copy<T>(const T*, T*, dim_t); \
  template void cross_device_primitives<Device::MPS, Device::CPU>::copy<T>(const T*, T*, dim_t);

  MPS_DECLARE_IMPL(float)
  MPS_DECLARE_IMPL(float16_t)
  MPS_DECLARE_IMPL(bfloat16_t)
  MPS_DECLARE_IMPL(int32_t)
  MPS_DECLARE_IMPL(int8_t)

}

#endif  // __APPLE__
