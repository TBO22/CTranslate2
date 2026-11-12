#include "test_utils.h"
#include "mps/kernels.h"
#include "ctranslate2/ops/gemm.h"
#include "ctranslate2/ops/ops.h"
#include "ctranslate2/layers/attention_layer.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace {

  void check_streaming_attention(DataType dtype, dim_t batch, dim_t heads, dim_t kv_heads,
                                 dim_t qlen, dim_t klen, dim_t depth, bool causal,
                                 dim_t offset = 0, dim_t window = 0,
                                 bool masked = false, bool interleaved = false,
                                 bool large_scores = false) {
    SCOPED_TRACE(::testing::Message() << int(dtype) << " B=" << batch << " H=" << heads
                 << "/" << kv_heads << " Q=" << qlen << " K=" << klen << " D=" << depth);
    std::vector<float> qv(batch * heads * qlen * depth);
    std::vector<float> kv(batch * kv_heads * klen * depth), vv(kv.size());
    for (size_t i = 0; i < qv.size(); ++i)
      qv[i] = large_scores ? 80.f : std::sin(float(i) * .13f);
    for (size_t i = 0; i < kv.size(); ++i) {
      kv[i] = large_scores ? ((i / depth) % 2 ? 80.f : -80.f) : std::cos(float(i) * .07f);
      vv[i] = std::sin(float(i) * .11f);
    }
    StorageView q = StorageView({batch, heads, qlen, depth}, qv).to(dtype);
    StorageView k = StorageView({batch, kv_heads, klen, depth}, kv).to(dtype);
    StorageView v = StorageView({batch, kv_heads, klen, depth}, vv).to(dtype);
    // Compare the exact representable inputs, including BF16 rounding.
    qv = q.to_float32().to_vector<float>();
    kv = k.to_float32().to_vector<float>();
    vv = v.to_float32().to_vector<float>();
    std::vector<int32_t> lens(batch * heads * qlen, klen);
    if (masked) {
      for (size_t i = 0; i < lens.size(); ++i)
        lens[i] = i % 7 == 0 ? -1 : int32_t(i % (klen + 3));
    }
    StorageView lengths({batch * heads * qlen}, lens, Device::MPS);
    auto qindex = [&](dim_t b, dim_t h, dim_t t, dim_t d) {
      return interleaved ? ((b * qlen + t) * heads + h) * depth + d
                         : ((b * heads + h) * qlen + t) * depth + d;
    };
    const float scale = 1.f / std::sqrt(float(depth));
    std::vector<float> reference(qv.size(), 0);
    for (dim_t b = 0; b < batch; ++b) {
      for (dim_t h = 0; h < heads; ++h) {
        const dim_t kh = h / (heads / kv_heads);
        for (dim_t t = 0; t < qlen; ++t) {
          dim_t end = masked ? std::max<dim_t>(0, std::min<dim_t>(lens[(b * heads + h) * qlen + t], klen)) : klen;
          if (causal)
            end = std::min(end, offset + t + 1);
          const dim_t first = window ? std::max<dim_t>(0, offset + t + 1 - window) : 0;
          if (end <= first)
            continue;
          std::vector<double> scores(end - first);
          double maximum = -std::numeric_limits<double>::infinity();
          for (dim_t j = first; j < end; ++j) {
            double score = 0;
            for (dim_t d = 0; d < depth; ++d)
              score += double(qv[qindex(b, h, t, d)]) * kv[((b * kv_heads + kh) * klen + j) * depth + d];
            scores[j - first] = score * scale;
            maximum = std::max(maximum, scores[j - first]);
          }
          double denominator = 0;
          for (double& score : scores) {
            score = std::exp(score - maximum);
            denominator += score;
          }
          for (dim_t d = 0; d < depth; ++d) {
            double sum = 0;
            for (dim_t j = first; j < end; ++j)
              sum += scores[j - first] * vv[((b * kv_heads + kh) * klen + j) * depth + d];
            reference[qindex(b, h, t, d)] = sum / denominator;
          }
        }
      }
    }
    q = q.to(Device::MPS);
    k = k.to(Device::MPS);
    v = v.to(Device::MPS);
    StorageView out(q.shape(), dtype, Device::MPS);
    ASSERT_TRUE(mps::streaming_attention(dtype, q.buffer(), k.buffer(), v.buffer(),
                                         masked ? lengths.data<int32_t>() : nullptr, out.buffer(),
                                         batch, heads, kv_heads, qlen, klen, depth, scale,
                                         causal, offset, window, interleaved));
    const auto actual = out.to_float32().to_vector<float>();
    float maximum_error = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
      ASSERT_TRUE(std::isfinite(actual[i])) << "index=" << i;
      maximum_error = std::max(maximum_error, std::abs(actual[i] - reference[i]));
    }
    EXPECT_LT(maximum_error, dtype == DataType::FLOAT16 ? 0.002f : 0.016f);
  }
}

TEST(MPSBackendTest, StreamingAttentionMasksGqaMqaAndDepthTails) {
  for (auto dtype : {DataType::FLOAT16, DataType::BFLOAT16}) {
    check_streaming_attention(dtype, 1, 4, 4, 7, 65, 13, true, 58);
    check_streaming_attention(dtype, 2, 6, 2, 33, 65, 64, true, 32, 17, true);
    check_streaming_attention(dtype, 1, 4, 1, 1, 2049, 128, true, 2048);
    check_streaming_attention(dtype, 1, 2, 1, 33, 67, 256, false);
    check_streaming_attention(dtype, 1, 2, 2, 5, 73, 7, false, 0, 0, true, true);
  }
}

TEST(MPSBackendTest, StreamingAttentionLongCausalPrefill) {
  for (auto dtype : {DataType::FLOAT16, DataType::BFLOAT16})
    check_streaming_attention(dtype, 1, 2, 1, 513, 513, 80, true);
}

TEST(MPSBackendTest, StreamingAttentionSoftmaxDoesNotOverflowHalfScores) {
  for (auto dtype : {DataType::FLOAT16, DataType::BFLOAT16})
    check_streaming_attention(dtype, 1, 2, 1, 5, 65, 128, true, 60, 0, false, false, true);
}

TEST(MPSBackendTest, StreamingAttentionDeclinesUnsupportedShapes) {
  StorageView q({514}, float16_t(0), Device::MPS), k(q), v(q), out(q);
  auto supported = [&](DataType dtype, dim_t h, dim_t kh, dim_t d, bool causal, dim_t window) {
    return mps::streaming_attention(dtype, q.buffer(), k.buffer(), v.buffer(), nullptr, out.buffer(),
                                    1, h, kh, 1, 1, d, 1, causal, 0, window);
  };
  EXPECT_FALSE(supported(DataType::FLOAT32, 2, 2, 64, false, 0));
  EXPECT_FALSE(supported(DataType::FLOAT16, 3, 2, 64, false, 0));
  EXPECT_FALSE(supported(DataType::FLOAT16, 2, 2, 257, false, 0));
  EXPECT_FALSE(supported(DataType::FLOAT16, 2, 2, 64, false, 4));
  EXPECT_FALSE(mps::streaming_attention(DataType::FLOAT16, q.buffer(), k.buffer(), v.buffer(), nullptr, out.buffer(),
                                        std::numeric_limits<dim_t>::max(), 2, 2, 1, 1, 64, 1));
  EXPECT_FALSE(mps::streaming_attention(DataType::FLOAT16, q.buffer(), k.buffer(), v.buffer(), nullptr, k.buffer(),
                                        1, 2, 2, 1, 1, 64, 1));
  EXPECT_FALSE(mps::streaming_attention(DataType::FLOAT16, q.buffer(), k.buffer(), v.buffer(), nullptr, out.buffer(),
                                        1, 2, 2, 1, 1, 64, std::numeric_limits<float>::infinity()));
}

TEST(MPSBackendTest, StreamingAttentionIgnoresMaskedNonfiniteValuesAndAllowsInplaceQueries) {
  for (auto dtype : {DataType::FLOAT16, DataType::BFLOAT16}) {
    constexpr dim_t queries = 4, keys = 9, depth = 7;
    auto q = StorageView({1, 1, queries, depth}, .25f).to(dtype).to(Device::MPS);
    auto k = StorageView({1, 1, keys, depth}, .125f).to(dtype).to(Device::MPS);
    std::vector<float> values(keys * depth, std::numeric_limits<float>::quiet_NaN());
    std::fill(values.begin(), values.begin() + 3 * depth, 2.f);
    auto v = StorageView({1, 1, keys, depth}, values).to(dtype).to(Device::MPS);
    StorageView lengths({queries}, std::vector<int32_t>{3, 0, -1, 3}, Device::MPS);
    ASSERT_TRUE(mps::streaming_attention(dtype, q.buffer(), k.buffer(), v.buffer(), lengths.data<int32_t>(),
                                          q.buffer(), 1, 1, 1, queries, keys, depth, 1.f));
    const auto actual = q.to_float32().to_vector<float>();
    for (dim_t row = 0; row < queries; ++row)
      for (dim_t d = 0; d < depth; ++d)
        ASSERT_EQ(actual[row * depth + d], row == 0 || row == 3 ? 2.f : 0.f);
  }
}

TEST(MPSBackendTest, DecoderPrefillGemmBetaZeroIgnoresNonfiniteOutput) {
  {
    StorageView a = StorageView({2047, 1024}, .125f).to_float16().to(Device::MPS);
    StorageView b = StorageView({4096, 1024}, .0625f).to_float16().to(Device::MPS);
    StorageView c = StorageView({2047, 4096}, std::numeric_limits<float>::quiet_NaN())
      .to_float16().to(Device::MPS);
    ops::Gemm(1, 0, false, true)(a, b, c);
    const auto actual = c.to_float32().to_vector<float>();
    for (float value : actual)
      ASSERT_EQ(value, 8.f) << "exact Qwen prefill shape";
  }
  for (dim_t rows : {dim_t(1), dim_t(33), dim_t(2047)}) {
    StorageView a = StorageView({rows, 128}, .125f).to_float16().to(Device::MPS);
    StorageView b = StorageView({1024, 128}, .0625f).to_float16().to(Device::MPS);
    StorageView c = StorageView({rows, 1024}, std::numeric_limits<float>::quiet_NaN())
      .to_float16().to(Device::MPS);
    ops::Gemm(1, 0, false, true)(a, b, c);
    const auto actual = c.to_float32().to_vector<float>();
    for (float value : actual)
      ASSERT_EQ(value, 1.f) << "rows=" << rows;
  }
  for (auto dtype : {DataType::FLOAT32, DataType::FLOAT16}) {
    StorageView a = StorageView({4096, 1}, .125f).to(dtype).to(Device::MPS);
    StorageView b = StorageView({1, 64}, .0625f).to(dtype).to(Device::MPS);
    StorageView c = StorageView({4096, 64}, std::numeric_limits<float>::quiet_NaN())
      .to(dtype).to(Device::MPS);
    ops::Gemm(1, 0, false, false)(a, b, c);
    const auto actual = c.to_float32().to_vector<float>();
    for (float value : actual)
      ASSERT_EQ(value, .0078125f) << "dtype=" << int(dtype);
  }
}

TEST(MPSBackendTest, DecoderPrefillGemmReusesGpuPoisonedStorage) {
  StorageView a = StorageView({2047, 1024}, .125f).to_float16().to(Device::MPS);
  StorageView nan_a = StorageView({2047, 1024}, std::numeric_limits<float>::quiet_NaN())
    .to_float16().to(Device::MPS);
  StorageView b = StorageView({4096, 1024}, .0625f).to_float16().to(Device::MPS);
  StorageView c(DataType::FLOAT16, Device::MPS);
  for (int i = 0; i < 4; ++i) {
    ops::Gemm(1, 0, false, true)(nan_a, b, c);
    ops::Gemm(1, 0, false, true)(a, b, c);
    const auto actual = c.to_float32().to_vector<float>();
    for (float value : actual)
      ASSERT_EQ(value, 8.f) << "iteration=" << i;
  }
}

TEST(MPSBackendTest, DecoderPrefillGemmDoesNotReadNanRowsBeyondMatrix) {
  constexpr dim_t rows = 2047, k = 1024, n = 4096;
  std::vector<float> av((rows + 1) * k, .125f);
  std::fill(av.begin() + rows * k, av.end(), std::numeric_limits<float>::quiet_NaN());
  StorageView backing = StorageView({rows + 1, k}, av).to_float16().to(Device::MPS);
  StorageView a(DataType::FLOAT16, Device::MPS);
  a.view(backing.data<float16_t>(), {rows, k});
  StorageView b = StorageView({n, k}, .0625f).to_float16().to(Device::MPS);
  StorageView c(DataType::FLOAT16, Device::MPS);
  ops::Gemm(1, 0, false, true)(a, b, c);
  const auto actual = c.to_float32().to_vector<float>();
  for (size_t i = 0; i < actual.size(); ++i)
    ASSERT_EQ(actual[i], 8.f) << "row=" << i / n << " col=" << i % n;
}

TEST(MPSBackendTest, NativePrefillGemmPreservesStridePaddingAndOddTails) {
  constexpr dim_t m = 1025, n = 1031, k = 1033;
  constexpr dim_t lda = 1040, ldb = 1040, ldc = 1040;
  const float poison = std::numeric_limits<float>::quiet_NaN();
  std::vector<float> av(m * k), bv(n * k);
  for (size_t i = 0; i < av.size(); ++i)
    av[i] = .05f * std::sin(float(i) * .13f);
  for (size_t i = 0; i < bv.size(); ++i)
    bv[i] = .05f * std::cos(float(i) * .07f);
  const auto a_cpu = StorageView({m, k}, av).to_float16();
  const auto b_cpu = StorageView({n, k}, bv).to_float16();
  StorageView reference;
  ops::Gemm(1, 0, false, true)(a_cpu.to_float32(), b_cpu.to_float32(), reference);
  const auto expected = reference.to_float16().to_float32().to_vector<float>();
  // Offset zero exercises native aligned loads. Offset one exercises the
  // generic fallback: it cannot use the native kernel's 16-byte vector loads.
  for (dim_t offset : {dim_t(0), dim_t(1)}) {
    SCOPED_TRACE(offset);
    std::vector<float16_t> a(m * lda + offset, float16_t(poison));
    std::vector<float16_t> b(n * ldb + offset, float16_t(poison));
    for (dim_t row = 0; row < m; ++row)
      std::copy_n(a_cpu.data<float16_t>() + row * k, k, a.data() + offset + row * lda);
    for (dim_t row = 0; row < n; ++row)
      std::copy_n(b_cpu.data<float16_t>() + row * k, k, b.data() + offset + row * ldb);
    StorageView agpu({dim_t(a.size())}, a, Device::MPS);
    StorageView bgpu({dim_t(b.size())}, b, Device::MPS);
    StorageView cgpu({m * ldc + 2}, float16_t(poison), Device::MPS);
    mps::gemm(DataType::FLOAT16, false, true, m, n, k, 1,
              agpu.data<float16_t>() + offset, lda,
              bgpu.data<float16_t>() + offset, ldb,
              0, cgpu.data<float16_t>() + 1, ldc);
    const auto actual = cgpu.to_float32().to_vector<float>();
    ASSERT_TRUE(std::isnan(actual.front()));
    ASSERT_TRUE(std::isnan(actual.back()));
    for (dim_t row = 0; row < m; ++row) {
      for (dim_t col = 0; col < n; ++col)
        ASSERT_NEAR(actual[1 + row * ldc + col], expected[row * n + col], .002f)
          << "row=" << row << " col=" << col;
      for (dim_t col = n; col < ldc; ++col)
        ASSERT_TRUE(std::isnan(actual[1 + row * ldc + col]));
    }
  }
}

TEST(MPSBackendTest, DecoderRotaryCacheGrowthPreservesEarlierPositions) {
  layers::RotaryEmbeddings rotary(128, false);
  StorageView initial = StorageView({1, 16, 2047, 128}, 1.f).to_float16().to(Device::MPS);
  rotary.apply(initial, 0);
  const auto reference = initial.to_float32().to_vector<float>();
  StorageView grow = StorageView({1, 16, 1, 128}, 1.f).to_float16().to(Device::MPS);
  rotary.apply(grow, 2048);
  StorageView repeated = StorageView({1, 16, 2047, 128}, 1.f).to_float16().to(Device::MPS);
  rotary.apply(repeated, 0);
  const auto actual = repeated.to_float32().to_vector<float>();
  for (size_t i = 0; i < actual.size(); ++i)
    ASSERT_FLOAT_EQ(actual[i], reference[i]) << "index=" << i;
}

TEST(MPSBackendTest, LongCausalAttentionIgnoresPoisonedScoreStorage) {
  constexpr dim_t time = 2047, heads = 2, depth = 128;
  StorageView q = StorageView({1, heads, time, depth}, .125f).to_float16().to(Device::MPS);
  StorageView k = StorageView({1, heads, time, depth}, .0625f).to_float16().to(Device::MPS);
  StorageView v = StorageView({1, heads, time, depth}, 1.f).to_float16().to(Device::MPS);
  std::vector<int32_t> lens(heads * time);
  for (dim_t h = 0; h < heads; ++h)
    for (dim_t t = 0; t < time; ++t)
      lens[h * time + t] = t + 1;
  StorageView lengths({heads * time}, lens, Device::MPS);
  for (int iteration = 0; iteration < 3; ++iteration) {
    StorageView scores = StorageView({1, heads, time, time}, std::numeric_limits<float>::quiet_NaN())
      .to_float16().to(Device::MPS);
    ops::MatMul(false, true, 1.f / std::sqrt(float(depth)))(q, k, scores);
    const auto raw = scores.to_float32().to_vector<float>();
    for (float value : raw)
      ASSERT_TRUE(std::isfinite(value)) << "QK iteration=" << iteration;
    ops::SoftMax()(scores, lengths, scores);
    const auto probabilities = scores.to_float32().to_vector<float>();
    for (float value : probabilities)
      ASSERT_TRUE(std::isfinite(value)) << "softmax iteration=" << iteration;
    StorageView out(DataType::FLOAT16, Device::MPS);
    ops::MatMul()(scores, v, out);
    const auto actual = out.to_float32().to_vector<float>();
    for (float value : actual)
      ASSERT_NEAR(value, 1.f, 0.002f) << "AV iteration=" << iteration;
  }
}
