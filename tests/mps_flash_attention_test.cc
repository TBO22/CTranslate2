#include "ctranslate2/ops/fused_attention.h"
#include "ctranslate2/ops/ops.h"
#include "mps/flash_attention.h"
#include "mps/kernels.h"
#include "test_utils.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace {
  StorageView reference_attention(const StorageView& q, const StorageView& k,
                                  const StorageView& v, dim_t valid,
                                  dim_t offset, dim_t window,
                                  const std::vector<int32_t>* lengths = nullptr,
                                  bool causal = true) {
    const dim_t b = q.dim(0), t = q.dim(1), h = q.dim(2), d = q.dim(3),
                kh = k.dim(2), cap = k.dim(1);
    const auto qv = q.to_float32().to_vector<float>();
    const auto kv = k.to_float32().to_vector<float>(),
               vv = v.to_float32().to_vector<float>();
    std::vector<float> result(q.size(), 0);
    for (dim_t batch = 0; batch < b; ++batch)
      for (dim_t head = 0; head < h; ++head)
        for (dim_t row = 0; row < t; ++row) {
          const dim_t first =
              window ? std::max<dim_t>(0, offset + row + 1 - window) : 0;
          dim_t end = causal ? std::min(valid, offset + row + 1) : valid;
          if (lengths)
            end = std::min(
                end,
                std::max<dim_t>(0, (*lengths)[(batch * h + head) * t + row]));
          if (end <= first)
            continue;
          std::vector<double> scores(end - first);
          double maximum = -std::numeric_limits<double>::infinity();
          for (dim_t key = first; key < end; ++key) {
            double score = 0;
            for (dim_t col = 0; col < d; ++col)
              score +=
                  double(qv[((batch * t + row) * h + head) * d + col]) *
                  kv[((batch * cap + key) * kh + head / (h / kh)) * d + col];
            scores[key - first] = score / std::sqrt(float(d));
            maximum = std::max(maximum, scores[key - first]);
          }
          double denominator = 0;
          for (auto& score : scores) {
            score = std::exp(score - maximum);
            denominator += score;
          }
          for (dim_t col = 0; col < d; ++col) {
            double value = 0;
            for (dim_t key = first; key < end; ++key)
              value +=
                  scores[key - first] *
                  vv[((batch * cap + key) * kh + head / (h / kh)) * d + col];
            result[((batch * t + row) * h + head) * d + col] =
                value / denominator;
          }
        }
    return StorageView(q.shape(), result);
  }

  StorageView signal(Shape shape, DataType dtype, float frequency) {
    dim_t size = 1;
    for (auto x : shape)
      size *= x;
    std::vector<float> values(size);
    for (size_t i = 0; i < values.size(); ++i)
      values[i] = std::sin(float(i) * frequency);
    return StorageView(shape, values).to(dtype);
  }
} // namespace

TEST(MPSBackendTest, FlashPrefillMatchesDoubleAcrossMasksWindowsAndTails) {
  for (auto dtype : {DataType::FLOAT16, DataType::BFLOAT16})
    for (dim_t depth : {13, 64, 128, 256})
      for (dim_t qlen : {7, 33, 129}) {
        SCOPED_TRACE(::testing::Message()
                     << int(dtype) << ' ' << depth << ' ' << qlen);
        constexpr dim_t batch = 2, heads = 4, kvheads = 2, offset = 17;
        const dim_t valid = offset + qlen, cap = 512;
        auto q = signal({batch, qlen, heads, depth}, dtype, .13f);
        std::vector<float> kv(batch * cap * kvheads * depth), vv(kv.size());
        for (dim_t b = 0; b < batch; ++b)
          for (dim_t t = 0; t < cap; ++t)
            for (dim_t j = 0; j < kvheads * depth; ++j) {
              const dim_t i = (b * cap + t) * kvheads * depth + j;
              kv[i] = t < valid - b * 5
                          ? std::sin(float(i) * .07f)
                          : std::numeric_limits<float>::quiet_NaN();
              vv[i] = t < valid - b * 5
                          ? std::cos(float(i) * .11f)
                          : std::numeric_limits<float>::quiet_NaN();
            }
        auto k = StorageView({batch, cap, kvheads, depth}, kv).to(dtype);
        auto v = StorageView(k.shape(), vv).to(dtype);
        std::vector<int32_t> lengths(batch * heads * qlen);
        for (dim_t b = 0; b < batch; ++b)
          for (dim_t h = 0; h < heads; ++h)
            for (dim_t r = 0; r < qlen; ++r)
              lengths[(b * heads + h) * qlen + r] = h == 0 ? 0 : valid - b * 5;
        auto qgpu = q.to(Device::MPS), kgpu = k.to(Device::MPS),
             vgpu = v.to(Device::MPS);
        StorageView mask({batch, heads, qlen}, lengths, Device::MPS),
            out(q.shape(), dtype, Device::MPS);
        for (dim_t window : {0, 19}) {
          ASSERT_TRUE(mps::flash_prefill(
              dtype, qgpu.buffer(), kgpu.buffer(), vgpu.buffer(),
              mask.data<int32_t>(), out.buffer(), batch, heads, kvheads, qlen,
              valid, cap, depth, 1.f / std::sqrt(float(depth)), offset, window,
              true, false));
          expect_storage_eq(
              out.to_float32(),
              reference_attention(q, k, v, valid, offset, window, &lengths),
              dtype == DataType::FLOAT16 ? .002f : .016f);
        }
        if (depth == 13 && qlen == 33) {
          std::vector<int32_t> interleaved(lengths.size());
          for (dim_t b = 0; b < batch; ++b)
            for (dim_t h = 0; h < heads; ++h)
              for (dim_t r = 0; r < qlen; ++r)
                interleaved[(b * qlen + r) * heads + h] =
                    lengths[(b * heads + h) * qlen + r];
          auto imask =
              StorageView({batch, qlen, heads}, interleaved, Device::MPS);
          ASSERT_TRUE(mps::flash_prefill(
              dtype, qgpu.buffer(), kgpu.buffer(), vgpu.buffer(),
              imask.data<int32_t>(), out.buffer(), batch, heads, kvheads, qlen,
              valid, cap, depth, 1.f / std::sqrt(float(depth)), offset, 0,
              false, true));
          expect_storage_eq(
              out.to_float32(),
              reference_attention(q, k, v, valid, offset, 0, &lengths, false),
              dtype == DataType::FLOAT16 ? .002f : .016f);
        }
      }
}

TEST(MPSBackendTest, FlashPrefillLongMqaAndLargeScoresAreStable) {
  for (auto dtype : {DataType::FLOAT16, DataType::BFLOAT16}) {
    constexpr dim_t n = 513, depth = 128;
    auto q = StorageView({1, n, 2, depth}, 80.f).to(dtype);
    auto k = StorageView({1, n, 1, depth}, 80.f).to(dtype);
    auto v = signal(k.shape(), dtype, .11f);
    auto qgpu = q.to(Device::MPS), kgpu = k.to(Device::MPS),
         vgpu = v.to(Device::MPS);
    StorageView out(q.shape(), dtype, Device::MPS);
    ASSERT_TRUE(
        mps::flash_prefill(dtype, qgpu.buffer(), kgpu.buffer(), vgpu.buffer(),
                           nullptr, out.buffer(), 1, 2, 1, n, n, n, depth,
                           1.f / std::sqrt(float(depth)), 0, 0, true, false));
    expect_storage_eq(out.to_float32(), reference_attention(q, k, v, n, 0, 0),
                      dtype == DataType::FLOAT16 ? .002f : .016f);
  }
}

TEST(MPSBackendTest, FlashDecodeMatchesDoubleReferenceAcrossSplitsGqaAndMasks) {
  for (auto dtype : {DataType::FLOAT16, DataType::BFLOAT16})
    for (dim_t depth : {dim_t(13), dim_t(64), dim_t(65), dim_t(128), dim_t(129),
                        dim_t(255), dim_t(256)})
      for (dim_t valid : {dim_t(65), dim_t(257), dim_t(2049)}) {
        SCOPED_TRACE(::testing::Message()
                     << int(dtype) << ' ' << depth << ' ' << valid);
        constexpr dim_t batch = 2, heads = 6, kvheads = 2;
        const dim_t cap = ((valid + 511) / 512) * 512;
        const auto q = signal({batch, 1, heads, depth}, dtype, .13f);
        const auto k = signal({batch, cap, kvheads, depth}, dtype, .07f);
        const auto v = signal(k.shape(), dtype, .11f);
        auto qgpu = q.to(Device::MPS), kgpu = k.to(Device::MPS),
             vgpu = v.to(Device::MPS);
        for (dim_t window : {dim_t(0), dim_t(17)}) {
          std::vector<int32_t> lens(batch * heads, valid);
          lens[0] = -1;
          lens[1] = 0;
          lens[2] = int32_t(valid + 3);
          lens[3] = int32_t(valid / 2);
          const auto expected =
              reference_attention(q, k, v, valid, valid - 1, window, &lens);
          StorageView bounds({batch * heads}, lens, Device::MPS),
              out(q.shape(), dtype, Device::MPS);
          const dim_t splits = (valid + 255) / 256;
          StorageView workspace({batch, heads, splits, depth + 2},
                                DataType::FLOAT32, Device::MPS);
          ASSERT_TRUE(mps::flash_decode(
              dtype, qgpu.buffer(), kgpu.buffer(), vgpu.buffer(),
              bounds.data<int32_t>(), out.buffer(), workspace.data<float>(),
              workspace.size(), batch, heads, kvheads, valid, cap, depth,
              1.f / std::sqrt(float(depth)), valid - 1, window));
          expect_storage_eq(out.to_float32(), expected,
                            dtype == DataType::FLOAT16 ? .002f : .016f);
        }
      }
}

TEST(MPSBackendTest, FlashDecodeStableLargeScoresMqaAndPoisonedUnusedCache) {
  for (auto dtype : {DataType::FLOAT16, DataType::BFLOAT16}) {
    constexpr dim_t valid = 513, cap = 1024, depth = 128, heads = 4;
    const auto q = StorageView({1, 1, heads, depth}, 80.f).to(dtype);
    std::vector<float> kv(cap * depth, std::numeric_limits<float>::quiet_NaN()),
        vv(kv);
    for (dim_t key = 0; key < valid; ++key)
      for (dim_t d = 0; d < depth; ++d) {
        kv[key * depth + d] = key % 2 ? 80.f : -80.f;
        vv[key * depth + d] = std::sin(float(key + d) * .1f);
      }
    const auto k = StorageView({1, cap, 1, depth}, kv).to(dtype),
               v = StorageView(k.shape(), vv).to(dtype);
    auto qgpu = q.to(Device::MPS), kgpu = k.to(Device::MPS),
         vgpu = v.to(Device::MPS);
    StorageView out(q.shape(), dtype, Device::MPS),
        work({heads, 3, depth + 2}, DataType::FLOAT32, Device::MPS);
    ASSERT_TRUE(mps::flash_decode(
        dtype, qgpu.buffer(), kgpu.buffer(), vgpu.buffer(), nullptr,
        out.buffer(), work.data<float>(), work.size(), 1, heads, 1, valid, cap,
        depth, 1.f / std::sqrt(float(depth)), valid - 1));
    expect_storage_eq(out.to_float32(),
                      reference_attention(q, k, v, valid, valid - 1, 0), .016f);
  }
}

TEST(MPSBackendTest, FlashDecodeCooperativeChunksRespectMasksAndTokenOrder) {
  constexpr dim_t batch = 2, queries = 3, heads = 6, kvheads = 2;
  constexpr dim_t valid = 257, capacity = 512, offset = valid - queries;
  for (auto dtype : {DataType::FLOAT16, DataType::BFLOAT16})
    for (dim_t depth : {65, 128}) {
      const auto q = signal({batch, queries, heads, depth}, dtype, .13f);
      const auto k = signal({batch, capacity, kvheads, depth}, dtype, .07f);
      const auto v = signal(k.shape(), dtype, .11f);
      auto qgpu = q.to(Device::MPS), kgpu = k.to(Device::MPS),
           vgpu = v.to(Device::MPS);
      std::vector<int32_t> lengths(batch * heads * queries, valid);
      lengths[0] = 0;
      lengths[3] = valid / 2;
      StorageView bounds({batch, heads, queries}, lengths, Device::MPS),
          out(q.shape(), dtype, Device::MPS),
          workspace({batch, heads, queries, 2, depth + 2}, DataType::FLOAT32,
                    Device::MPS);
      for (bool causal : {false, true})
        for (dim_t window : {0, 17}) {
          if (!causal && window)
            continue;
          ASSERT_TRUE(mps::flash_decode(
              dtype, qgpu.buffer(), kgpu.buffer(), vgpu.buffer(),
              bounds.data<int32_t>(), out.buffer(), workspace.data<float>(),
              workspace.size(), batch, heads, kvheads, valid, capacity, depth,
              1.f / std::sqrt(float(depth)), offset, window, causal, queries,
              true, true));
          expect_storage_eq(out.to_float32(),
                            reference_attention(q, k, v, valid, offset, window,
                                                &lengths, causal),
                            dtype == DataType::FLOAT16 ? .002f : .016f);
        }
    }
}

TEST(MPSBackendTest,
     SharedFlashDecodeBhtdAndInterleavedBeamMatchDoubleReference) {
  for (auto dtype : {DataType::FLOAT16, DataType::BFLOAT16})
    for (dim_t queries : {dim_t(1), dim_t(5)})
      for (dim_t keys : {dim_t(33), dim_t(1501)}) {
        constexpr dim_t batch = 2, heads = 3, depth = 13;
        auto q = signal({batch, queries, heads, depth}, dtype, .13f);
        auto k = signal({batch, keys, heads, depth}, dtype, .07f),
             v = signal(k.shape(), dtype, .11f);
        StorageView kt(dtype), vt(dtype), qt(dtype);
        ops::Transpose({0, 2, 1, 3})(k, kt);
        ops::Transpose({0, 2, 1, 3})(v, vt);
        const bool interleaved = queries > 1;
        if (!interleaved)
          ops::Transpose({0, 2, 1, 3})(q, qt);
        std::vector<int32_t> mask(batch * heads * queries, keys);
        mask[0] = 0;
        mask[1] = -1;
        mask[2] = keys + 9;
        mask.back() = keys / 2;
        const auto expected =
            reference_attention(q, k, v, keys, keys, 0, &mask);
        auto qgpu = (interleaved ? q : qt).to(Device::MPS);
        auto kgpu = kt.to(Device::MPS), vgpu = vt.to(Device::MPS);
        StorageView lengths({batch, heads, queries}, mask, Device::MPS),
            out(dtype, Device::MPS), converted(dtype, Device::MPS);
        ASSERT_TRUE(ops::FusedAttention()(qgpu, kgpu, vgpu, &lengths,
                                          1.f / std::sqrt(float(depth)),
                                          interleaved, out));
        if (interleaved)
          converted = std::move(out);
        else
          ops::Transpose({0, 2, 1, 3})(out, converted);
        expect_storage_eq(converted.to_float32(), expected,
                          dtype == DataType::FLOAT16 ? .002f : .016f);
      }
}

TEST(MPSBackendTest,
     SharedFlashKeepsGenericFallbackForPrefillAndLargeHeadDimensions) {
  for (auto shape : {Shape{1, 2, 17, 64}, Shape{1, 2, 1, 257}}) {
    StorageView q(shape, float16_t(.1f), Device::MPS),
        k({1, 2, 33, shape.back()}, float16_t(.2f), Device::MPS),
        v(k.shape(), float16_t(.3f), Device::MPS),
        out(DataType::FLOAT16, Device::MPS);
    EXPECT_FALSE(ops::FusedAttention()(q, k, v, nullptr, .125f, false, out));
  }
}

TEST(MPSBackendTest,
     CompactFlashCacheAppendGrowthAndMultiTokenChunksMatchReference) {
  for (auto dtype : {DataType::FLOAT16, DataType::BFLOAT16}) {
    constexpr dim_t batch = 2, heads = 4, kvheads = 2, depth = 13;
    StorageView ck(dtype, Device::MPS), cv(dtype, Device::MPS);
    StorageView allk(dtype), allv(dtype);
    dim_t offset = 0;
    const void* initial = nullptr;
    for (dim_t count : {dim_t(511), dim_t(1), dim_t(3), dim_t(1)}) {
      const auto q = signal({batch, count, heads, depth}, dtype, .13f);
      const auto k = signal({batch, count, kvheads, depth}, dtype, .07f),
                 v = signal(k.shape(), dtype, .11f);
      if (!offset) {
        allk = k;
        allv = v;
      } else {
        StorageView nk(dtype), nv(dtype);
        ops::Concat(1)({&allk, &k}, nk);
        ops::Concat(1)({&allv, &v}, nv);
        allk = std::move(nk);
        allv = std::move(nv);
      }
      auto qgpu = q.to(Device::MPS), kgpu = k.to(Device::MPS),
           vgpu = v.to(Device::MPS);
      StorageView out(dtype, Device::MPS);
      mps::flash_attention(qgpu, kgpu, vgpu, nullptr, out, &ck, &cv, nullptr,
                           true, 1.f / std::sqrt(float(depth)), offset, 17);
      expect_storage_eq(
          out.to_float32(),
          reference_attention(q, allk, allv, offset + count, offset, 17),
          dtype == DataType::FLOAT16 ? .003f : .025f);
      if (!offset) {
        initial = ck.buffer();
        EXPECT_EQ(ck.dim(1), 512);
      }
      if (offset == 511)
        EXPECT_EQ(ck.buffer(), initial);
      if (offset == 512) {
        EXPECT_NE(ck.buffer(), initial);
        EXPECT_EQ(ck.dim(1), 1024);
      }
      StorageView prefix(dtype, Device::MPS);
      ops::Slide(1, 0, offset + count)(ck, prefix);
      expect_storage_eq(prefix.to(Device::CPU), allk, 0);
      ops::Slide(1, 0, offset + count)(cv, prefix);
      expect_storage_eq(prefix.to(Device::CPU), allv, 0);
      offset += count;
    }
  }
}

TEST(MPSBackendTest,
     InterleavedRotaryMatchesCpuAcrossHeadsBatchesAndPartialDimensions) {
  for (auto dtype : {DataType::FLOAT32, DataType::FLOAT16, DataType::BFLOAT16})
    for (bool interleave : {false, true}) {
      const auto input = signal({2, 5, 3, 10}, dtype, .13f);
      const auto sine = signal({5, 6}, dtype, .07f),
                 cosine = signal({5, 6}, dtype, .11f);
      StorageView reference, actual(dtype, Device::MPS), transposed, rotated;
      ops::Rotary rotary(6, interleave);
      // The CPU primitive's established layout is BHTD. Transpose explicitly
      // so this reference does not inherit its interleaved-layout limitation.
      ops::Transpose({0, 2, 1, 3})(input.to_float32(), transposed);
      rotary(transposed, sine.to_float32(), cosine.to_float32(), rotated, true);
      ops::Transpose({0, 2, 1, 3})(rotated, reference);
      rotary(input.to(Device::MPS), sine.to(Device::MPS),
             cosine.to(Device::MPS), actual, false);
      expect_storage_eq(actual.to_float32(), reference.to_float32(),
                        dtype == DataType::BFLOAT16 ? .016f : .002f);
    }
}

TEST(MPSBackendTest,
     FlashMqaInterleavedMasksAndCacheBeamReorderingPreserveTokenOrder) {
  constexpr dim_t batch = 2, heads = 4, depth = 13, time = 7;
  const auto dtype = DataType::FLOAT16;
  auto q = signal({batch, time, heads, depth}, dtype, .13f);
  auto k = signal({batch, time, 1, depth}, dtype, .07f),
       v = signal(k.shape(), dtype, .11f);
  std::vector<int32_t> mask(batch * heads * time), interleaved(mask.size());
  for (dim_t b = 0; b < batch; ++b)
    for (dim_t h = 0; h < heads; ++h)
      for (dim_t t = 0; t < time; ++t) {
        const int32_t value = std::min<dim_t>(t + 1, b == 0 ? 3 : time);
        mask[(b * heads + h) * time + t] = value;
        interleaved[(b * time + t) * heads + h] = value;
      }
  StorageView lengths({batch, time, heads}, interleaved, Device::MPS);
  auto qgpu = q.to(Device::MPS), kgpu = k.to(Device::MPS),
       vgpu = v.to(Device::MPS);
  StorageView ck(dtype, Device::MPS), cv(dtype, Device::MPS),
      out(dtype, Device::MPS);
  mps::flash_attention(qgpu, kgpu, vgpu, &lengths, out, &ck, &cv, nullptr, true,
                       1.f / std::sqrt(float(depth)), 0, 0, true, true);
  expect_storage_eq(out.to_float32(),
                    reference_attention(q, k, v, time, 0, 0, &mask), .003f);
  StorageView indices({3}, std::vector<int32_t>{1, 1, 0});
  auto gpuindices = indices.to(Device::MPS);
  StorageView reordered_k(dtype, Device::MPS), reordered_v(dtype, Device::MPS),
      refk(dtype), refv(dtype);
  ops::Gather()(ck, gpuindices, reordered_k);
  ops::Gather()(cv, gpuindices, reordered_v);
  ops::Gather()(k, indices, refk);
  ops::Gather()(v, indices, refv);
  ck = std::move(reordered_k);
  cv = std::move(reordered_v);
  q = signal({3, 1, heads, depth}, dtype, .13f);
  k = signal({3, 1, 1, depth}, dtype, .17f);
  v = signal(k.shape(), dtype, .19f);
  StorageView allk(dtype), allv(dtype);
  ops::Concat(1)({&refk, &k}, allk);
  ops::Concat(1)({&refv, &v}, allv);
  qgpu = q.to(Device::MPS);
  kgpu = k.to(Device::MPS);
  vgpu = v.to(Device::MPS);
  mps::flash_attention(qgpu, kgpu, vgpu, nullptr, out, &ck, &cv, nullptr, true,
                       1.f / std::sqrt(float(depth)), time, 0);
  expect_storage_eq(out.to_float32(),
                    reference_attention(q, allk, allv, time + 1, time, 0),
                    .003f);
}
