#ifdef __APPLE__
#include <stdexcept>
#include <vector>
#include "ctranslate2/types.h"
#include "ctranslate2/storage_view.h"

// --- Standard Ops ---
#include "ctranslate2/ops/dequantize.h"
#include "ctranslate2/ops/quantize.h"
#include "ctranslate2/ops/softmax.h"
#include "ctranslate2/ops/rms_norm.h"
#include "ctranslate2/ops/layer_norm.h"
#include "ctranslate2/ops/bias_add.h"
#include "ctranslate2/ops/tile.h"
#include "ctranslate2/ops/split.h"
#include "ctranslate2/ops/concat.h"
#include "ctranslate2/ops/gather.h"
#include "ctranslate2/ops/slide.h"
#include "ctranslate2/ops/multinomial.h"
#include "ctranslate2/ops/topk.h"
#include "ctranslate2/ops/rotary.h"
#include "ctranslate2/ops/conv1d.h"
#include "ctranslate2/ops/alibi_add.h"
#include "ctranslate2/ops/median_filter.h"
#include "ctranslate2/ops/gumbel_max.h"
#include "ctranslate2/ops/mean.h"
#include "ctranslate2/ops/flash_attention.h"
#include "ctranslate2/ops/gemm.h"
#include "ctranslate2/ops/nccl_ops.h"
#include "ctranslate2/ops/topp_mask.h"

// --- AWQ Ops (Corrected Paths) ---
#include "ctranslate2/ops/awq/gemm.h"
#include "ctranslate2/ops/awq/gemv.h"
#include "ctranslate2/ops/awq/dequantize_awq.h"

namespace ctranslate2 {
namespace ops {

#define THROW_MPS(NAME) throw std::runtime_error("MPS " #NAME " not implemented");

// --- Quantization ---
template <Device D, typename T, typename Q>
void Dequantize::dequantize(const StorageView&, const StorageView&, StorageView&) const { THROW_MPS(Dequantize) }
#define INST_DEQUANTIZE(T) \
template void Dequantize::dequantize<Device::MPS, int8_t, T>(const StorageView&, const StorageView&, StorageView&) const;
INST_DEQUANTIZE(float)
INST_DEQUANTIZE(float16_t)
INST_DEQUANTIZE(bfloat16_t)

template <Device D, typename T>
void Dequantize::dequantize_gemm_output(const StorageView&, const StorageView&, const StorageView&, bool, bool, const StorageView*, StorageView&) const { THROW_MPS(DequantizeGemm) }
#define INST_DEQ_GEMM(T) \
template void Dequantize::dequantize_gemm_output<Device::MPS, T>(const StorageView&, const StorageView&, const StorageView&, bool, bool, const StorageView*, StorageView&) const;
INST_DEQ_GEMM(float)
INST_DEQ_GEMM(float16_t)
INST_DEQ_GEMM(bfloat16_t)

template <Device D, typename T, typename Q>
void Quantize::quantize(const StorageView&, StorageView&, StorageView&) const { THROW_MPS(Quantize) }
#define INST_QUANTIZE(T) \
template void Quantize::quantize<Device::MPS, T, int8_t>(const StorageView&, StorageView&, StorageView&) const;
INST_QUANTIZE(float)
INST_QUANTIZE(float16_t)
INST_QUANTIZE(bfloat16_t)

// --- Math & Activation ---
template <Device D, typename T>
void SoftMax::compute(const StorageView&, const StorageView*, StorageView&) const { THROW_MPS(SoftMax) }
#define INST_SOFTMAX(T) template void SoftMax::compute<Device::MPS, T>(const StorageView&, const StorageView*, StorageView&) const;
INST_SOFTMAX(float)
INST_SOFTMAX(float16_t)
INST_SOFTMAX(bfloat16_t)

template <Device D, typename T>
void RMSNorm::compute(const StorageView&, const StorageView&, StorageView&) const { THROW_MPS(RMSNorm) }
#define INST_RMS(T) template void RMSNorm::compute<Device::MPS, T>(const StorageView&, const StorageView&, StorageView&) const;
INST_RMS(float)
INST_RMS(float16_t)
INST_RMS(bfloat16_t)

template <Device D, typename T>
void LayerNorm::compute(const StorageView*, const StorageView*, const StorageView&, dim_t, dim_t, dim_t, dim_t, StorageView&) const { THROW_MPS(LayerNorm) }
#define INST_LN(T) template void LayerNorm::compute<Device::MPS, T>(const StorageView*, const StorageView*, const StorageView&, dim_t, dim_t, dim_t, dim_t, StorageView&) const;
INST_LN(float)
INST_LN(float16_t)
INST_LN(bfloat16_t)

template <Device D, typename T>
void BiasAdd::compute(const StorageView&, const StorageView&, StorageView&, const StorageView*) const { THROW_MPS(BiasAdd) }
#define INST_BIAS(T) template void BiasAdd::compute<Device::MPS, T>(const StorageView&, const StorageView&, StorageView&, const StorageView*) const;
INST_BIAS(float)
INST_BIAS(float16_t)
INST_BIAS(bfloat16_t)

// --- Manipulation ---
template <Device D, typename T>
void Tile::compute(const StorageView&, dim_t, dim_t, StorageView&) const { THROW_MPS(Tile) }
#define INST_TILE(T) template void Tile::compute<Device::MPS, T>(const StorageView&, dim_t, dim_t, StorageView&) const;
INST_TILE(float)
INST_TILE(float16_t)
INST_TILE(bfloat16_t)
INST_TILE(int8_t)
INST_TILE(int32_t)
INST_TILE(int16_t)

template <Device D, typename T>
void Split::compute(const StorageView&, std::vector<StorageView*>&) const { THROW_MPS(Split) }
#define INST_SPLIT(T) template void Split::compute<Device::MPS, T>(const StorageView&, std::vector<StorageView*>&) const;
INST_SPLIT(float)
INST_SPLIT(float16_t)
INST_SPLIT(bfloat16_t)
INST_SPLIT(int8_t)
INST_SPLIT(int32_t)
INST_SPLIT(int16_t)

template <Device D, typename T>
void Concat::compute(const std::vector<const StorageView*>&, StorageView&) const { THROW_MPS(Concat) }
#define INST_CONCAT(T) template void Concat::compute<Device::MPS, T>(const std::vector<const StorageView*>&, StorageView&) const;
INST_CONCAT(float)
INST_CONCAT(float16_t)
INST_CONCAT(bfloat16_t)
INST_CONCAT(int8_t)
INST_CONCAT(int32_t)
INST_CONCAT(int16_t)

template <Device D, typename T>
void Gather::compute(const StorageView&, const StorageView&, dim_t, dim_t, StorageView&) const { THROW_MPS(Gather) }
#define INST_GATHER(T) template void Gather::compute<Device::MPS, T>(const StorageView&, const StorageView&, dim_t, dim_t, StorageView&) const;
INST_GATHER(float)
INST_GATHER(float16_t)
INST_GATHER(bfloat16_t)
INST_GATHER(int8_t)
INST_GATHER(int32_t)
INST_GATHER(int16_t)

template <Device D, typename T>
void Slide::compute(const StorageView&, StorageView&, const dim_t&) const { THROW_MPS(Slide) }
#define INST_SLIDE(T) template void Slide::compute<Device::MPS, T>(const StorageView&, StorageView&, const dim_t&) const;
INST_SLIDE(float)
INST_SLIDE(float16_t)
INST_SLIDE(bfloat16_t)
INST_SLIDE(int8_t)
INST_SLIDE(int32_t)
INST_SLIDE(int16_t)

// --- Advanced ---
template <Device D, typename T>
void Multinomial::compute(const StorageView&, StorageView&) const { THROW_MPS(Multinomial) }
#define INST_MULTINOMIAL(T) template void Multinomial::compute<Device::MPS, T>(const StorageView&, StorageView&) const;
INST_MULTINOMIAL(float)
INST_MULTINOMIAL(float16_t)
INST_MULTINOMIAL(bfloat16_t)

template <Device D, typename T, typename IndexT>
void TopK::compute(const StorageView&, StorageView&, StorageView&) const { THROW_MPS(TopK) }
#define INST_TOPK(T) template void TopK::compute<Device::MPS, T, int32_t>(const StorageView&, StorageView&, StorageView&) const;
INST_TOPK(float)
INST_TOPK(float16_t)
INST_TOPK(bfloat16_t)

template <Device D, typename T>
void Rotary::compute(const StorageView&, const StorageView&, const StorageView&, StorageView&, bool) const { THROW_MPS(Rotary) }
#define INST_ROTARY(T) template void Rotary::compute<Device::MPS, T>(const StorageView&, const StorageView&, const StorageView&, StorageView&, bool) const;
INST_ROTARY(float)
INST_ROTARY(float16_t)
INST_ROTARY(bfloat16_t)

template <Device D, typename T>
void Conv1D::compute(const StorageView&, const StorageView&, const StorageView*, StorageView&, const StorageView*) const { THROW_MPS(Conv1D) }
#define INST_CONV1D(T) template void Conv1D::compute<Device::MPS, T>(const StorageView&, const StorageView&, const StorageView*, StorageView&, const StorageView*) const;
INST_CONV1D(float)
INST_CONV1D(float16_t)
INST_CONV1D(bfloat16_t)

template <Device D, typename T>
void AlibiAdd::compute(const StorageView&, const StorageView&, dim_t, StorageView&) const { THROW_MPS(AlibiAdd) }
#define INST_ALIBI(T) template void AlibiAdd::compute<Device::MPS, T>(const StorageView&, const StorageView&, dim_t, StorageView&) const;
INST_ALIBI(float)
INST_ALIBI(float16_t)
INST_ALIBI(bfloat16_t)

template <Device D, typename T>
void MedianFilter::compute(const StorageView&, dim_t, StorageView&) const { THROW_MPS(MedianFilter) }
#define INST_MEDIAN(T) template void MedianFilter::compute<Device::MPS, T>(const StorageView&, dim_t, StorageView&) const;
INST_MEDIAN(float)
INST_MEDIAN(float16_t)
INST_MEDIAN(bfloat16_t)

template <Device D, typename T>
void GumbelMax::add_gumbel_noise(const StorageView&, StorageView&) const { THROW_MPS(GumbelMax) }
#define INST_GUMBEL(T) template void GumbelMax::add_gumbel_noise<Device::MPS, T>(const StorageView&, StorageView&) const;
INST_GUMBEL(float)
INST_GUMBEL(float16_t)
INST_GUMBEL(bfloat16_t)

// --- Missing Stats ---
template <Device D, typename T>
void Mean::compute(const StorageView&, dim_t, dim_t, dim_t, bool, StorageView&) const { THROW_MPS(Mean) }
#define INST_MEAN(T) template void Mean::compute<Device::MPS, T>(const StorageView&, dim_t, dim_t, dim_t, bool, StorageView&) const;
INST_MEAN(float)
INST_MEAN(float16_t)
INST_MEAN(bfloat16_t)

// --- Flash Attention & AWQ ---
template<> void FlashAttention::compute<Device::MPS>(StorageView&, StorageView&, StorageView&, StorageView&, StorageView*, StorageView*, StorageView*, bool, StorageView*, StorageView*, bool, StorageView*, dim_t) const { THROW_MPS(FlashAttention) }

template<> void DequantizeAwq::dequantize<Device::MPS, int32_t, float16_t>(const StorageView&, const StorageView&, const StorageView&, StorageView&) const { THROW_MPS(DequantizeAwq) }

template<> void GemmAwq::compute<Device::MPS, float16_t, int32_t>(const StorageView&, const StorageView&, const StorageView&, const StorageView&, StorageView&) const { THROW_MPS(GemmAwq) }
template<> void GemvAwq::compute_gemv<Device::MPS, float16_t, int32_t>(const StorageView&, const StorageView&, const StorageView&, const StorageView&, StorageView&) const { THROW_MPS(GemvAwq) }
template<> void GemvAwq::compute_gemv2<Device::MPS, float16_t, int32_t>(const StorageView&, const StorageView&, const StorageView&, const StorageView&, StorageView&) const { THROW_MPS(GemvAwq2) }

// --- NCCL / Distributed ---
template <Device D, typename T>
void GatherAll::compute(const StorageView&, StorageView&) const { THROW_MPS(GatherAll) }
#define INST_GATHERALL(T) template void GatherAll::compute<Device::MPS, T>(const StorageView&, StorageView&) const;
INST_GATHERALL(float)
INST_GATHERALL(float16_t)
INST_GATHERALL(bfloat16_t)
INST_GATHERALL(int8_t)
INST_GATHERALL(int32_t)
INST_GATHERALL(int16_t)

template <Device D, typename T>
void ReduceAll::compute(const StorageView&, StorageView&) const { THROW_MPS(ReduceAll) }
#define INST_REDUCEALL(T) template void ReduceAll::compute<Device::MPS, T>(const StorageView&, StorageView&) const;
INST_REDUCEALL(float)
INST_REDUCEALL(float16_t)
INST_REDUCEALL(bfloat16_t)
INST_REDUCEALL(int8_t)
INST_REDUCEALL(int32_t)
INST_REDUCEALL(int16_t)

template<>
void TopPMask::compute<Device::MPS, float>(const StorageView&, const StorageView&, StorageView&) const { THROW_MPS(TopPMask) }
template<>
void TopPMask::compute<Device::MPS, float16_t>(const StorageView&, const StorageView&, StorageView&) const { THROW_MPS(TopPMask) }
template<>
void TopPMask::compute<Device::MPS, bfloat16_t>(const StorageView&, const StorageView&, StorageView&) const { THROW_MPS(TopPMask) }

template<> dim_t TopPMask::max_num_classes<Device::MPS>() { return 0; }

} // namespace ops
} // namespace ctranslate2
#endif