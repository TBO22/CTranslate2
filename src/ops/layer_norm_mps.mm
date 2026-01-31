#ifdef __APPLE__
#ifdef CT2_WITH_MPS

#include <memory>

#include "ctranslate2/ops/layer_norm.h"
#include "ctranslate2/primitives.h"

namespace ctranslate2 {
namespace ops {

template <>
void LayerNorm::compute<Device::MPS, float>(
    const StorageView* beta,
    const StorageView* gamma,
    const StorageView& input,
    const dim_t axis,
    const dim_t outer_size,
    const dim_t axis_size,
    const dim_t inner_size,
    StorageView& output) const {

  const dim_t size = input.size();

  // --- Move tensors to CPU ---
  StorageView input_cpu(input.shape(), input.dtype(), Device::CPU);
  StorageView output_cpu(output.shape(), output.dtype(), Device::CPU);

  cross_device_primitives<Device::MPS, Device::CPU>::copy(
      input.data<float>(), input_cpu.data<float>(), size);

  std::unique_ptr<StorageView> beta_cpu;
  std::unique_ptr<StorageView> gamma_cpu;

  if (beta) {
    beta_cpu = std::make_unique<StorageView>(beta->shape(), beta->dtype(), Device::CPU);
    cross_device_primitives<Device::MPS, Device::CPU>::copy(
        beta->data<float>(), beta_cpu->data<float>(), beta->size());
  }

  if (gamma) {
    gamma_cpu = std::make_unique<StorageView>(gamma->shape(), gamma->dtype(), Device::CPU);
    cross_device_primitives<Device::MPS, Device::CPU>::copy(
        gamma->data<float>(), gamma_cpu->data<float>(), gamma->size());
  }

  // --- CPU implementation (correct + stable) ---
  compute<Device::CPU, float>(
      beta_cpu.get(),
      gamma_cpu.get(),
      input_cpu,
      axis,
      outer_size,
      axis_size,
      inner_size,
      output_cpu);

  // --- Copy back to MPS ---
  cross_device_primitives<Device::CPU, Device::MPS>::copy(
      output_cpu.data<float>(), output.data<float>(), size);
}

}  // namespace ops
}  // namespace ctranslate2

#endif
#endif
