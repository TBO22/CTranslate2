#ifdef __APPLE__
#ifdef CT2_WITH_MPS

#include <memory>

#include "ctranslate2/ops/mean.h"
#include "ctranslate2/primitives.h"

namespace ctranslate2 {
namespace ops {

template <>
void Mean::compute<Device::MPS, float>(
    const StorageView& input,
    const dim_t outer_size,
    const dim_t axis_size,
    const dim_t inner_size,
    const bool get_sum,
    StorageView& output) const {

  const dim_t input_size = input.size();
  const dim_t output_size = output.size();

  // --- Move input to CPU ---
  StorageView input_cpu(input.shape(), input.dtype(), Device::CPU);
  StorageView output_cpu(output.shape(), output.dtype(), Device::CPU);

  cross_device_primitives<Device::MPS, Device::CPU>::copy(
      input.data<float>(), input_cpu.data<float>(), input_size);

  // --- Run CPU Mean implementation ---
  compute<Device::CPU, float>(
      input_cpu,
      outer_size,
      axis_size,
      inner_size,
      get_sum,
      output_cpu);

  // --- Copy result back to MPS ---
  cross_device_primitives<Device::CPU, Device::MPS>::copy(
      output_cpu.data<float>(), output.data<float>(), output_size);
}

}  // namespace ops
}  // namespace ctranslate2

#endif
#endif
