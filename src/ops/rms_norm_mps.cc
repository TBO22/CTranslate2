#ifdef __APPLE__
#ifdef CT2_WITH_MPS

#include <memory>

#include "ctranslate2/ops/rms_norm.h"
#include "ctranslate2/primitives.h"

namespace ctranslate2 {
  namespace ops {

    template <>
    void RMSNorm::compute<Device::MPS, float>(const StorageView& gamma,
                                              const StorageView& input,
                                              StorageView& output) const {
      const dim_t size = input.size();
      StorageView input_cpu(input.shape(), input.dtype(), Device::CPU);
      StorageView gamma_cpu(gamma.shape(), gamma.dtype(), Device::CPU);
      StorageView output_cpu(output.shape(), output.dtype(), Device::CPU);
      cross_device_primitives<Device::MPS, Device::CPU>::copy(
          input.data<float>(), input_cpu.data<float>(), size);
      cross_device_primitives<Device::MPS, Device::CPU>::copy(
          gamma.data<float>(), gamma_cpu.data<float>(), gamma.size());

      compute<Device::CPU, float>(gamma_cpu, input_cpu, output_cpu);

      cross_device_primitives<Device::CPU, Device::MPS>::copy(
          output_cpu.data<float>(), output.data<float>(), size);
    }

  }
}

#endif
#endif
