#ifdef __APPLE__
#ifdef CT2_WITH_MPS

#include <memory>

#include "ctranslate2/ops/softmax.h"
#include "ctranslate2/primitives.h"

namespace ctranslate2 {
  namespace ops {

    template <>
    void SoftMax::compute<Device::MPS, float>(const StorageView& input,
                                              const StorageView* lengths,
                                              StorageView& output) const {
      const dim_t size = input.size();

      StorageView input_cpu(input.shape(), input.dtype(), Device::CPU);
      StorageView output_cpu(output.shape(), output.dtype(), Device::CPU);
      cross_device_primitives<Device::MPS, Device::CPU>::copy(
          input.data<float>(), input_cpu.data<float>(), size);

      std::unique_ptr<StorageView> lengths_cpu;
      if (lengths) {
        lengths_cpu = std::make_unique<StorageView>(lengths->shape(), lengths->dtype(), Device::CPU);
        cross_device_primitives<Device::MPS, Device::CPU>::copy(
            lengths->data<int32_t>(), lengths_cpu->data<int32_t>(), lengths->size());
      }

      SoftMax::compute<Device::CPU, float>(input_cpu, lengths_cpu.get(), output_cpu);

      cross_device_primitives<Device::CPU, Device::MPS>::copy(
          output_cpu.data<float>(), output.data<float>(), size);
    }

  }
}

#endif
#endif
