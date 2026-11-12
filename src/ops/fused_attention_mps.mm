#include "ctranslate2/ops/fused_attention.h"

#include "mps/kernels.h"
#include "mps/ops_utils.h"
#include "mps/utils.h"
#include <cstring>
#include <cstdlib>

namespace ctranslate2 {
  namespace ops {

    static bool fused_attention_enabled() {
      static const bool enabled = mps::env_flag("CT2_MPS_USE_FUSED_ATTENTION");
      return enabled;
    }

    static int streaming_attention_mode() {
      // Experimental until real-model quality and device-generation coverage
      // justify changing the established attention path. "1" selects decode;
      // "prefill" also selects the tiled prefill implementation.
      static const int mode = []() {
        const char* value = std::getenv("CT2_MPS_USE_STREAMING_ATTENTION");
        if (!value || !value[0] || value[0] == '0')
          return 0;
        return std::strcmp(value, "prefill") == 0 ? 2 : 1;
      }();
      return mode;
    }

    static bool native_decode_enabled() {
      static const bool enabled = !mps::env_flag("CT2_MPS_DISABLE_FLASH_DECODE");
      return enabled;
    }

    template<>
    bool FusedAttention::supports_interleaved_queries<Device::MPS>(DataType dtype) const {
      return (native_decode_enabled() && mps::env_flag("CT2_MPS_USE_FLASH_CROSS_ATTENTION")
              && (dtype == DataType::FLOAT16 || dtype == DataType::BFLOAT16))
        || (fused_attention_enabled() && dtype == DataType::FLOAT16)
        || (streaming_attention_mode()
            && (dtype == DataType::FLOAT16 || dtype == DataType::BFLOAT16));
    }

    template<>
    bool FusedAttention::compute<Device::MPS>(const StorageView& queries,
                                               const StorageView& keys,
                                               const StorageView& values,
                                               const StorageView* values_lengths,
                                               float queries_scale,
                                               bool interleaved_queries,
                                               StorageView& output) const {
      if ((!native_decode_enabled() && !fused_attention_enabled() && !streaming_attention_mode())
          || keys.dtype() != queries.dtype() || values.dtype() != queries.dtype()
          || (values_lengths && (values_lengths->device() != queries.device()
                                 || values_lengths->dtype() != DataType::INT32)))
        return false;

      const bool standard_layout = !interleaved_queries
                                   && queries.rank() == 4
                                   && keys.rank() == 4
                                   && values.rank() == 4
                                   && queries.dim(0) == keys.dim(0)
                                   && queries.dim(0) == values.dim(0)
                                   && queries.dim(1) == keys.dim(1)
                                   && queries.dim(1) == values.dim(1)
                                   && keys.dim(2) == values.dim(2)
                                   && queries.dim(3) == keys.dim(3)
                                   && queries.dim(3) == values.dim(3);
      const bool interleaved_layout = interleaved_queries
                                      && queries.rank() == 4
                                      && keys.rank() == 4
                                      && values.rank() == 4
                                      && queries.dim(0) == keys.dim(0)
                                      && queries.dim(0) == values.dim(0)
                                      && queries.dim(2) == keys.dim(1)
                                      && queries.dim(2) == values.dim(1)
                                      && keys.dim(2) == values.dim(2)
                                      && queries.dim(3) == keys.dim(3)
                                      && queries.dim(3) == values.dim(3);
      if (!standard_layout && !interleaved_layout)
        return false;

      output.resize_as(queries);
      const dim_t num_heads = interleaved_layout ? queries.dim(2) : queries.dim(1);
      const dim_t query_length = interleaved_layout ? queries.dim(1) : queries.dim(2);
      const dim_t matrix_batch = queries.dim(0) * num_heads;
      if (values_lengths && values_lengths->size() != matrix_batch * query_length)
        return false;
      // Standard BHTD self-attention and shared encoder KV with BQHD beam
      // queries (Whisper/Marian). Keep the existing encoder/prefill path and
      // attention-weight requests; no KV transposes or copies are needed here.
      if (native_decode_enabled()
          && (queries.dtype() == DataType::FLOAT16 || queries.dtype() == DataType::BFLOAT16)
          && (query_length == 1 || (interleaved_layout && query_length <= 16))
          && queries.dim(3) <= 256 && keys.dim(2) > 0) {
        const dim_t splits = (keys.dim(2) + 255) / 256;
        StorageView partials(DataType::FLOAT32, Device::MPS);
        if (splits > 1)
          partials.resize({matrix_batch, query_length, splits, queries.dim(3) + 2});
        if (mps::flash_decode(queries.dtype(), queries.buffer(), keys.buffer(), values.buffer(),
                              values_lengths ? values_lengths->data<int32_t>() : nullptr,
                              output.buffer(), splits > 1 ? partials.data<float>() : nullptr,
                              partials.size(), queries.dim(0), num_heads, num_heads,
                              keys.dim(2), keys.dim(2), queries.dim(3), queries_scale,
                              0, 0, false, query_length, false, interleaved_layout))
          return true;
      }
      if (streaming_attention_mode()
          && (query_length == 1 || streaming_attention_mode() == 2)
          && mps::streaming_attention(queries.dtype(),
                                      queries.buffer(), keys.buffer(), values.buffer(),
                                      values_lengths ? values_lengths->data<int32_t>() : nullptr,
                                      output.buffer(), queries.dim(0), num_heads, num_heads,
                                      query_length, keys.dim(2), queries.dim(3), queries_scale,
                                      false, 0, 0, interleaved_layout))
        return true;
      if (!fused_attention_enabled() || queries.dtype() != DataType::FLOAT16)
        return false;
      return mps::fused_attention(DataType::FLOAT16,
                                  queries.data<float16_t>(),
                                  keys.data<float16_t>(),
                                  values.data<float16_t>(),
                                  values_lengths ? values_lengths->data<int32_t>() : nullptr,
                                  output.data<float16_t>(),
                                  matrix_batch,
                                  query_length,
                                  keys.dim(2),
                                  queries.dim(3),
                                  queries_scale,
                                  interleaved_layout ? num_heads : 0);
    }

  }
}
