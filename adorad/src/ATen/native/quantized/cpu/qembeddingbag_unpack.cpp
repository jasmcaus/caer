#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/native/quantized/cpu/embedding_packed_params.h>
#include <ATen/native/quantized/cpu/fbgemm_utils.h>
#include <torch/library.h>

torch::class_<EmbeddingPackedParamsBase> register_embedding_params();

at::Tensor PackedEmbeddingBagWeight::unpack() {
  auto packed_weight = packed_w;
  at::Tensor weight_origin;

  if (bit_rate_ == 8 || bit_rate_ == 4) {
    const auto input_rows = packed_weight.size(0);
    const auto input_columns = packed_weight.size(1);
    int scale_bias_bytes;
    const auto num_elem_per_byte = 8 / bit_rate_;
    if (bit_rate_ == 8) {
      // The last 2 values are used to store the FP32 scale and zero_point
      // values per row.
      scale_bias_bytes = 8;
    } else {
      scale_bias_bytes = 4;
    }

    const auto* input = packed_weight.data_ptr<uint8_t>();
    // Calculate the output shape, accounting for the last n bytes to be used
    // for scale/bias rest of the entries are packed depending on the bit_width.
    std::vector<int64_t> output_shape = {
        input_rows,
        static_cast<std::int64_t>(input_columns - scale_bias_bytes) *
            num_elem_per_byte};

    auto scales = at::from_blob(
        w_scale.data(), w_scale.size(), device(c10::kCPU).dtype(c10::kFloat));
    auto zero_points = at::from_blob(
        w_zp.data(), w_zp.size(), device(c10::kCPU).dtype(c10::kFloat));

    auto output_columns = output_shape[1];
    uint8_t* output_data;

    // Allocate output weight tensor based on the bit_width
    if (bit_rate_ == 8) {
      weight_origin = at::_empty_per_channel_affine_quantized(
          output_shape,
          scales.toType(c10::kFloat),
          zero_points.toType(c10::kFloat),
          0, // The output channel axis is 0
          device(c10::kCPU).dtype(c10::kQUInt8));
      output_data = static_cast<uint8_t*>(weight_origin.data_ptr());
    } else {
      // We create empty qtensor with the full output shape, and dtype set to
      // quint4x2 This will internally allocate appropriate storage bytes to
      // account for the packed nature of this dtype.
      weight_origin = at::_empty_per_channel_affine_quantized(
          output_shape,
          scales.toType(c10::kFloat),
          zero_points.toType(c10::kFloat),
          0, // The output channel axis is 0
          device(c10::kCPU).dtype(c10::kQUInt4x2));
      output_data = static_cast<uint8_t*>(weight_origin.data_ptr());
    }

    // Copy over the data from the packed weight to the output.
    // For sub-byte tensors this will copy the packed bytes over since the
    // sub_byte qtensors are expected to store data in packed format.
    at::parallel_for(0, input_rows, 1, [&](int32_t start_idx, int32_t end_idx) {
      for (int64_t row = start_idx; row < end_idx; ++row) {
        const std::uint8_t* input_row = input + row * input_columns;
        uint8_t* output_row =
            output_data + row * output_columns / num_elem_per_byte;

        for (std::size_t col = 0; col < output_columns / num_elem_per_byte;
             ++col) {
          output_row[col] = input_row[col];
        } // output_columns
      }
    });

    return weight_origin;
  }
  TORCH_INTERNAL_ASSERT(
      false,
      "We currently only support 8-bit and 4-bit quantization of embedding_bag.");
  return weight_origin;
}

namespace at {
namespace native {
namespace {

Tensor qembeddingbag_byte_unpack(const Tensor& packed_weight) {
  const auto input_rows = packed_weight.size(0);
  const auto input_columns = packed_weight.size(1);

  // The last 2 values are used to store the FP32 scale and zero_point values
  // per row.
  int output_columns = input_columns - 2 * sizeof(float);

  const auto* input = packed_weight.data_ptr<uint8_t>();
  std::vector<int64_t> output_shape = {input_rows, output_columns};
  at::Tensor output = at::empty(
      output_shape,
      packed_weight.options().dtype(kFloat),
      packed_weight.suggest_memory_format());
  float* output_data = output.data_ptr<float>();

#ifdef USE_FBGEMM
    at::parallel_for(
      0, input_rows, 1, [&](int32_t start_idx, int32_t end_idx) {
        for (int64_t row = start_idx; row < end_idx; ++row) {
          fbgemm::Fused8BitRowwiseQuantizedSBFloatToFloat(
            input + row * input_columns,
            1,
            input_columns,
            output_data + row * output_columns);
        }
      });
#else
  for (std::size_t row = 0; row < input_rows; ++row) {
    const std::uint8_t* input_row = input + row * input_columns;
    const float* input_row_scale_zp =
        reinterpret_cast<const float*>(input_row + output_columns);
    float* output_row = output_data + row * output_columns;

    for (std::size_t col = 0; col < output_columns; ++col) {
      output_row[col] =
          input_row[col] * input_row_scale_zp[0] + input_row_scale_zp[1];
    } // output_columns
  } // input_rows
#endif // USE_FBGEMM
  return output;
}

Tensor _qembeddingbag_nbit_unpack_helper(
    const Tensor& packed_weight,
    int BIT_RATE) {
  const auto input_rows = packed_weight.size(0);
  const auto input_columns = packed_weight.size(1);
  const auto* input_data = packed_weight.data_ptr<uint8_t>();
  int NUM_ELEM_PER_BYTE = 8 / BIT_RATE;

  // The last 4 bytes per row are two fp16 scale and zero_point.
  // The rest of input_columns is the number of values in the original row.
  std::vector<int64_t> output_dimensions = {
      input_rows,
      static_cast<std::int64_t>(input_columns - 2 * sizeof(at::Half)) *
          NUM_ELEM_PER_BYTE};

  auto output = at::empty(
      output_dimensions,
      packed_weight.options().dtype(kFloat),
      packed_weight.suggest_memory_format());
  float* output_data = output.data_ptr<float>();
#ifdef USE_FBGEMM
    at::parallel_for(
      0, input_rows, 1, [&](int32_t start_idx, int32_t end_idx) {
        for (int64_t row = start_idx; row < end_idx; ++row) {
          fbgemm::FusedNBitRowwiseQuantizedSBHalfToFloat(BIT_RATE,
            input_data + row * input_columns,
            1,
            input_columns,
            output_data + row * output_dimensions[1]);
        }
      });
#else
  auto output_columns = output_dimensions[1];
  for (size_t row = 0; row < input_rows; ++row) {
    float* output_row = output_data + row * output_columns;
    const std::uint8_t* input_row = input_data + row * input_columns;
    const at::Half* input_row_scale_zp = reinterpret_cast<const at::Half*>(
        input_row +
        (output_columns + NUM_ELEM_PER_BYTE - 1) / NUM_ELEM_PER_BYTE);
    float scale = input_row_scale_zp[0];
    float zero_point = input_row_scale_zp[1];

    for (int col = 0; col < output_columns; ++col) {
      std::uint8_t quantized = input_row[col / NUM_ELEM_PER_BYTE];
      quantized >>= (col % NUM_ELEM_PER_BYTE) * BIT_RATE;
      quantized &= (1 << BIT_RATE) - 1;
      output_row[col] = scale * quantized + zero_point;
    } // output_columns
  } // input_rows
#endif // USE_FBGEMM

  return output;
}

// De-quantizes the result of the qembeddingbag_4bit_prepack operator.
// The input is expected to first have quantized values,
// then 2-byte fp16 scale and 2-byte zero_offset.
// The output is a matrix containing only the values, but de-quantized.
// De-quantization is performed by multiplying each value by its
// row's scale and zero_point parameters. The de-quantized values
// will thus not be exactly equal to the original, un-quantized
// floating point values.
Tensor qembeddingbag_4bit_unpack(const Tensor& packed_weight) {
  return _qembeddingbag_nbit_unpack_helper(packed_weight, 4 /*BIT_RATE*/);
}

// De-quantizes the result of the qembeddingbag_2bit_prepack operator.
// The input is expected to first have quantized values,
// then 2-byte fp16 scale and 2-byte zero_offset.
// The output is a matrix containing only the values, but de-quantized.
// De-quantization is performed by multiplying each value by its
// row's scale and zero_point parameters. The de-quantized values
// will thus not be exactly equal to the original, un-quantized
// floating point values.
Tensor qembeddingbag_2bit_unpack(const Tensor& packed_weight) {
  return _qembeddingbag_nbit_unpack_helper(packed_weight, 2 /*BIT_RATE*/);
}

class QEmbeddingUnpackWeights final {
 public:
  static at::Tensor run(
      const c10::intrusive_ptr<EmbeddingPackedParamsBase>& packed_weight) {
    return packed_weight->unpack();
  }
};

TORCH_LIBRARY_IMPL(quantized, CPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("quantized::embedding_bag_byte_unpack"),
      qembeddingbag_byte_unpack);
  m.impl(
      TORCH_SELECTIVE_NAME("quantized::embedding_bag_4bit_unpack"),
      qembeddingbag_4bit_unpack);
  m.impl(
      TORCH_SELECTIVE_NAME("quantized::embedding_bag_2bit_unpack"),
      qembeddingbag_2bit_unpack);
}

TORCH_LIBRARY_IMPL(quantized, CatchAll, m) {
  // Unpack the packed embedding_bag weights using TorchBind custom class.
  // TODO extend to support 4-bit qtensor.
  m.impl(
      TORCH_SELECTIVE_NAME("quantized::embedding_bag_unpack"),
      TORCH_FN(QEmbeddingUnpackWeights::run));
}

} // namespace
} // namespace native
} // namespace at
