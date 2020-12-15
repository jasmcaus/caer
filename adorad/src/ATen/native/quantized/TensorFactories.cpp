#include <ATen/ATen.h>
#include <ATen/quantized/Quantizer.h>
#include <c10/core/QScheme.h>
#include <c10/core/TensorOptions.h>

namespace at {
namespace native {

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ empty ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// We explicitly pass in scale and zero_point because we don't have the infra
// ready to support quantizer in python frontend, once that is ready, we'll
// change to use quantizer
Tensor empty_affine_quantized(
    IntArrayRef size,
    const TensorOptions& options_,
    double scale,
    int64_t zero_point,
    c10::optional<c10::MemoryFormat> optional_memory_format) {
  TORCH_CHECK(
    !(options_.has_memory_format() && optional_memory_format.has_value()),
    "Cannot set memory_format both in TensorOptions and explicit argument; please delete "
    "the redundant setter.");
  auto options = options_.merge_memory_format(optional_memory_format);
  TORCH_CHECK(
      options.has_dtype(),
      "Must provide data type for Tensor creation functions.");
  return new_qtensor(
      size,
      options,
      make_per_tensor_affine_quantizer(
          scale, zero_point, typeMetaToScalarType(options.dtype())));
}

Tensor empty_per_channel_affine_quantized(
    IntArrayRef size,
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis,
    const TensorOptions& options_,
    c10::optional<c10::MemoryFormat> optional_memory_format) {
  TORCH_CHECK(
    !(options_.has_memory_format() && optional_memory_format.has_value()),
    "Cannot set memory_format both in TensorOptions and explicit argument; please delete "
    "the redundant setter.");
  auto options = options_.merge_memory_format(optional_memory_format);
  TORCH_CHECK(
      options.has_dtype(),
      "Must provide data type for Tensor creation functions.");
  QuantizerPtr quantizer = make_per_channel_affine_quantizer(
          scales, zero_points, axis, typeMetaToScalarType(options.dtype()));
  return new_qtensor(
      size,
      options,
      quantizer);
}

// Provide better error message if dtype is wrong
Tensor empty_affine_quantized_other_backends_stub(
    IntArrayRef,
    const TensorOptions&,
    double,
    int64_t,
    c10::optional<c10::MemoryFormat>) {
  TORCH_CHECK(false, "Creation of quantized tensor requires quantized dtype like torch.quint8");
}

Tensor empty_per_channel_affine_quantized_other_backends_stub(
    IntArrayRef,
    const Tensor&,
    const Tensor&,
    int64_t,
    const TensorOptions&,
    c10::optional<c10::MemoryFormat>) {
  TORCH_CHECK(false, "Creation of quantized tensor requires quantized dtype like torch.quint8");
}

// Create an empty quantized Tensor with size, based on the options
// and quantization parameters of the input quantized Tensor
Tensor empty_quantized(IntArrayRef size, const Tensor& qtensor) {
  Tensor output;
  if (qtensor.qscheme() == kPerTensorAffine) {
    output = at::_empty_affine_quantized(size, qtensor.options(),
                                         qtensor.q_scale(),
                                         qtensor.q_zero_point());
  } else if (qtensor.qscheme() == kPerChannelAffine) {
    output = at::_empty_per_channel_affine_quantized(
        size,
        qtensor.q_per_channel_scales(),
        qtensor.q_per_channel_zero_points(),
        qtensor.q_per_channel_axis(),
        qtensor.options());
  } else {
    TORCH_CHECK(false,
                "QScheme not supported by empty_quantized:",
                toString(qtensor.qscheme()));
  }
  return output;
}

} // namespace native
} // namespace at
