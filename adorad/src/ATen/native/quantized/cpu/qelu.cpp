#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <torch/library.h>
#include <ATen/quantized/Quantizer.h>
#include <ATen/native/quantized/cpu/quantized_ops.h>

namespace at {
namespace native {

DEFINE_DISPATCH(qelu_stub);

Tensor quantized_elu(
    const Tensor& qx, double output_scale, int64_t output_zero_point, Scalar alpha, Scalar scale, Scalar input_scale) {
  Tensor qy = at::_empty_affine_quantized(qx.sizes(), qx.options(), output_scale, output_zero_point);
  qelu_stub(qx.device().type(), qx, alpha, scale, input_scale, qy);
  return qy;
}

Tensor quantized_celu(const Tensor& qx, double output_scale, int64_t output_zero_point, Scalar alpha) {
  TORCH_CHECK(alpha.to<double>() != 0,
      "ZeroDivisionError: alpha cannot be 0 for CELU");
  double inv_alpha = 1. / alpha.to<double>();
  return quantized_elu(qx, output_scale, output_zero_point, alpha, Scalar(1.0), Scalar(inv_alpha));
}

TORCH_LIBRARY_IMPL(quantized, QuantizedCPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("quantized::elu"), quantized_elu);
  m.impl(TORCH_SELECTIVE_NAME("quantized::celu"), quantized_celu);
}

}}  // namespace at::native
