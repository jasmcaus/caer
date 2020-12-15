#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>

#if !AT_MKLDNN_ENABLED()

namespace at {
namespace native {

Tensor& mkldnn_add_out(
    Tensor& result,
    const Tensor& self,
    const Tensor& other,
    Scalar alpha) {
  TORCH_CHECK(false, "mkldnn_add_out: ATen not compiled with MKLDNN support");
}

Tensor mkldnn_add(const Tensor& self, const Tensor& other, Scalar alpha) {
  TORCH_CHECK(false, "mkldnn_add: ATen not compiled with MKLDNN support");
}

Tensor& mkldnn_add_(Tensor& self, const Tensor& other, Scalar alpha) {
  TORCH_CHECK(false, "mkldnn_add_: ATen not compiled with MKLDNN support");
}

Tensor& mkldnn_mul_out(Tensor& result, const Tensor& self, const Tensor& other) {
  TORCH_CHECK(false, "mkldnn_mul_out: ATen not compiled with MKLDNN support");
}

Tensor mkldnn_mul(const Tensor& self, const Tensor& other) {
  TORCH_CHECK(false, "mkldnn_mul: ATen not compiled with MKLDNN support");
}

Tensor& mkldnn_mul_(Tensor& self, const Tensor& other) {
  TORCH_CHECK(false, "mkldnn_mul_: ATen not compiled with MKLDNN support");
}

} // namespace native
} // namespace at

#else // AT_MKLDNN_EBABLED

#include <ATen/native/mkldnn/MKLDNNCommon.h>

namespace at {
namespace native {

Tensor& mkldnn_add_out(
    Tensor& result,
    const Tensor& self,
    const Tensor& other,
    Scalar alpha) {
  ideep::tensor& x = itensor_from_mkldnn(self);
  ideep::tensor& y = itensor_from_mkldnn(other);

  ideep::tensor& z = itensor_from_mkldnn(result);
  const std::vector<float> scales{1.0, alpha.to<float>()};
  ideep::sum::compute(scales, {x, y}, z);

  return result;
}

Tensor mkldnn_add(const Tensor& self, const Tensor& other, Scalar alpha) {
  ideep::tensor& x = itensor_from_mkldnn(self);
  ideep::tensor& y = itensor_from_mkldnn(other);

  ideep::tensor z;
  const std::vector<float> scales{1.0, alpha.to<float>()};
  ideep::sum::compute(scales, {x, y}, z);

  return new_with_itensor_mkldnn(std::move(z), optTypeMetaToScalarType(self.options().dtype_opt()),
                                 self.options().device_opt());
}

Tensor& mkldnn_add_(Tensor& self, const Tensor& other, Scalar alpha) {
  return native::mkldnn_add_out(self, self, other, alpha);
}

Tensor& mkldnn_mul_out(Tensor& result, const Tensor& self, const Tensor& other) {
  TORCH_CHECK(result.sizes() == self.sizes(),
             "mkldnn_mul_out: the output size should be same as input size");
  ideep::tensor& z = itensor_from_mkldnn(result);
  ideep::tensor& x = itensor_from_mkldnn(self);

  // for zero_dim tensor
  if (other.ndimension() == 0) {
    ideep::eltwise_forward::compute(
      x, z, ideep::algorithm::eltwise_linear,
      ideep::prop_kind::forward_inference, /*alpha*/ other.item().to<float>());

    return result;
  } else {
    TORCH_CHECK(self.sizes() == other.sizes(),
               "mkldnn_mul_out: currently mkldnn not support broadcasting");
    ideep::tensor y = itensor_from_mkldnn(other);
    ideep::binary::compute(x, y, z, dnnl::algorithm::binary_mul);

    return result;
  }
}

Tensor mkldnn_mul(const Tensor& self, const Tensor& other) {
  Tensor result = empty_mkldnn(self.sizes(), optTypeMetaToScalarType(self.options().dtype_opt()),
                               self.options().layout_opt(), self.options().device_opt(),
                               self.options().pinned_memory_opt());
  return native::mkldnn_mul_out(result, self, other);
}

Tensor& mkldnn_mul_(Tensor& self, const Tensor& other) {
  return native::mkldnn_mul_out(self, self, other);
}

} // namespace native
} // namespace at

#endif // AT_MKLDNN_EBABLED
