#include <ATen/native/Lerp.h>

#include <ATen/ATen.h>
#include <ATen/CPUApplyUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>

namespace at {
namespace native {

Tensor& lerp_cpu_tensor_out(Tensor& result, const Tensor& self,
                            const Tensor& end, const Tensor& weight) {
  TORCH_CHECK(weight.dim() <= std::max(self.dim(), end.dim()),
           "weight should be of dimension max(self.dim(), end.dim()) or lesser");
  lerp_kernel_tensor_weight(kCPU, result, self, end, weight);
  return result;
}

Tensor& lerp_cpu_scalar_out(Tensor& result, const Tensor& self,
                            const Tensor& end, Scalar weight) {
  lerp_kernel_scalar_weight(kCPU, result, self, end, weight);
  return result;
}

Tensor& lerp_cpu_tensor_(Tensor& self, const Tensor& end, const Tensor& weight) {
  TORCH_CHECK(weight.dim() <= std::max(self.dim(), end.dim()),
           "weight should be of dimension max(self.dim(), end.dim()) or lesser");
  lerp_kernel_tensor_weight(kCPU, self, self, end, weight);
  return self;
}

Tensor& lerp_cpu_scalar_(Tensor& self, const Tensor& end, Scalar weight) {
  lerp_kernel_scalar_weight(kCPU, self, self, end, weight);
  return self;
}

Tensor lerp_cpu_tensor(const Tensor& self, const Tensor& end, const Tensor& weight) {
  TORCH_CHECK(weight.dim() <= std::max(self.dim(), end.dim()),
           "weight should be of dimension max(self.dim(), end.dim()) or lesser");
  Tensor result = at::empty({0}, self.options());
  lerp_kernel_tensor_weight(kCPU, result, self, end, weight);
  return result;
}

Tensor lerp_cpu_scalar(const Tensor& self, const Tensor& end, Scalar weight) {
  Tensor result = at::empty({0}, self.options());
  lerp_kernel_scalar_weight(kCPU, result, self, end, weight);
  return result;
}

DEFINE_DISPATCH(lerp_kernel_scalar_weight);
DEFINE_DISPATCH(lerp_kernel_tensor_weight);

} // namespace native
} // namespace at
