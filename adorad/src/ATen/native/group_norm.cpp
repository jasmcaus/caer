#include <ATen/native/group_norm.h>

#include <array>
#include <functional>
#include <numeric>
#include <tuple>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/CPUApplyUtils.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>

namespace at {
namespace native {

std::tuple<Tensor, Tensor, Tensor> native_group_norm(
    const Tensor& X,
    const Tensor& gamma /* optional */,
    const Tensor& beta /* optional */,
    int64_t N,
    int64_t C,
    int64_t HxW,
    int64_t group,
    double eps) {
  Tensor Y = at::native::empty_like(X, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  Tensor mean = at::empty({N, group}, X.options());
  Tensor rstd = at::empty({N, group}, X.options());
  GroupNormKernel(
      X.device().type(), X, gamma, beta, N, C, HxW, group, eps, Y, mean, rstd);
  return std::make_tuple(Y, mean, rstd);
}

std::tuple<Tensor, Tensor, Tensor> native_group_norm_backward(
    const Tensor& dY,
    const Tensor& X,
    const Tensor& mean,
    const Tensor& rstd,
    const Tensor& gamma,
    int64_t N,
    int64_t C,
    int64_t HxW,
    int64_t group,
    std::array<bool, 3> grad_input_mask) {
  Tensor dX;
  Tensor dgamma;
  Tensor dbeta;
  if (grad_input_mask[0]) {
    dX = at::native::empty_like(X, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  }
  if (grad_input_mask[1]) {
    dgamma = at::native::empty_like(gamma, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  }
  if (grad_input_mask[2]) {
    dbeta = at::native::empty_like(gamma, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  }
  GroupNormBackwardKernel(
      X.device().type(),
      dY,
      X,
      mean,
      rstd,
      gamma,
      N,
      C,
      HxW,
      group,
      dX,
      dgamma,
      dbeta);
  return std::make_tuple(dX, dgamma, dbeta);
}

Tensor group_norm(
    const Tensor& input,
    int64_t num_groups,
    const Tensor& weight /* optional */,
    const Tensor& bias /* optional */,
    double eps,
    bool /* cudnn_enabled, deprecated */) {
  const int64_t N = input.size(0);
  const int64_t C = input.size(1);
  TORCH_CHECK(
      C % num_groups == 0,
      "Expected number of channels in input to be divisible by ",
      "num_groups, but got input of shape ",
      input.sizes(),
      " and "
      "num_groups=",
      num_groups);
  TORCH_CHECK(
      !weight.defined() || (weight.dim() == 1 && weight.numel() == C),
      "Expected weight to be a vector of size equal to the number of ",
      "channels in input, but got weight of shape ",
      weight.sizes(),
      " and input of shape ",
      input.sizes());
  TORCH_CHECK(
      !bias.defined() || (bias.dim() == 1 && bias.numel() == C),
      "Expected bias to be a vector of size equal to the number of ",
      "channels in input, but got bias of shape ",
      weight.sizes(),
      " and input of shape ",
      input.sizes());

  const auto input_shape = input.sizes();
  const int64_t HxW =
      prod_intlist(input_shape.cbegin() + 2, input_shape.cend());

  const Tensor kEmpty;
  const auto& X = input.is_contiguous() ? input : input.contiguous();
  const auto& gamma = weight.defined() ? weight.contiguous() : kEmpty;
  const auto& beta = bias.defined() ? bias.contiguous() : kEmpty;
  TORCH_CHECK(!gamma.defined() || gamma.numel() == C);
  TORCH_CHECK(!beta.defined() || beta.numel() == C);
  return std::get<0>(
      at::native_group_norm(X, gamma, beta, N, C, HxW, num_groups, eps));
}

DEFINE_DISPATCH(GroupNormKernel);
DEFINE_DISPATCH(GroupNormBackwardKernel);

// Ported from pytorch/xla repo
std::tuple<at::Tensor, at::Tensor, at::Tensor> math_group_norm(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    int64_t N,
    int64_t C,
    int64_t HxW,
    int64_t group,
    double eps) {
  auto input_shape = input.sizes();
  at::Tensor input_reshaped = input.view({1, N * group, N ? -1 : 1});
  auto outputs = at::native_batch_norm(
      input_reshaped,
      /*weight=*/{},
      /*bias=*/{},
      /*running_mean=*/{},
      /*running_var=*/{},
      /*training=*/true,
      /*momentum=*/0,
      eps);
  at::Tensor out = std::get<0>(outputs);
  out = out.view(input_shape);
  std::vector<int64_t> affine_param_shape(input.dim(), 1);
  affine_param_shape[1] = C;
  if (weight.defined() && bias.defined()) {
    out = bias.view(affine_param_shape)
              .addcmul(out, weight.view(affine_param_shape), 1);
  } else if (weight.defined()) {
    out = out.mul(weight.view(affine_param_shape));
  } else if (bias.defined()) {
    out = out.add(bias.view(affine_param_shape));
  }
  at::Tensor mean = std::get<1>(outputs).view({N, group});
  at::Tensor rstd = std::get<2>(outputs).view({N, group});
  return std::make_tuple(out, mean, rstd);
}
} // namespace native
} // namespace at
