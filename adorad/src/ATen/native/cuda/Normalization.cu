#include <ATen/native/cuda/Normalization.cuh>

namespace at { namespace native {

std::tuple<Tensor&, Tensor&, Tensor&> batch_norm_cuda_out(Tensor& output, Tensor& save_mean, Tensor& save_invstd, const Tensor& self, const Tensor& weight, const Tensor& bias,
                                                   const Tensor& running_mean, const Tensor& running_var, bool train, double momentum, double epsilon) {
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, self.scalar_type(), "batch_norm_cuda", [&] {
    auto mean_st = running_mean.dtype();
    auto var_st = running_var.dtype();
    TORCH_CHECK(mean_st == var_st, "running_mean and running_var need to have the same data types");
    bool is_half_float = std::is_same<scalar_t, at::Half>::value && mean_st == at::kFloat;
    bool is_bfloat16_float = std::is_same<scalar_t, at::BFloat16>::value && mean_st == at::kFloat;
    if (cuda::detail::canUse32BitIndexMath(self)) {
      if (is_half_float || is_bfloat16_float) {
        batch_norm_cuda_template<scalar_t, float, int32_t>(output, save_mean, save_invstd, self, weight, bias, running_mean, running_var, train, momentum, epsilon);
      } else {
        batch_norm_cuda_template<scalar_t, scalar_t, int32_t>(output, save_mean, save_invstd, self, weight, bias, running_mean, running_var, train, momentum, epsilon);
      }
    } else {
      if (is_half_float || is_bfloat16_float) {
        batch_norm_cuda_template<scalar_t, float, int64_t>(output, save_mean, save_invstd, self, weight, bias, running_mean, running_var, train, momentum, epsilon);
      } else {
        batch_norm_cuda_template<scalar_t, scalar_t, int64_t>(output, save_mean, save_invstd, self, weight, bias, running_mean, running_var, train, momentum, epsilon);
      }
    }
  });
  return std::tuple<Tensor&, Tensor&, Tensor&>(output, save_mean, save_invstd);
}

std::tuple<Tensor, Tensor, Tensor> batch_norm_cuda(const Tensor& self, const Tensor& weight, const Tensor& bias,
                                                   const Tensor& running_mean, const Tensor& running_var, bool train, double momentum, double epsilon) {
  auto output = at::empty_like(self, at::MemoryFormat::Contiguous);
  int64_t n_input = self.size(1);
  auto input_options = self.options();
  // Accumulate in higher precision if input is half/bfloat16
  if (self.scalar_type() == at::ScalarType::Half || self.scalar_type() == at::ScalarType::BFloat16) {
    input_options = input_options.dtype(ScalarType::Float);
  }
  Tensor save_mean, save_invstd;
  if (train) {
    save_mean = at::empty({n_input}, input_options);
    save_invstd = at::empty({n_input}, input_options);
  } else {
    save_mean = at::empty({0}, input_options);
    save_invstd = at::empty({0}, input_options);
  }

  batch_norm_cuda_out(output, save_mean, save_invstd, self, weight, bias, running_mean, running_var, train, momentum, epsilon);
  return std::make_tuple(output, save_mean, save_invstd);
}

std::tuple<Tensor, Tensor, Tensor> batch_norm_backward_cuda(const Tensor& grad_out, const Tensor& self, const Tensor& weight, const Tensor& running_mean, const Tensor& running_var,
                                                            const Tensor& save_mean, const Tensor& save_invstd, bool train, double epsilon, std::array<bool,3> grad_input_mask) {
  return AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, self.scalar_type(), "batch_norm_backward_cuda", [&] {
    auto mean_st = running_mean.dtype();
    auto var_st = running_var.dtype();
    TORCH_CHECK(mean_st == var_st, "running_mean and running_var need to have the same data types");
    bool is_half_float = std::is_same<scalar_t, at::Half>::value && mean_st == at::kFloat;
    bool is_bfloat16_float = std::is_same<scalar_t, at::BFloat16>::value && mean_st == at::kFloat;
    if (cuda::detail::canUse32BitIndexMath(self)) {
      if (is_half_float || is_bfloat16_float) {
        return batch_norm_backward_cuda_template<scalar_t, float, int32_t>(grad_out, self, weight, running_mean, running_var, save_mean, save_invstd, train, epsilon, grad_input_mask);
      } else {
        return batch_norm_backward_cuda_template<scalar_t, scalar_t, int32_t>(grad_out, self, weight, running_mean, running_var, save_mean, save_invstd, train, epsilon, grad_input_mask);
      }
    } else {
      if (is_half_float || is_bfloat16_float) {
        return batch_norm_backward_cuda_template<scalar_t, float, int64_t>(grad_out, self, weight, running_mean, running_var, save_mean, save_invstd, train, epsilon, grad_input_mask);
      } else {
        return batch_norm_backward_cuda_template<scalar_t, scalar_t, int64_t>(grad_out, self, weight, running_mean, running_var, save_mean, save_invstd, train, epsilon, grad_input_mask);
      }
    }
  });
}

std::tuple<Tensor, Tensor> batch_norm_stats_cuda(const Tensor& self, double epsilon) {
  return AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, self.scalar_type(), "batch_norm_stats_cuda", [&] {
    if (cuda::detail::canUse32BitIndexMath(self)) {
      return batch_norm_stats_cuda_template<scalar_t, int32_t>(self, epsilon);
    } else {
      return batch_norm_stats_cuda_template<scalar_t, int64_t>(self, epsilon);
    }
  });
}

Tensor batch_norm_elemt_cuda(const Tensor& self, const Tensor& weight, const Tensor& bias,
                             const Tensor& mean, const Tensor& invstd, double epsilon) {
  auto output = at::empty_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  batch_norm_elemt_cuda_out(output, self, weight, bias, mean, invstd, epsilon);
  return output;
}

Tensor& batch_norm_elemt_cuda_out(Tensor& output, const Tensor& self, const Tensor& weight, const Tensor& bias,
                             const Tensor& mean, const Tensor& invstd, double epsilon) {
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, self.scalar_type(), "batch_norm_elemt", [&] {
    auto mean_st = mean.dtype();
    auto invstd_st = invstd.dtype();
    TORCH_CHECK(mean_st == invstd_st, "mean and invstd need to have the same data types");
    bool is_half_float = std::is_same<scalar_t, at::Half>::value && mean_st == at::kFloat;
    bool is_bfloat16_float = std::is_same<scalar_t, at::BFloat16>::value && mean_st == at::kFloat;
    if (cuda::detail::canUse32BitIndexMath(self)) {
      if (is_half_float || is_bfloat16_float) {
        batch_norm_elemt_cuda_template<scalar_t, float, int32_t>(output, self, weight, bias, mean, invstd, epsilon);
      } else {
        batch_norm_elemt_cuda_template<scalar_t, scalar_t, int32_t>(output, self, weight, bias, mean, invstd, epsilon);
      }
    } else {
      if (is_half_float || is_bfloat16_float) {
        batch_norm_elemt_cuda_template<scalar_t, float, int64_t>(output, self, weight, bias, mean, invstd, epsilon);
      } else {
        batch_norm_elemt_cuda_template<scalar_t, scalar_t, int64_t>(output, self, weight, bias, mean, invstd, epsilon);
      }
    }
  });
    return output;
}

// accepting input(self) here to determine template data types, since running_mean/running_var are optional
std::tuple<Tensor, Tensor> batch_norm_gather_stats_cuda(const Tensor& self, const Tensor& mean, const Tensor& invstd, const Tensor& running_mean,
                                                        const Tensor& running_var, double momentum, double epsilon, int64_t count) {
  std::vector<int64_t> counts(mean.size(0), count);
  Tensor counts_ = at::from_blob((void*)counts.data(), {(int64_t)counts.size()}, self.options().dtype(at::kLong).device(at::kCPU));
  counts_ = counts_.to(self.device()).to(running_mean.dtype());
  return batch_norm_gather_stats_with_counts_cuda(self, mean, invstd, running_mean, running_var, momentum, epsilon, counts_);
}


std::tuple<Tensor, Tensor> batch_norm_gather_stats_with_counts_cuda(const Tensor& self, const Tensor& mean, const Tensor& invstd, const Tensor& running_mean,
                                                        const Tensor& running_var, double momentum, double epsilon, const Tensor& counts) {
  
  return AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, running_mean.scalar_type(), "batch_norm_update_stats_cuda", [&] {
    using accscalar_t = at::acc_type<scalar_t, true>;
    if (cuda::detail::canUse32BitIndexMath(self)) {
      return batch_norm_gather_stats_cuda_template<scalar_t, accscalar_t, int32_t>(mean, invstd, running_mean, running_var, momentum, epsilon, counts);
    } else {
      return batch_norm_gather_stats_cuda_template<scalar_t, accscalar_t, int64_t>(mean, invstd, running_mean, running_var, momentum, epsilon, counts);
    }
  });
}

std::tuple<Tensor, Tensor, Tensor, Tensor> batch_norm_backward_reduce_cuda(const Tensor& self, const Tensor& input, const Tensor& mean, const Tensor& invstd,
                                                                           const Tensor& weight, bool input_g, bool weight_g, bool bias_g) {
  return AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, self.scalar_type(), "batch_norm_backward_reduce", [&] {
    auto mean_st = mean.dtype();
    auto invstd_st = invstd.dtype();
    TORCH_CHECK(mean_st == invstd_st, "mean and invstd need to have the same data types");
    bool is_half_float = std::is_same<scalar_t, at::Half>::value && mean_st == at::kFloat;
    bool is_bfloat16_float = std::is_same<scalar_t, at::BFloat16>::value && mean_st == at::kFloat;
    if (cuda::detail::canUse32BitIndexMath(self)) {
      if (is_half_float || is_bfloat16_float) {
        return batch_norm_backward_reduce_cuda_template<scalar_t, float, int32_t>(self, input, mean, invstd, weight, input_g, weight_g, bias_g);
      } else {
        return batch_norm_backward_reduce_cuda_template<scalar_t, scalar_t, int32_t>(self, input, mean, invstd, weight, input_g, weight_g, bias_g);
      }
    } else {
      if (is_half_float || is_bfloat16_float) {
        return batch_norm_backward_reduce_cuda_template<scalar_t, float, int64_t>(self, input, mean, invstd, weight, input_g, weight_g, bias_g);
      } else {
        return batch_norm_backward_reduce_cuda_template<scalar_t, scalar_t, int64_t>(self, input, mean, invstd, weight, input_g, weight_g, bias_g);
      }
    }
  });
}

Tensor batch_norm_backward_elemt_cuda(const Tensor& self, const Tensor& input, const Tensor& mean, const Tensor& invstd,
                                      const Tensor& weight, const Tensor& mean_dy, const Tensor& mean_dy_xmu) {
  return AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, self.scalar_type(), "batch_norm_backward_elemt", [&] {
    auto mean_st = mean.dtype();
    auto invstd_st = invstd.dtype();
    TORCH_CHECK(mean_st == invstd_st, "mean and invstd need to have the same data types");
    bool is_half_float = std::is_same<scalar_t, at::Half>::value && mean_st == at::kFloat;
    bool is_bfloat16_float = std::is_same<scalar_t, at::BFloat16>::value && mean_st == at::kFloat;
    if (cuda::detail::canUse32BitIndexMath(self)) {
      if (is_half_float || is_bfloat16_float) {
        return batch_norm_backward_elemt_cuda_template<scalar_t, float, int32_t>(self, input, mean, invstd, weight, mean_dy, mean_dy_xmu);
      } else {
        return batch_norm_backward_elemt_cuda_template<scalar_t, scalar_t, int32_t>(self, input, mean, invstd, weight, mean_dy, mean_dy_xmu);
      }
    } else {
      if (is_half_float || is_bfloat16_float) {
        return batch_norm_backward_elemt_cuda_template<scalar_t, float, int64_t>(self, input, mean, invstd, weight, mean_dy, mean_dy_xmu);
      } else {
        return batch_norm_backward_elemt_cuda_template<scalar_t, scalar_t, int64_t>(self, input, mean, invstd, weight, mean_dy, mean_dy_xmu);
      }
    }
  });
}

std::tuple<Tensor, Tensor> batch_norm_update_stats_cuda(
        const Tensor& self, const Tensor& running_mean, const Tensor& running_var, double momentum) {
  return AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, self.scalar_type(), "batch_norm_backward", [&] {
    auto mean_st = running_mean.dtype();
    auto var_st = running_var.dtype();
    TORCH_CHECK(mean_st == var_st, "running_mean and running_var need to have the same data types");
    // <sigh> Some workloads depend on passing in half input and float stats, which is
    // usually handled by cuDNN. However, the JIT sometimes replaces cuDNN calls with this
    // one so it needs to support the same case, or people start to complain.
    bool is_half_float = std::is_same<scalar_t, at::Half>::value && mean_st == at::kFloat;
    bool is_bfloat16_float = std::is_same<scalar_t, at::BFloat16>::value && mean_st == at::kFloat;
    if (cuda::detail::canUse32BitIndexMath(self)) {
      if (is_half_float || is_bfloat16_float) {
        return batch_norm_update_stats_cuda_template<scalar_t, float, int32_t>(self, running_mean, running_var, momentum);
      } else {
        return batch_norm_update_stats_cuda_template<scalar_t, scalar_t, int32_t>(self, running_mean, running_var, momentum);
      }
    } else {
      if (is_half_float || is_bfloat16_float) {
        return batch_norm_update_stats_cuda_template<scalar_t, float, int64_t>(self, running_mean, running_var, momentum);
      } else {
        return batch_norm_update_stats_cuda_template<scalar_t, scalar_t, int64_t>(self, running_mean, running_var, momentum);
      }
    }
  });
}

} } // namespace at::native
