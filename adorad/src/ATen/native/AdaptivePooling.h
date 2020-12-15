#pragma once

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>

namespace at { namespace native {

using adaptive_avg_pooling_fn = void(*)(Tensor& output, const Tensor& input, IntArrayRef output_size);
using adaptive_avg_pooling_backward_fn = void(*)(Tensor& grad_input, const Tensor& grad_output);
DECLARE_DISPATCH(adaptive_avg_pooling_fn, adaptive_avg_pool2d_kernel);
DECLARE_DISPATCH(adaptive_avg_pooling_backward_fn, adaptive_avg_pool2d_backward_kernel);

static inline int64_t start_index(int64_t a, int64_t b, int64_t c) {
  return (int64_t)std::floor((float)(a * c) / b);
}

static inline int64_t end_index(int64_t a, int64_t b, int64_t c) {
  return (int64_t)std::ceil((float)((a + 1) * c) / b);
}

}} // namespace at::native
