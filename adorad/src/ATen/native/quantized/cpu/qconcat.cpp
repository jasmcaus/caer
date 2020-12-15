#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <torch/library.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/native/quantized/cpu/quantized_ops.h>
#include <ATen/native/TensorIterator.h>

#include <algorithm>
#include <vector>

namespace at {
namespace native {

DEFINE_DISPATCH(qcat_nhwc_stub);
DEFINE_DISPATCH(qcat_relu_nhwc_stub);

namespace {

bool is_cat_nhwc_fast_path(const c10::List<Tensor>& qxs, int dim) {
  TORCH_CHECK(qxs.size() > 0);
  bool is_fast_path = dim == 1;
  for (const at::Tensor& qx : qxs) {
    is_fast_path &= qx.dim() == 4;
    is_fast_path &= qx.is_contiguous(c10::MemoryFormat::ChannelsLast);
  }
  return is_fast_path;
}

bool is_valid_quantization_scheme(const Tensor& t) {
  const auto qtype = t.qscheme();
  return (qtype == kPerTensorAffine) || (qtype == kPerTensorSymmetric);
}

/* Quantized concatenation.
 *
 * Note: This function uses a dequantization.
 */
template <bool ReLUFused>
Tensor quantized_cat_impl(
    const c10::List<Tensor>& qxs,
    int64_t dim,
    double scale,
    int64_t zero_point) {
  if (is_cat_nhwc_fast_path(qxs, dim)) {
    if (ReLUFused) {
      return qcat_relu_nhwc_stub(at::kCPU, qxs, dim, scale, zero_point);
    } else {
      return qcat_nhwc_stub(at::kCPU, qxs, dim, scale, zero_point);
    }
  }

  const auto x_dtype = qxs.get(0).scalar_type();
  const auto x_qscheme = qxs.get(0).qscheme();
  std::vector<Tensor> xs;
  xs.reserve(qxs.size());
  for (const at::Tensor& qx : qxs) {
    TORCH_CHECK(x_dtype == qx.scalar_type(), "All dtypes must be the same.");
    TORCH_CHECK(
        x_qscheme == qx.qscheme(), "Quantization schemes must be the same.");
    xs.push_back(qx.dequantize());
  }
  const Tensor y = at::cat(xs, dim);
  Tensor qy;
  AT_DISPATCH_QINT_TYPES(x_dtype, "qcat", [&]() {
    qy = at::quantize_per_tensor(y, scale, zero_point, SCALAR_TYPE);
    if (ReLUFused) {
      auto iter = TensorIterator::unary_op(qy, qy);
      cpu_kernel(iter, [&](scalar_t value) -> scalar_t {
        return scalar_t(std::max<underlying_t>(value.val_, zero_point));
      });
    }
  });
  return qy;
}

template <bool ReLUFused = false>
Tensor qcat(
    const c10::List<Tensor>& qxs,
    int64_t dim,
    c10::optional<double> scale,
    c10::optional<int64_t> zero_point) {
  TORCH_CHECK(is_valid_quantization_scheme(qxs[0]),
              "Only per-tensor quantization is supported in 'cat'!")
  double _scale = scale.has_value() ? scale.value() : qxs.get(0).q_scale();
  int64_t _zero_point =
      zero_point.has_value() ? zero_point.value() : qxs.get(0).q_zero_point();
  return quantized_cat_impl<ReLUFused>(qxs, dim, _scale, _zero_point);
}

template <bool ReLUFused = false>
Tensor qcat_out(const c10::List<Tensor>& qxs, int64_t dim, Tensor out) {
  TORCH_CHECK(is_valid_quantization_scheme(qxs[0]),
              "Only per-tensor quantization is supported in 'cat'!")
  TORCH_CHECK(is_valid_quantization_scheme(out),
              "Only per-tensor quantization is supported in 'cat'!")
  auto out_ =
      quantized_cat_impl<ReLUFused>(qxs, dim, out.q_scale(), out.q_zero_point());
  at::native::copy_(out, out_, /*non_blocking=*/false);
  return out;
}

} // namespace

TORCH_LIBRARY_IMPL(quantized, QuantizedCPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("quantized::cat"), TORCH_FN(qcat<false>));
  m.impl(TORCH_SELECTIVE_NAME("quantized::cat_relu"), TORCH_FN(qcat<true>));
  m.impl(TORCH_SELECTIVE_NAME("quantized::cat_out"), TORCH_FN(qcat_out<false>));
  m.impl(TORCH_SELECTIVE_NAME("quantized::cat_relu_out"), TORCH_FN(qcat_out<true>));
}

Tensor cat_quantized_cpu(TensorList qxs, int64_t dim) {
  TORCH_CHECK(is_valid_quantization_scheme(qxs[0]),
              "Only per-tensor quantization is supported in 'cat'!")
  double _scale = qxs[0].q_scale();
  int64_t _zero_point = qxs[0].q_zero_point();
  return quantized_cat_impl<false>(c10::List<Tensor>(qxs), dim, _scale, _zero_point);
}

Tensor& cat_out_quantized_cpu(Tensor& out, TensorList qxs, int64_t dim) {
  TORCH_CHECK(is_valid_quantization_scheme(qxs[0]),
              "Only per-tensor quantization is supported in 'cat'!")
  TORCH_CHECK(is_valid_quantization_scheme(out),
              "Only per-tensor quantization is supported in 'cat'!")
  auto out_ = quantized_cat_impl<false>(c10::List<Tensor>(qxs), dim, out.q_scale(),
                                        out.q_zero_point());
  at::native::copy_(out, out_, /*non_blocking=*/false);
  return out;
}

}  // namespace native
}  // namespace at
