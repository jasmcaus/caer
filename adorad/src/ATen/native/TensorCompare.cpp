#include <ATen/ATen.h>
#include <ATen/CPUApplyUtils.h>
#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <c10/util/Exception.h>
#include <ATen/native/TensorCompare.h>
#include <ATen/NamedTensorUtils.h>

namespace at { namespace native {

DEFINE_DISPATCH(where_kernel);
DEFINE_DISPATCH(max_stub);
DEFINE_DISPATCH(min_stub);
DEFINE_DISPATCH(_aminmax_stub);
DEFINE_DISPATCH(isposinf_stub);
DEFINE_DISPATCH(isneginf_stub);

bool allclose(const Tensor& self, const Tensor& other, double rtol, double atol, bool equal_nan) {
  return at::isclose(self, other, rtol, atol, equal_nan).all().item<uint8_t>();
}

// Note [closeness]
// A number A is close to B when either:
//
// (1) A is equal to B, with NaNs comparing equal when equal_nan is true.
// (2) The error abs(A - B) is finite and less than the max error
//      (atol + abs(rtol * B)).
//
// Note that this is consistent with NumPy's isclose but divergent from
// Python's isclose, which computes the max error symmetrically as
// max(rtol * max(abs(A), abs(B)), atol).
// TODO: use bitwise operator overloads once we add them
// TODO: revisit complex inputs and equal_nan=true after
//  https://github.com/numpy/numpy/issues/15959 is resolved
Tensor isclose(const Tensor& self, const Tensor& other, double rtol, double atol, bool equal_nan) {
  TORCH_CHECK(self.scalar_type() == other.scalar_type(), self.scalar_type(), " did not match ", other.scalar_type());
  TORCH_CHECK(!(self.is_complex() && equal_nan),
    "isclose with equal_nan=True is not supported for complex inputs.");

  // Checks that rtol and atol are non-negative
  // Note: consistent with Python's isclose but divergent from NumPy's, which
  //  allows negative atol and rtol.
  TORCH_CHECK(rtol >= 0, "rtol must be greater than or equal to zero, but got ", rtol);
  TORCH_CHECK(atol >= 0, "atol must be greater than or equal to zero, but got ", atol);

  // Computes equality closeness
  Tensor close = self == other;
  if (equal_nan && self.is_floating_point()) {
      close.__ior__((self != self).__iand__(other != other));
  }

  // Note [closeness error computation]
  // atol and rtol are provided as doubles, so the computation
  // rtol * other will produce a float or complex tensor.
  // When the difference (self - other) is compared to it then the
  // tensor representing the difference will also be cast to float or complex.
  // However, since (self - other) in uint8 is very likely to produce a
  // negative value, this moves the cast forward so the difference is
  // always computed in a float or complex type.
  // If the values of the integer tensors cannot be exactly represented
  // by the default scalar type then this may cause an incorrect result.

  // Computes allowed and actual error
  Tensor cast_other;
  if (c10::isIntegralType(self.scalar_type(), /*include_bool=*/true)) {
    cast_other = other.to(at::get_default_dtype());
  } else {
    cast_other = other;
  }
  Tensor allowed_error = atol + (rtol * cast_other).abs();
  Tensor actual_error = (self - cast_other).abs();

  // Computes finite closeness
  close.__ior__(at::isfinite(actual_error).__iand__(actual_error <= allowed_error));

  return close;
}

Tensor isnan(const Tensor& self) {
  return self != self;
}

Tensor isreal(const Tensor& self) {
  // Note: Integral and Floating tensor values are always real
  if (c10::isIntegralType(self.scalar_type(), /*include_bool=*/true) ||
      c10::isFloatingType(self.scalar_type())) {
    return at::ones_like(self, at::kBool, at::MemoryFormat::Preserve);
  }

  return at::imag(self) == 0;
}

Tensor isinf(const Tensor &self) {
  // Note: Integral tensor values are never infinite
  if (c10::isIntegralType(self.scalar_type(), /*include_bool=*/true)) {
    return at::zeros_like(self, at::kBool, at::MemoryFormat::Preserve);
  }

  // Note: a complex value is infinite when either part is infinite
  if (self.is_complex()) {
    return at::isinf(at::real(self)).__ior__
          (at::isinf(at::imag(self)));
  }

  return AT_DISPATCH_FLOATING_TYPES_AND_HALF(self.scalar_type(), "isinf", [&]() {
    return self.abs() == std::numeric_limits<scalar_t>::infinity();
  });
}

Tensor isposinf(const Tensor &self) {
  Tensor result = at::empty_like(self, at::kBool, at::MemoryFormat::Preserve);
  at::isposinf_out(result, self);
  return result;
}

Tensor& isposinf_out(Tensor& result, const Tensor& self) {
  TORCH_CHECK(!self.is_complex(), "isposinf does not support complex inputs.");
  TORCH_CHECK(result.scalar_type() == at::kBool, "isposinf does not support non-boolean outputs.");
  result.resize_(self.sizes());

  if (c10::isIntegralType(self.scalar_type(), /*include_bool=*/true)) {
    result.fill_(false);
  } else {
    auto iter = TensorIteratorConfig()
      .check_all_same_dtype(false)
      .add_output(result)
      .add_input(self)
      .build();
    isposinf_stub(iter.device_type(), iter);
  }
  return result;
}

Tensor isneginf(const Tensor &self) {
  Tensor result = at::empty_like(self, at::kBool, at::MemoryFormat::Preserve);
  at::isneginf_out(result, self);
  return result;
}

Tensor& isneginf_out(Tensor& result, const Tensor& self) {
  TORCH_CHECK(!self.is_complex(), "isneginf does not support complex inputs.");
  TORCH_CHECK(result.scalar_type() == at::kBool, "isneginf does not support non-boolean outputs.");
  result.resize_(self.sizes());

  if (c10::isIntegralType(self.scalar_type(), /*include_bool=*/true)) {
    result.fill_(false);
  } else {
    auto iter = TensorIteratorConfig()
      .check_all_same_dtype(false)
      .add_output(result)
      .add_input(self)
      .build();
    isneginf_stub(iter.device_type(), iter);
  }
  return result;
}

Tensor isfinite(const Tensor& self) {
  // Note: Integral tensor values are always finite
  if (c10::isIntegralType(self.scalar_type(), /*include_bool=*/true)) {
    return at::ones_like(self, at::kBool, at::MemoryFormat::Preserve);
  }

  // Note: a complex value is finite iff both parts are finite
  if (self.is_complex()) {
    return at::isfinite(self.abs());
  }

  return AT_DISPATCH_FLOATING_TYPES_AND_HALF(self.scalar_type(), "isfinite", [&]() {
    return (self == self) * (self.abs() != std::numeric_limits<scalar_t>::infinity());
  });
}

bool is_nonzero(const Tensor& self) {
  auto n = self.numel();
  TORCH_CHECK(n != 0, "Boolean value of Tensor with no values is ambiguous");
  TORCH_CHECK(n < 2, "Boolean value of Tensor with more than one value is ambiguous");

  Scalar localScalar = self.item();
  if (localScalar.isFloatingPoint()) {
    return localScalar.to<double>() != 0;
  } else if (localScalar.isComplex()) {
     return localScalar.to<c10::complex<double>>() != c10::complex<double>(0.0, 0.0);
  } else if (localScalar.isIntegral(false)){
    return localScalar.to<int64_t>() != 0;
  } else if (localScalar.isBoolean()) {
    return localScalar.to<bool>();
  }
  TORCH_INTERNAL_ASSERT(false, "Expected non-Tensor backend scalar");
}

namespace {

static Tensor wrapped_scalar_tensor(
    Scalar scalar,
    Device device,
    bool use_default_dtype = false) {
  at::Tensor tensor;
  if (use_default_dtype) {
    tensor = scalar_to_tensor_default_dtype(scalar, device);
  } else {
    tensor = scalar_to_tensor(scalar, device);
  }
  tensor.unsafeGetTensorImpl()->set_wrapped_number(true);
  return tensor;
}

} // anonymous namespace

Tensor where(const Tensor& condition, const Tensor& self, const Tensor& other) {
  TORCH_CHECK(condition.device() == self.device() && self.device() == other.device(),
              "Expected condition, x and y to be on the same device, but condition is on ",
              condition.device(), " and x and y are on ", self.device(), " and ", other.device(),
              " respectively");
  TORCH_CHECK(condition.scalar_type() == ScalarType::Byte || condition.scalar_type() == ScalarType::Bool,
              "Expected condition to have ScalarType Byte, but got ScalarType ",
              toString(condition.scalar_type()));
  Tensor b_condition, b_self, b_other;
  std::tie(b_condition, b_self, b_other) = expand_outplace(condition, self, other, "where");
  return at::_s_where(b_condition, b_self, b_other);
}

Tensor where(const Tensor& condition, Scalar self, const Tensor& other) {
  return at::where(condition, wrapped_scalar_tensor(self, other.device()), other);
}

Tensor where(const Tensor& condition, const Tensor& self, Scalar other) {
  return at::where(condition, self, wrapped_scalar_tensor(other, self.device()));
}

Tensor where(const Tensor& condition, Scalar self, Scalar other) {
  const auto device = condition.device();
  const Tensor& other_t = wrapped_scalar_tensor(other, device, /*use_default_dtype=*/true);
  const Tensor& self_t = wrapped_scalar_tensor(self, device, /*use_default_dtype=*/true);
  return at::where(condition, self_t, other_t);
}

std::vector<Tensor> where(const Tensor& condition) {
  return condition.nonzero_numpy();
}

Tensor _s_where(const Tensor& condition, const Tensor& self, const Tensor& other) {
  TORCH_CHECK(self.dtype() == other.dtype(), "expected scalar type ", self.dtype(), " but found ", other.dtype());
  Tensor ret = at::empty(self.sizes(), self.options());
  auto iter = at::TensorIteratorConfig()
    .check_all_same_dtype(false)
    .add_output(ret)
    .add_input(condition)
    .add_input(self)
    .add_input(other)
    .build();
  where_kernel(iter.device_type(), iter, condition.scalar_type());
  return ret;
}

std::tuple<Tensor, Tensor> mode(const Tensor& self, int64_t dim, bool keepdim) {
  Tensor values = at::empty({0}, self.options());
  Tensor indices = at::empty({0}, self.options().dtype(kLong));
  return at::native::mode_out(values, indices, self, dim, keepdim);
}

std::tuple<Tensor &,Tensor &> mode_out(Tensor& values, Tensor& indices,
                                       const Tensor& self, int64_t dim, bool keepdim) {
  TORCH_CHECK(self.device().type() == DeviceType::CPU || self.device().type() == DeviceType::CUDA,
              "mode only supports CPU AND CUDA device type, got: ", self.device().type());
  TORCH_CHECK(self.layout() == Layout::Strided,
              "mode only supports strided layout, got: ", self.layout());
  dim = maybe_wrap_dim(dim, self.dim());
  if (_dimreduce_return_trivial_no_ident(values, self, dim, keepdim, "mode")) {
    AT_ASSERT(values.dim() == 0);
    indices.resize_({}).fill_(0);
    return std::forward_as_tuple(values, indices);
  } else {
    auto result = [&]() {
      NoNamesGuard guard;
      return at::_mode_out(values, indices, self, dim, keepdim);
    }();
    namedinference::propagate_names_for_reduction(std::get<0>(result), self, dim, keepdim);
    namedinference::propagate_names_for_reduction(std::get<1>(result), self, dim, keepdim);
    return result;
  }
}

std::tuple<Tensor, Tensor> max(const Tensor& self, int64_t dim, bool keepdim) {
  TORCH_CHECK(!self.is_complex(), "max is not yet implemented for complex tensors.");
  Tensor max_indices = at::empty({0}, self.options().dtype(kLong));
  if (self.is_quantized()) {
    Tensor max = at::empty({0}, self.options().dtype(toUnderlying(self.scalar_type())));
    at::native::max_out(max, max_indices, self.int_repr(), dim, keepdim);
    // TODO: qscheme
    return std::tuple<Tensor, Tensor>(at::_make_per_tensor_quantized_tensor(max, self.q_scale(), self.q_zero_point()), max_indices);
  } else {
    Tensor max = at::empty({0}, self.options());
    return at::native::max_out(max, max_indices, self, dim, keepdim);
  }
}

static std::tuple<Tensor &,Tensor &> max_out_impl(Tensor& max, Tensor& max_indices,
                                                  const Tensor& self, int64_t dim, bool keepdim) {
  TORCH_CHECK(!self.is_complex(), "max is not yet implemented for complex tensors.");
  TORCH_CHECK(self.device().type() == DeviceType::CPU || self.device().type() == DeviceType::CUDA,
              "max only supports CPU AND CUDA device type, got: ", self.device().type());
  TORCH_CHECK(self.layout() == Layout::Strided,
              "max only supports strided layout, got: ", self.layout());
  TORCH_CHECK(self.device() == max.device(),
              "expected device ", self.device(), " but got ",
              max.device(), " for max values output");
  TORCH_CHECK(self.device() == max_indices.device(),
              "expected device ", self.device(), " but got ",
              max_indices.device(), " for indices output");
  dim = maybe_wrap_dim(dim, self.dim());
  if (_dimreduce_return_trivial_no_ident(max, self, dim, keepdim, "max")) {
    AT_ASSERT(max.dim() == 0);
    max_indices.resize_({}).fill_(0);
    return std::forward_as_tuple(max, max_indices);
  } else {
    max_stub(self.device().type(), max, max_indices, self, dim, keepdim);
    return std::tuple<Tensor &,Tensor &>{max, max_indices};
  }
}

std::tuple<Tensor&,Tensor&> max_out(Tensor& max, Tensor& max_indices,
                                      const Tensor& self, int64_t dim, bool keepdim) {
  TORCH_CHECK(!self.is_complex(), "max is not yet implemented for complex tensors.");
  auto result = [&]() {
    NoNamesGuard guard;
    return max_out_impl(max, max_indices, self, dim, keepdim);
  }();
  namedinference::propagate_names_for_reduction(max, self, dim, keepdim);
  namedinference::propagate_names_for_reduction(max_indices, self, dim, keepdim);
  return result;
}

std::tuple<Tensor, Tensor> min(const Tensor& self, int64_t dim, bool keepdim) {
  TORCH_CHECK(!self.is_complex(), "min is not yet implemented for complex tensors.");
  Tensor min_indices = at::empty({0}, self.options().dtype(kLong));
  if (self.is_quantized()) {
    Tensor min = at::empty({0}, self.options().dtype(toUnderlying(self.scalar_type())));
    at::native::min_out(min, min_indices, self.int_repr(), dim, keepdim);
    return std::tuple<Tensor, Tensor>(at::_make_per_tensor_quantized_tensor(min, self.q_scale(), self.q_zero_point()), min_indices);
  } else {
    Tensor min = at::empty({0}, self.options());
    return at::native::min_out(min, min_indices, self, dim, keepdim);
  }
}

static std::tuple<Tensor &, Tensor &> _aminmax_out_impl(Tensor& min, Tensor& max,
                                                  const Tensor& self, int64_t dim, bool keepdim) {
  TORCH_CHECK(!self.is_complex(), "max is not yet implemented for complex tensors.");
  TORCH_CHECK(self.device().type() == DeviceType::CPU || self.device().type() == DeviceType::CUDA,
              "min_max_val only supports CPU AND CUDA device type, got: ", self.device().type());
  TORCH_CHECK(self.layout() == Layout::Strided,
              "min_max only supports strided layout, got: ", self.layout());
  TORCH_CHECK(self.device() == min.device(),
              "expected device ", self.device(), " but got ",
              min.device(), " for min values output");
  TORCH_CHECK(self.device() == max.device(),
              "expected device ", self.device(), " but got ",
              max.device(), " for max values output");
  dim = maybe_wrap_dim(dim, self.dim());
  if (_dimreduce_return_trivial_no_ident(min, self, dim, keepdim, "min") &&
      _dimreduce_return_trivial_no_ident(max, self, dim, keepdim, "max")) {
    return std::forward_as_tuple(min, max);
  } else {
    _aminmax_stub(self.device().type(), min, max, self, dim, keepdim);
    return std::tuple<Tensor &, Tensor &>{min, max};
  }
}

std::tuple<Tensor, Tensor> _aminmax(const Tensor& self, int64_t dim, bool keepdim) {
  TORCH_CHECK(!self.is_complex(), "min_max is not yet implemented for complex tensors.");
  TORCH_CHECK(!self.is_quantized(), "min is not yet implemented for quantized tensors.");

  Tensor min = at::empty({0}, self.options());
  Tensor max = at::empty({0}, self.options());

  auto result = _aminmax_out_impl(min, max, self, dim, keepdim);
  return result;
}

static std::tuple<Tensor &,Tensor &> min_out_impl(Tensor& min, Tensor& min_indices,
                                                  const Tensor& self, int64_t dim, bool keepdim) {
  TORCH_CHECK(!self.is_complex(), "min is not yet implemented for complex tensors.");
  TORCH_CHECK(self.device().type() == DeviceType::CPU || self.device().type() == DeviceType::CUDA,
              "min only supports CPU AND CUDA device type, got: ", self.device().type());
  TORCH_CHECK(self.layout() == Layout::Strided,
              "min only supports strided layout, got: ", self.layout());
  TORCH_CHECK(self.device() == min.device(),
              "expected device ", self.device(), " but got ",
              min.device(), " for min values output");
  TORCH_CHECK(self.device() == min_indices.device(),
              "expected device ", self.device(), " but got ",
              min_indices.device(), " for indices output");
  dim = maybe_wrap_dim(dim, self.dim());
  if (_dimreduce_return_trivial_no_ident(min, self, dim, keepdim, "min")) {
    AT_ASSERT(min.dim() == 0);
    min_indices.resize_({}).fill_(0);
    return std::forward_as_tuple(min, min_indices);
  } else {
    min_stub(self.device().type(), min, min_indices, self, dim, keepdim);
    return std::tuple<Tensor &,Tensor &>{min, min_indices};
  }
}

std::tuple<Tensor&,Tensor&> min_out(Tensor& min, Tensor& min_indices,
                                    const Tensor& self, int64_t dim, bool keepdim) {
  TORCH_CHECK(!self.is_complex(), "min is not yet implemented for complex tensors.");
  auto result = [&]() {
    NoNamesGuard guard;
    return min_out_impl(min, min_indices, self, dim, keepdim);
  }();
  namedinference::propagate_names_for_reduction(min, self, dim, keepdim);
  namedinference::propagate_names_for_reduction(min_indices, self, dim, keepdim);
  return result;
}


// Named tensor overloads

std::tuple<Tensor, Tensor> min(const Tensor& self, Dimname dim, bool keepdim) {
  TORCH_CHECK(!self.is_complex(), "min is not yet implemented for complex tensors.");
  return at::min(self, dimname_to_position(self, dim), keepdim);
}
std::tuple<Tensor &,Tensor &> min_out(Tensor& min, Tensor& min_indices,
                                      const Tensor& self, Dimname dim, bool keepdim) {
  TORCH_CHECK(!self.is_complex(), "min is not yet implemented for complex tensors.");
  return at::min_out(min, min_indices, self, dimname_to_position(self, dim), keepdim);
}
std::tuple<Tensor, Tensor> max(const Tensor& self, Dimname dim, bool keepdim) {
  TORCH_CHECK(!self.is_complex(), "max is not yet implemented for complex tensors.");
  return at::max(self, dimname_to_position(self, dim), keepdim);
}
std::tuple<Tensor &,Tensor &> max_out(Tensor& max, Tensor& max_indices,
                                      const Tensor& self, Dimname dim, bool keepdim) {
  TORCH_CHECK(!self.is_complex(), "max is not yet implemented for complex tensors.");
  return at::max_out(max, max_indices, self, dimname_to_position(self, dim), keepdim);
}
Tensor argmax(const Tensor& self, Dimname dim, bool keepdim) {
  reportNYIDimnameOverload("argmax");
}
Tensor argmin(const Tensor& self, Dimname dim, bool keepdim) {
  reportNYIDimnameOverload("argmin");
}
Tensor argsort(const Tensor& self, Dimname dim, bool keepdim) {
  reportNYIDimnameOverload("argsort");
}
std::tuple<Tensor, Tensor> mode(const Tensor& self, Dimname dim, bool keepdim) {
  return at::mode(self, dimname_to_position(self, dim), keepdim);
}
std::tuple<Tensor &,Tensor &> mode_out(Tensor& values, Tensor& indices,
                                       const Tensor& self, Dimname dim, bool keepdim) {
  return at::mode_out(values, indices, self, dimname_to_position(self, dim), keepdim);
}

}} // namespace at::native
