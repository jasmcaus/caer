#include <ATen/native/ReduceOps.h>

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/ExpandUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/WrapDimUtilsMulti.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/native/TensorDimApply.h>
#include <ATen/native/SharedReduceOps.h>

#include <algorithm>
#include <functional>
#include <limits>
#include <numeric>
#include <vector>
#include <map>
#include <cmath>
#include <cfloat>
#include <type_traits>

namespace at {
namespace native {

DEFINE_DISPATCH(sum_stub);
DEFINE_DISPATCH(nansum_stub);
DEFINE_DISPATCH(std_var_stub);
DEFINE_DISPATCH(prod_stub);
DEFINE_DISPATCH(norm_stub);
DEFINE_DISPATCH(mean_stub);
DEFINE_DISPATCH(and_stub);
DEFINE_DISPATCH(or_stub);
DEFINE_DISPATCH(min_values_stub);
DEFINE_DISPATCH(max_values_stub);
DEFINE_DISPATCH(argmax_stub);
DEFINE_DISPATCH(argmin_stub);
DEFINE_DISPATCH(cumsum_stub);
DEFINE_DISPATCH(cumprod_stub);
DEFINE_DISPATCH(logcumsumexp_stub);

Tensor _logcumsumexp_cpu(const Tensor& self, int64_t dim) {
  Tensor result = at::empty_like(self, MemoryFormat::Contiguous);
  return _logcumsumexp_out_cpu(result, self, dim);
}

Tensor& _logcumsumexp_out_cpu(Tensor& result, const Tensor& self, int64_t dim) {
  logcumsumexp_stub(self.device().type(), result, self, dim);
  return result;
}

Tensor logcumsumexp(const Tensor& self, int64_t dim) {
  auto result = [&]() {
    NoNamesGuard guard;
    return at::_logcumsumexp(self, dim);
  }();
  namedinference::propagate_names(result, self);
  return result;
}

Tensor& logcumsumexp_out(Tensor& result, const Tensor& self, int64_t dim) {
  check_scalar_type_device_layout_equal(result, self);
  {
    NoNamesGuard guard;
    at::_logcumsumexp_out(result, self.toType(result.scalar_type()), dim);
  }
  namedinference::propagate_names(result, self);
  return result;
}

Tensor _cumsum_cpu(const Tensor& self, int64_t dim) {
  Tensor result = at::empty_like(self, MemoryFormat::Contiguous);
  cumsum_stub(self.device().type(), result, self, dim);
  return result;
}

Tensor& _cumsum_out_cpu(Tensor& result, const Tensor& self, int64_t dim) {
  cumsum_stub(self.device().type(), result, self, dim);
  return result;
}

Tensor cumsum(const Tensor& self, int64_t dim, c10::optional<ScalarType> dtype) {
  auto result = [&]() {
    NoNamesGuard guard;
    return at::_cumsum(integer_upcast(self, dtype), dim);
  }();
  namedinference::propagate_names(result, self);
  return result;
}

Tensor& cumsum_(Tensor& self, int64_t dim, c10::optional<ScalarType> dtype) {
  TORCH_CHECK(
          !dtype.has_value() || (self.scalar_type() == dtype.value()),
          "provided dtype must match the dtype of self tensor in cumsum. Got ",
          toString(self.scalar_type()),
          " and ",
          toString(dtype.value()),
          ".");

  return at::_cumsum_out(self, self, dim);
}

Tensor& cumsum_out(Tensor& result, const Tensor& self, int64_t dim, c10::optional<ScalarType> dtype) {
  // result type is favored over dtype; check that they match if provided (NumPy doesn't check)
  TORCH_CHECK(
      !dtype.has_value() || (result.scalar_type() == dtype.value()),
      "provided dtype must match dtype of result in cumsum. Got ",
      toString(result.scalar_type()),
      " and ",
      toString(dtype.value()),
      ".");
  {
    NoNamesGuard guard;
    at::_cumsum_out(result, self.toType(result.scalar_type()), dim);
  }
  namedinference::propagate_names(result, self);
  return result;
}

Tensor _cumprod_cpu(const Tensor& self, int64_t dim) {
  Tensor result = at::empty_like(self, MemoryFormat::Contiguous);
  cumprod_stub(self.device().type(), result, self, dim);
  return result;
}

Tensor& _cumprod_out_cpu(Tensor& result, const Tensor& self, int64_t dim) {
  cumprod_stub(self.device().type(), result, self, dim);
  return result;
}

Tensor cumprod(const Tensor& self, int64_t dim, c10::optional<ScalarType> dtype) {
  auto result = [&]() {
    NoNamesGuard guard;
    return at::_cumprod(integer_upcast(self, dtype), dim);
  }();
  namedinference::propagate_names(result, self);
  return result;
}

Tensor& cumprod_(Tensor& self, int64_t dim, c10::optional<ScalarType> dtype) {
    TORCH_CHECK(
            !dtype.has_value() || (self.scalar_type() == dtype.value()),
            "provided dtype must match the dtype of self tensor in cumprod. Got ",
            toString(self.scalar_type()),
            " and ",
            toString(dtype.value()),
            ".");

    return at::_cumprod_out(self, self, dim);
}

Tensor& cumprod_out(Tensor& result, const Tensor& self, int64_t dim, c10::optional<ScalarType> dtype) {
  // result type is favored over dtype; check that they match if provided (NumPy doesn't check)
  TORCH_CHECK(
      !dtype.has_value() || (result.scalar_type() == dtype.value()),
      "provided dtype must match dtype of result in cumprod. Got ",
      toString(result.scalar_type()),
      " and ",
      toString(dtype.value()),
      ".");
  {
    NoNamesGuard guard;
    at::_cumprod_out(result, self.toType(result.scalar_type()), dim);
  }
  namedinference::propagate_names(result, self);
  return result;
}

static Tensor sum_scan_exclusive(const Tensor& x, int64_t dim) {
  Tensor ret = at::cumsum(-x, dim);

  int64_t end_idx = ret.size(dim) - 1;
  Tensor ret_sum = ret.narrow(dim, end_idx, 1).clone(at::MemoryFormat::Preserve);
  ret -= ret_sum.expand_as(ret);
  ret += x;
  return ret;
}

Tensor cumprod_backward(const Tensor& grad, const Tensor& input, int64_t dim) {
  /*
    There are two algorithms to do this. The first one
    is very efficient, but works only when there are no
    nonzero elements in the input.

    The second one is much more complex, but it doesn't
    assume anything on the input. The main downside is
    that it takes time O(n^2), where n = input.size(self.dim)
    (i.e. the length of the cumulative product). This is in
    contrast to the forward pass and the efficient algorithm,
    which are both O(n).

    The second algorithm is a simple application of the chain
    rule. If x is an n-dimensional vector, and y = cumprod(x),
    and F is the final cost, then

    dF / dx_k = sum_j (dF / dy_j) * (dy_j / dx_k)   (1)

    The term dF / dy_j is just grad_output[j] (assuming again
    everything is one-dimensional).

    The term (dy_j / dx_k) is easilly seen to be

    if j >= k
      dy_j / dx_k = prod_{1 <= i <= j, i != k} x_i
    else:
      dy_j / dx_k = 0

    Note that the indicator (j>=k) can be taken out
    by replacing the sum in (1) with a sum from
    j = k to n.

    Thus,
    df / dx_k = sum_{k <= j <= n} grad_output[j] * (dy_j / dx_k)

    with
    dy_j / dx_k = prod_{1 <= i <= j, i != k} x_i     (2)

    Note that this last term is just the cumulative product
    with k omitted. Thus, if x_k (the input) is nonzero, we can
    just express this as

    dy_j / dx_k = (prod_{1 <= i <= j} x_i) / x_k
                = y_j / x_k

    So therefore,

    df / dx_k = sum_{k <= j <= n} grad_output[j] * y_j / x_k

    so

    grad_output = sum_scan_exclusiv(grad_output * output) / input

    If the input is nonzero, we need to calculate the dy_j / dx_k
    by using the formula (2), called in the code omitted_products.

    The way the code calculates it is simply by noting that

    prod_{1 <= i <= j, i != k} x_i
        = (prod_{1 <= i <= k} x_i) * (prod_{k + 1 <= i <= j} x_i)

    the first term is calculated as prods_until_k, which since
    doesn't depend in j is easy to vectorize.

    The second term (indexed by j) is the cumulative product of
    x_{k+1}, x_{k+2}, ..., x_n, and it's named in the code
    prods_from_k_pkus_1, and it's calculated as a cumprod.

    In order to vectorize this properly, we need to add to
    omitted_products the dimensions where k > j, and therefore
    dy_j / dx_k = 0, which is done right after the assert.
  */

  if (input.dim() == 0 || input.numel() == 0) {
    return grad;
  }
  dim = at::maybe_wrap_dim(dim, input.sizes().size());
  int64_t dim_size = input.size(dim);
  if (dim_size == 1) {
    return grad;
  }

  // Simple case with nonzero elements in the input
  if ((input != 0).all().item<uint8_t>()) {
    Tensor result = at::cumprod(input, dim);
    return sum_scan_exclusive(result * grad, dim) / input;
  }

  auto ones_size = input.sizes().vec();
  ones_size[dim] = 1;
  Tensor ones = at::ones({1}, grad.options()).expand(ones_size);
  Tensor grad_input = at::zeros(input.sizes(), grad.options());
  Tensor prods_from_k_plus_1;
  Tensor omitted_products;
  for (int k = 0; k < dim_size; ++k) {
    if (k == 0) {
      prods_from_k_plus_1 = at::cumprod(input.slice(dim, k + 1), dim);
      omitted_products = at::cat({ones, prods_from_k_plus_1}, dim);
    } else if (k == dim_size - 1) {
      Tensor prods_until_k = at::prod(input.slice(dim, 0, k), dim, true);
      omitted_products = prods_until_k;
    } else {
      Tensor prods_until_k = at::prod(input.slice(dim, 0, k), dim, true);
      prods_from_k_plus_1 = at::cumprod(input.slice(dim, k+1), dim);
      omitted_products = prods_until_k.expand_as(prods_from_k_plus_1) * prods_from_k_plus_1;
      omitted_products = at::cat({prods_until_k, omitted_products}, dim);
    }

    // At this point omitted_products is the same size
    // as input, except on the dimension dim where it's
    // dim_size - k
    AT_ASSERT(omitted_products.size(dim) == dim_size - k);

    grad_input.select(dim, k).copy_(
        at::sum(grad.slice(dim, k) * omitted_products,dim));
  }

  return grad_input;
}

// Implement std::is_nan<IntegralType> for MSVC.
namespace {
#ifdef _MSC_VER
template<typename T>
inline typename std::enable_if<std::is_integral<T>::value, bool>::type isnan_(T x) {
  return false;
}
template<typename T>
inline typename std::enable_if<!std::is_integral<T>::value, bool>::type isnan_(T x) {
  return std::isnan(x);
}
#else
template<typename T>
inline bool isnan_(T x) {
  return std::isnan(x);
}
#endif
}

template<typename T1, typename T2, typename Operation>
void cummax_cummin_helper(const T1* self_data, T1* values_data, T2* indices_data,
          int self_dim_size, int self_stride, int values_stride, int indices_stride) {
      Operation op;
      T1 out = self_data[0];
      int idx = 0;
      for(int i = 0; i < self_dim_size; i++) {
        T1 curr_elem = self_data[i*self_stride];
        if(isnan_(curr_elem) || (!isnan_(out) && op(curr_elem, out))) {
            out = self_data[i*self_stride];
            idx = i;
        }
        values_data[i*values_stride] = out;
        indices_data[i*indices_stride] = idx;
      }
}

void cummax_helper_cpu(const Tensor& self, Tensor& values, Tensor& indices, int64_t dim) {
  AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Bool,
    self.scalar_type(), "cummax_cpu",
    [&] {
      at::native::tensor_dim_apply3<scalar_t, int64_t>(self, values, indices, dim, cummax_cummin_helper<scalar_t, int64_t, std::greater_equal<scalar_t>>);
    });
}

std::tuple<Tensor&, Tensor&> cummax_out(Tensor& values, Tensor& indices, const Tensor& self, int64_t dim) {
  check_scalar_type_device_layout_equal(values, self);
  check_scalar_type_device_layout_equal(indices, at::empty({0}, self.options().dtype(at::kLong)));
  {
    NoNamesGuard guard;
    values.resize_(self.sizes());
    indices.resize_(self.sizes());
    if(self.dim() == 0) {
      values.fill_(self);
      indices.fill_(0);
    } else if(self.numel() != 0) {
      dim = maybe_wrap_dim(dim, self.dim());
      at::_cummax_helper(self, values, indices, dim);
    }
  }
  namedinference::propagate_names(values, self);
  namedinference::propagate_names(indices, self);
  return std::forward_as_tuple(values, indices);
}

std::tuple<Tensor, Tensor> cummax(const Tensor& self, int64_t dim) {
  auto values = at::empty(self.sizes(), self.options());
  auto indices = at::empty(self.sizes(), self.options().dtype(at::kLong));
  at::cummax_out(values, indices, self, dim);
  return std::make_tuple(values, indices);
}

void cummin_helper_cpu(const Tensor& self, Tensor& values, Tensor& indices, int64_t dim) {
  AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Bool,
    self.scalar_type(), "cummin_cpu",
    [&] {
      at::native::tensor_dim_apply3<scalar_t, int64_t>(self, values, indices, dim, cummax_cummin_helper<scalar_t, int64_t, std::less_equal<scalar_t>>);
    });
}

std::tuple<Tensor&, Tensor&> cummin_out(Tensor& values, Tensor& indices, const Tensor& self, int64_t dim) {
  check_scalar_type_device_layout_equal(values, self);
  check_scalar_type_device_layout_equal(indices, at::empty({0}, self.options().dtype(at::kLong)));
  {
    NoNamesGuard guard;
    values.resize_(self.sizes());
    indices.resize_(self.sizes());
    if(self.dim() == 0) {
      values.fill_(self);
      indices.fill_(0);
    } else if(self.numel() != 0) {
      dim = maybe_wrap_dim(dim, self.dim());
      at::_cummin_helper(self, values, indices, dim);
    }
  }
  namedinference::propagate_names(values, self);
  namedinference::propagate_names(indices, self);
  return std::forward_as_tuple(values, indices);
}

std::tuple<Tensor, Tensor> cummin(const Tensor& self, int64_t dim) {
  auto values = at::empty(self.sizes(), self.options());
  auto indices = at::empty(self.sizes(), self.options().dtype(at::kLong));
  at::cummin_out(values, indices, self, dim);
  return std::make_tuple(values, indices);
}

Tensor cummaxmin_backward(const Tensor& grad, const Tensor& input, const Tensor& indices, int64_t dim) {
  if (input.numel() == 0) {
    return input;
  }
  auto result = at::zeros(input.sizes(), input.options());
  return result.scatter_add_(dim, indices, grad);
}


// ALL REDUCE #################################################################

static ScalarType get_dtype(Tensor& result, const Tensor& self, optional<ScalarType> dtype,
                            bool promote_integers=false) {
  if (dtype.has_value()) {
    return dtype.value();
  } else if (result.defined()) {
    return result.scalar_type();
  }
  ScalarType src_type = self.scalar_type();
  if (promote_integers && at::isIntegralType(src_type, /*includeBool=*/true)) {
    return kLong;
  }
  return src_type;
}

Tensor& sum_out(Tensor& result, const Tensor& self, IntArrayRef dim,
                       bool keepdim, optional<ScalarType> opt_dtype) {
  ScalarType dtype = get_dtype(result, self, opt_dtype, true);
  auto iter = make_reduction("sum", result, self, dim, keepdim, dtype);
  if (iter.numel() == 0) {
    result.zero_();
  } else {
    sum_stub(iter.device_type(), iter);
  }
  return result;
}

Tensor sum(const Tensor &self, c10::optional<ScalarType> dtype) {
  return at::native::sum(self, std::vector<int64_t>{}, false, dtype);
}
Tensor sum(const Tensor& self, IntArrayRef dim, bool keepdim, c10::optional<ScalarType> dtype) {
  Tensor result;
  return at::native::sum_out(result, self, dim, keepdim, dtype);
}
Tensor sum(const Tensor& self, DimnameList dim, bool keepdim, c10::optional<ScalarType> dtype) {
  return at::sum(self, dimnames_to_positions(self, dim), keepdim, dtype);
}

Tensor& sum_out(Tensor& result, const Tensor& self, DimnameList dim,
                bool keepdim, optional<ScalarType> opt_dtype) {
  return at::sum_out(result, self, dimnames_to_positions(self, dim), keepdim, opt_dtype);
}

Tensor& nansum_out(Tensor& result, const Tensor& self, IntArrayRef dim,
                       bool keepdim, optional<ScalarType> opt_dtype) {
  TORCH_CHECK(!c10::isComplexType(self.scalar_type()), "nansum does not support complex inputs");
  // For integral types, use existing sum as
  // integral types don't have `Nan`.
  if (c10::isIntegralType(self.scalar_type(), true)){
    return at::sum_out(result, self, dim, keepdim, opt_dtype);
  }

  ScalarType dtype = get_dtype(result, self, opt_dtype, true);
  auto iter = make_reduction("nansum", result, self, dim, keepdim, dtype);
  if (iter.numel() == 0) {
    result = result.zero_();
  } else {
    nansum_stub(iter.device_type(), iter);
  }
  return result;
}

Tensor nansum(const Tensor &self, c10::optional<ScalarType> dtype) {
  return at::native::nansum(self, std::vector<int64_t>{}, false, dtype);
}

Tensor nansum(const Tensor& self, IntArrayRef dim, bool keepdim, c10::optional<ScalarType> dtype) {
  Tensor result;
  return at::native::nansum_out(result, self, dim, keepdim, dtype);
}

static Tensor& prod_out_impl(Tensor& result, const Tensor& self, IntArrayRef dim,
                        bool keepdim, c10::optional<ScalarType> opt_dtype) {
  ScalarType dtype = get_dtype(result, self, opt_dtype, true);
  auto iter = make_reduction("prod", result, self, dim, keepdim, dtype);
  if (iter.numel() == 0) {
    result.fill_(1);
  } else {
    prod_stub(iter.device_type(), iter);
  }
  return result;
}

// NOTE: this could be implemented via diag and sum, but this has perf problems,
// see https://github.com/pytorch/pytorch/pull/47305,
Tensor trace_cpu(const Tensor& self) {
  Tensor result;
  ScalarType dtype = get_dtype(result, self, c10::nullopt, true);
  result = at::empty({}, self.options().dtype(dtype));
  AT_DISPATCH_ALL_TYPES(self.scalar_type(), "trace", [&] {
    using accscalar_t = at::acc_type<scalar_t, false>;
    accscalar_t sum = 0;
    const auto* t_data = self.data_ptr<scalar_t>();

    int64_t t_stride_0, t_stride_1, t_diag_size;

    TORCH_CHECK(self.dim() == 2, "trace: expected a matrix, but got tensor with dim ", self.dim());

    t_stride_0 = self.stride(0);
    t_stride_1 = self.stride(1);

    t_diag_size = std::min(self.size(0), self.size(1));
    for (int64_t i = 0; i < t_diag_size; i++) {
      sum += t_data[i * (t_stride_0 + t_stride_1)];
    }

    // all integer types get promoted to kLong
    if (result.scalar_type() == at::kLong) {
      *result.data_ptr<int64_t>() = sum;
    } else {
      *result.data_ptr<scalar_t>() = sum;
    }
  });

  return result;
}

Tensor prod(const Tensor& self, int64_t dim, bool keepdim, c10::optional<ScalarType> dtype) {
  Tensor result;
  native::prod_out_impl(result, self, dim, keepdim, dtype);
  return result;
}

Tensor prod(const Tensor &self, c10::optional<ScalarType> dtype) {
  Tensor result;
  return at::native::prod_out_impl(result, self, {}, false, dtype);
}

Tensor& prod_out(Tensor& result, const Tensor& self, int64_t dim, bool keepdim, c10::optional<ScalarType> dtype) {
  return at::native::prod_out_impl(result, self, dim, keepdim, dtype);
}

Tensor prod(const Tensor& self, Dimname dim, bool keepdim, c10::optional<ScalarType> dtype) {
  return at::prod(self, dimname_to_position(self, dim), keepdim, dtype);
}

Tensor& prod_out(Tensor& result, const Tensor& self, Dimname dim,
                 bool keepdim, optional<ScalarType> opt_dtype) {
  return at::prod_out(result, self, dimname_to_position(self, dim), keepdim, opt_dtype);
}

Tensor &mean_out_cpu_gpu(Tensor &result, const Tensor &self, IntArrayRef dim,
                 bool keepdim, c10::optional<ScalarType> opt_dtype) {
  ScalarType scalarType = opt_dtype.has_value() ? opt_dtype.value() : self.scalar_type();
  TORCH_CHECK(
      at::isFloatingType(scalarType) || at::isComplexType(scalarType),
      "Can only calculate the mean of floating types. Got ",
      toString(scalarType),
      " instead.");
  ScalarType dtype = get_dtype(result, self, opt_dtype, true);
  // TODO: the TensorIterator reduction implementation of mean
  // (mean_kernel_impl()) is unvectorized and leads to very poor performance
  // for production workloads. Once that's fixed, the following code can be used
  // in lieu of the sum + divide implementation below.
  if (self.device().is_cpu()) {
    int64_t dim_prod = 1;
    if (dim.size() == 0 || self.ndimension() == 0) {
      dim_prod = self.numel();
    } else {
      for (auto d : dim) {
        dim_prod *= self.size(d);
      }
    }
    at::sum_out(result, self, dim, keepdim, dtype).div_(dim_prod);
    return result;
  }

  auto iter = make_reduction("mean", result, self, dim, keepdim, dtype);
  if (iter.numel() == 0) {
    result.fill_(std::numeric_limits<double>::quiet_NaN());
  } else {
    mean_stub(iter.device_type(), iter);
  }
  return result;
}

Tensor mean_cpu_gpu(const Tensor &self, optional<ScalarType> dtype) {
  return at::native::mean_cpu_gpu(self, IntArrayRef{}, false, dtype);
}

Tensor mean_cpu_gpu(const Tensor& self, IntArrayRef dim, bool keepdim, optional<ScalarType> dtype) {
  Tensor result;
  return at::native::mean_out_cpu_gpu(result, self, dim, keepdim, dtype);
}

Tensor mean(const Tensor& self, DimnameList dim, bool keepdim, optional<ScalarType> dtype) {
  return at::mean(self, dimnames_to_positions(self, dim), keepdim, dtype);
}

Tensor& mean_out(Tensor& result, const Tensor& self, DimnameList dim,
                 bool keepdim, c10::optional<ScalarType> opt_dtype) {
  return at::mean_out(result, self, dimnames_to_positions(self, dim), keepdim, opt_dtype);
}

static Tensor squeeze_multiple(const Tensor& self, IntArrayRef dims) {
  int ndims = self.sizes().size();
  auto dims_to_squeeze = at::dim_list_to_bitset(dims, ndims);
  Tensor result = self;
  for (int i = ndims - 1; i >= 0; --i) {
    if (dims_to_squeeze[i]) {
      result = result.squeeze(i);
    }
  }
  return result;
}

static Tensor& logsumexp_out_impl(Tensor& result, const Tensor& self, IntArrayRef dims, bool keepdim) {
  // can't take max of empty tensor
  if (self.numel() != 0) {
    auto maxes = at::amax(self, dims, true);
    auto maxes_squeezed = (keepdim ? maxes : squeeze_multiple(maxes, dims));
    maxes_squeezed.masked_fill_(maxes_squeezed.abs() == INFINITY, 0);
    at::sum_out(result, at::exp(self - maxes), dims, keepdim);
    result.log_().add_(maxes_squeezed);
  } else {
    at::sum_out(result, at::exp(self), dims, keepdim);
    result.log_();
  }
  return result;
}

Tensor& logsumexp_out(Tensor& result, const Tensor& self, IntArrayRef dims, bool keepdim) {
  {
    NoNamesGuard guard;
    logsumexp_out_impl(result, self, dims, keepdim);
  }
  namedinference::propagate_names_for_reduction(result, self, dims, keepdim);
  return result;
}

Tensor logsumexp(const Tensor& self, IntArrayRef dims, bool keepdim) {
  Tensor result = at::empty({0}, self.options());
  return at::native::logsumexp_out(result, self, dims, keepdim);
}

Tensor logsumexp(const Tensor& self, DimnameList dims, bool keepdim) {
  return at::logsumexp(self, dimnames_to_positions(self, dims), keepdim);
}

Tensor& logsumexp_out(Tensor& result, const Tensor& self, DimnameList dims, bool keepdim) {
  return at::logsumexp_out(result, self, dimnames_to_positions(self, dims), keepdim);
}

static Tensor& norm_out(Tensor &result, const Tensor &self, optional<Scalar> opt_p,
                               IntArrayRef dim, bool keepdim, optional<ScalarType> opt_dtype) {
  auto p = opt_p.value_or(2.0).to<double>();
  TORCH_CHECK(self.device().type() == DeviceType::CPU || self.device().type() == DeviceType::CUDA,
              "norm only supports CPU AND CUDA device type, got: ", self.device().type());
  TORCH_CHECK(self.layout() == Layout::Strided,
              "norm only supports strided layout, got: ", self.layout());

  ScalarType in_dtype = opt_dtype.has_value() ? opt_dtype.value() : self.scalar_type();
  TORCH_CHECK(
      at::isFloatingType(in_dtype) || at::isComplexType(in_dtype),
      "Can only calculate the norm of floating point and complex dtypes. Got ",
      toString(in_dtype),
      " instead.");

  ScalarType out_dtype = result.defined() ? result.scalar_type() : (opt_dtype.has_value() ? opt_dtype.value() : toValueType(self.scalar_type()));

  auto iter = make_reduction("norm", result, self, dim, keepdim, in_dtype, out_dtype);

  if (iter.numel() == 0) {
    result.zero_();
  } else {
    norm_stub(iter.device_type(), iter, p);
  }
  return result;
}

static inline Tensor _norm(const Tensor &self, Scalar p) {
  if (self.is_sparse()) {
    // Sparse tensors need a different implementation because their values
    // are accessed with a different API than strided tensors
    return at::native_norm(self, p);
  } else {
    TORCH_CHECK(self.device().type() == DeviceType::CPU || self.device().type() == DeviceType::CUDA,
                "norm only supports CPU AND CUDA device type, got: ", self.device().type());
    TORCH_CHECK(self.layout() == Layout::Strided,
                "norm only supports strided layout, got: ", self.layout());
    TORCH_CHECK(at::isFloatingType(self.scalar_type()) || at::isComplexType(self.scalar_type()),
                "norm only supports floating-point dtypes");

    Tensor result;
    return at::native::norm_out(result, self, p, IntArrayRef{}, false, c10::nullopt);
  }
}

Tensor &norm_out(Tensor& result, const Tensor& self, optional<Scalar> p, IntArrayRef dim, bool keepdim, ScalarType dtype) {
  return at::native::norm_out(result, self, p, dim, keepdim, optional<ScalarType>(dtype));
}

Tensor &norm_out(Tensor& result, const Tensor& self, optional<Scalar> p, IntArrayRef dim, bool keepdim) {
  return at::native::norm_out(result, self, p, dim, keepdim, c10::nullopt);
}

static Tensor norm(const Tensor& self, optional<Scalar> p, IntArrayRef dim, bool keepdim,
            optional<ScalarType> opt_dtype) {
  if (self.is_sparse()) {
    // Sparse tensors need a different implementation because their values
    // are accessed with a different API than strided tensors
    return at::native_norm(self, p, dim, keepdim, opt_dtype);
  } else {
    Tensor result;
    return at::native::norm_out(result, self, p, dim, keepdim, opt_dtype);
  }
}

Tensor norm(const Tensor& self, optional<Scalar> p, IntArrayRef dim, bool keepdim, ScalarType dtype) {
  return at::native::norm(self, p, dim, keepdim, optional<ScalarType>(dtype));
}

Tensor norm(const Tensor& self, optional<Scalar> p, ScalarType dtype) {
  return at::native::norm(self, p, IntArrayRef{}, false, optional<ScalarType>(dtype));
}

Tensor norm(const Tensor& self, optional<Scalar> p, IntArrayRef dim, bool keepdim) {
  return at::native::norm(self, p, dim, keepdim, c10::nullopt);
}

// leave it so we support sparse tensors
Tensor norm(const Tensor& self, Scalar p) {
  return at::native::_norm(self, p);
}

inline Tensor & _all(Tensor & result, TensorIterator & iter) {
  if (iter.numel() == 0) {
    result.fill_(1);
  } else {
    and_stub(iter.device_type(), iter);
  }

  return result;
}

Tensor all(const Tensor& self) {
  TORCH_CHECK(self.device().type() == DeviceType::CPU || self.device().type() == DeviceType::CUDA,
              "all only supports CPU AND CUDA device type, got: ", self.device().type());
  TORCH_CHECK(self.layout() == Layout::Strided,
              "all only supports strided layout, got: ", self.layout());

  Tensor result = at::empty({0}, self.options());
  auto iter = make_reduction(
    "all", result, self, {}, false, self.scalar_type());
  return _all(result, iter);
}

Tensor all(const Tensor& self, int64_t dim, bool keepdim) {
  Tensor result = at::empty({0}, self.options());
  return at::native::all_out(result, self, dim, keepdim);
}

Tensor &all_out(Tensor &result, const Tensor &self, int64_t dim, bool keepdim) {
  TORCH_CHECK(self.device().type() == DeviceType::CPU || self.device().type() == DeviceType::CUDA,
              "all only supports CPU AND CUDA device type, got: ", self.device().type());
  TORCH_CHECK(self.layout() == Layout::Strided,
              "all only supports strided layout, got: ", self.layout());

  dim = maybe_wrap_dim(dim, self.dim());
  if (_dimreduce_return_trivial(result, self, 1, dim, keepdim)) {
    return result;
  } else {
    auto iter = make_reduction(
      "all", result, self, dim, keepdim, self.scalar_type());
    return _all(result, iter);
  }
}

inline Tensor & _any(Tensor & result, TensorIterator & iter) {
  if (iter.numel() == 0) {
    result.fill_(0);
  } else {
    or_stub(iter.device_type(), iter);
  }

  return result;
}

Tensor any(const Tensor& self) {
  TORCH_CHECK(self.device().type() == DeviceType::CPU || self.device().type() == DeviceType::CUDA,
              "any only supports CPU AND CUDA device type, got: ", self.device().type());
  TORCH_CHECK(self.layout() == Layout::Strided || self.layout() == Layout::Sparse,
              "any only supports strided AND sparse layout, got: ", self.layout());

  Tensor result = at::empty({0}, self.options());
  auto iter = make_reduction(
    "any", result, self, {}, false, self.scalar_type());
  return _any(result, iter);
}

Tensor any(const Tensor& self, int64_t dim, bool keepdim) {
  Tensor result = at::empty({0}, self.options());
  return at::native::any_out(result, self, dim, keepdim);
}

Tensor &any_out(Tensor &result, const Tensor &self, int64_t dim, bool keepdim) {
  TORCH_CHECK(self.device().type() == DeviceType::CPU || self.device().type() == DeviceType::CUDA,
              "any only supports CPU AND CUDA device type, got: ", self.device().type());
  TORCH_CHECK(self.layout() == Layout::Strided,
              "any only supports strided layout, got: ", self.layout());

  dim = maybe_wrap_dim(dim, self.dim());
  if (_dimreduce_return_trivial(result, self, 0, dim, keepdim)) {
    return result;
  } else {
    auto iter = make_reduction(
      "any", result, self, dim, keepdim, self.scalar_type());
    return _any(result, iter);
  }
}

Tensor &amin_out(Tensor& result, const Tensor& self, IntArrayRef dim, bool keepdim) {
  TORCH_CHECK(self.scalar_type() == result.scalar_type(), "Illegal dtype for self, and out:", self.scalar_type(), result.scalar_type());
  auto iter = make_reduction("amin", result, self, dim, keepdim, self.scalar_type());
  TORCH_CHECK(iter.numel() > 0, "operation does not have an identity");
  min_values_stub(iter.device_type(), iter);
  return result;
}

Tensor amin(const Tensor& self, IntArrayRef dim, bool keepdim) {
  Tensor result = at::empty({0}, self.options());
  return at::amin_out(result, self, dim, keepdim);
}

Tensor &amax_out(Tensor& result, const Tensor& self, IntArrayRef dim, bool keepdim) {
  TORCH_CHECK(self.scalar_type() == result.scalar_type(), "Illegal dtype for self, and out:", self.scalar_type(), result.scalar_type());
  auto iter = make_reduction("amax", result, self, dim, keepdim, self.scalar_type());
  TORCH_CHECK(iter.numel() > 0, "operation does not have an identity");
  max_values_stub(iter.device_type(), iter);
  return result;
}

Tensor amax(const Tensor& self, IntArrayRef dim, bool keepdim) {
  Tensor result = at::empty({0}, self.options());
  return at::amax_out(result, self, dim, keepdim);
}

Tensor& argmax_out(Tensor& result, const Tensor& self, c10::optional<int64_t> dim, bool keepdim) {
  TORCH_CHECK(self.numel() > 0, "cannot perform reduction function argmax on a "
      "tensor with no elements because the operation does not have an identity");
  Tensor in;
  if (dim) {
    auto sizes = self.sizes();
    auto wrap_dim = maybe_wrap_dim(dim.value(), self.dim());
    if (sizes[wrap_dim] == 1) {
      if (keepdim) {
        result = at::zeros(sizes, self.options().dtype(at::kLong));
      } else {
        auto sizes_vec = sizes.vec();
        sizes_vec.erase(sizes_vec.begin() + wrap_dim);
        result = at::zeros(sizes_vec, self.options().dtype(at::kLong));
      }
      return result;
    }
    in = self;
  } else {
    in = self.reshape({-1});
    keepdim = false;
  }
  auto itr = make_reduction("argmax", result, in, dim.value_or(0), keepdim,
      self.scalar_type(), at::kLong);
  argmax_stub(itr.device_type(), itr);
  return result;
}

Tensor argmax(const Tensor& self, c10::optional<int64_t> dim, bool keepdims) {
  Tensor result = at::empty({0}, self.options().dtype(at::kLong));
  return at::native::argmax_out(result, self, dim, keepdims);
}

Tensor& argmin_out(Tensor& result, const Tensor& self, c10::optional<int64_t> dim, bool keepdim) {
  TORCH_CHECK(self.numel() > 0, "cannot perform reduction function argmin on a "
      "tensor with no elements because the operation does not have an identity");
  Tensor in;
  if (dim) {
    auto sizes = self.sizes();
    auto wrap_dim = maybe_wrap_dim(dim.value(), self.dim());
    if (sizes[wrap_dim] == 1) {
      if (keepdim) {
        result = at::zeros(sizes, self.options().dtype(at::kLong));
      } else {
        auto sizes_vec = sizes.vec();
        sizes_vec.erase(sizes_vec.begin() + wrap_dim);
        result = at::zeros(sizes_vec, self.options().dtype(at::kLong));
      }
      return result;
    }
    in = self;
  } else {
    in = self.reshape({-1});
    keepdim = false;
  }
  auto itr = make_reduction("argmin", result, in, dim.value_or(0), keepdim,
      self.scalar_type(), at::kLong);
  argmin_stub(itr.device_type(), itr);
  return result;
}

Tensor argmin(const Tensor& self, c10::optional<int64_t> dim, bool keepdims) {
  Tensor result = at::empty({0}, self.options().dtype(at::kLong));
  return at::native::argmin_out(result, self, dim, keepdims);
}

static Tensor& std_var_out(Tensor& result, const Tensor& self, IntArrayRef dim, bool unbiased, bool keepdim, bool take_sqrt) {
  TORCH_CHECK(self.device().type() == DeviceType::CPU || self.device().type() == DeviceType::CUDA,
              "std and var only supports CPU AND CUDA device type, got: ", self.device().type());
  TORCH_CHECK(self.layout() == Layout::Strided,
              "std and var only supports strided layout, got: ", self.layout());
  TORCH_CHECK(at::isFloatingType(self.scalar_type()) || at::isComplexType(self.scalar_type()),
              "std and var only support floating-point dtypes");

  if (at::isComplexType(self.scalar_type())){
    ScalarType dtype = c10::toValueType(get_dtype(result, self, {}, true));
    Tensor real_in = at::real(self);
    Tensor real_out = at::empty({0}, self.options().dtype(dtype));
    auto iter = make_reduction("std or var", real_out, real_in, dim, keepdim, dtype);
    if (iter.numel() == 0) {
      real_out.fill_(NAN);
    } else {
      std_var_stub(iter.device_type(), iter, unbiased, false);
    }
    Tensor imag_in = at::imag(self);
    Tensor imag_out = at::empty({0}, self.options().dtype(dtype));
    iter = make_reduction("std or var", imag_out, imag_in, dim, keepdim, dtype);
    if (iter.numel() == 0) {
      imag_out.fill_(NAN);
    } else {
      std_var_stub(iter.device_type(), iter, unbiased, false);
    }
    at::add_out(result, real_out, imag_out);
    take_sqrt ? at::sqrt_out(result, result) : result;
  } else{
    ScalarType dtype = get_dtype(result, self, {}, true);
    auto iter = make_reduction("std or var", result, self, dim, keepdim, dtype);
    if (iter.numel() == 0) {
      result.fill_(NAN);
    } else {
      std_var_stub(iter.device_type(), iter, unbiased, take_sqrt);
    }
  }
  return result;
}

static std::tuple<Tensor&,Tensor&> std_var_mean_out(const char* fname, Tensor &result1, Tensor &result2, const Tensor &self, IntArrayRef dim, bool unbiased, bool keepdim, bool take_sqrt) {
  AT_ASSERT(result1.defined() && result2.defined());
  TORCH_CHECK(self.device().type() == DeviceType::CPU || self.device().type() == DeviceType::CUDA,
              fname, " only supports CPU AND CUDA device type, got: ", self.device().type());
  TORCH_CHECK(self.layout() == Layout::Strided,
              fname, " only supports strided layout, got: ", self.layout());
  TORCH_CHECK(at::isFloatingType(self.scalar_type()) || at::isComplexType(self.scalar_type()),
              fname, " only support floating-point dtypes");
  TORCH_CHECK(result1.scalar_type() == result2.scalar_type(),
           "provided by result1 dtype must match dtype of result2. Got ",
           toString(result1.scalar_type()),
           " and ",
           toString(result2.scalar_type()),
           ".");
  if (at::isComplexType(self.scalar_type())){
    ScalarType dtype = c10::toValueType(get_dtype(result1, self, {}, true));
    Tensor real_in = at::real(self);
    Tensor real_out_var = at::empty({0}, self.options().dtype(dtype));
    Tensor real_out_mean = at::empty({0}, self.options().dtype(dtype));
    auto iter = make_reduction(fname, real_out_var, real_out_mean, real_in, dim, keepdim, dtype);
    if (iter.numel() == 0) {
      real_out_var.fill_(NAN);
      real_out_mean.fill_(NAN);
    } else {
      std_var_stub(iter.device_type(), iter, unbiased, false);
    }
    Tensor imag_in = at::imag(self);
    Tensor imag_out_var = at::empty({0}, self.options().dtype(dtype));
    Tensor imag_out_mean = at::empty({0}, self.options().dtype(dtype));
    iter = make_reduction(fname, imag_out_var, imag_out_mean, imag_in, dim, keepdim, dtype);
    if (iter.numel() == 0) {
      imag_out_var.fill_(NAN);
      imag_out_mean.fill_(NAN);
    } else {
      std_var_stub(iter.device_type(), iter, unbiased, false);
    }
    at::add_out(result1, real_out_var, imag_out_var);
    take_sqrt ? at::sqrt_out(result1, result1) : result1;
    at::add_out(result2, real_out_mean, at::mul(imag_out_mean, c10::complex<double>{0.0, 1.0}));
  } else {
    ScalarType dtype = get_dtype(result1, self, {}, true);
    auto iter = make_reduction(fname, result1, result2, self, dim, keepdim, dtype);
    if (iter.numel() == 0) {
      result1.fill_(NAN);
      result2.fill_(NAN);
    } else {
      std_var_stub(iter.device_type(), iter, unbiased, take_sqrt);
    }
  }
  return std::tuple<Tensor&, Tensor&>(result1, result2);
}

std::tuple<Tensor&,Tensor&> var_mean_out(Tensor &result1, Tensor &result2, const Tensor &self, IntArrayRef dim, bool unbiased, bool keepdim) {
  return std_var_mean_out("var_mean", result1, result2, self, dim, unbiased, keepdim, false);
}

std::tuple<Tensor&,Tensor&> std_mean_out(Tensor &result1, Tensor &result2, const Tensor &self, IntArrayRef dim, bool unbiased, bool keepdim) {
  return std_var_mean_out("std_mean", result1, result2, self, dim, unbiased, keepdim, true);
}

std::tuple<Tensor&,Tensor&> var_mean_out(Tensor &result1, Tensor &result2, const Tensor &self, bool unbiased) {
  return std_var_mean_out("var_mean", result1, result2, self, {}, unbiased, false, false);
}

std::tuple<Tensor&,Tensor&> std_mean_out(Tensor &result1, Tensor &result2, const Tensor &self, bool unbiased) {
  return std_var_mean_out("std_mean", result1, result2, self, {}, unbiased, false, true);
}

std::tuple<Tensor,Tensor> var_mean(const Tensor& self, IntArrayRef dim, bool unbiased, bool keepdim) {
  Tensor result1 = at::empty({0}, self.options());
  Tensor result2 = at::empty({0}, self.options());
  return at::native::var_mean_out(result1, result2, self, dim, unbiased, keepdim);
}

std::tuple<Tensor,Tensor> std_mean(const Tensor& self, IntArrayRef dim, bool unbiased, bool keepdim) {
  Tensor result1 = at::empty({0}, self.options());
  Tensor result2 = at::empty({0}, self.options());
  return at::native::std_mean_out(result1, result2, self, dim, unbiased, keepdim);
}

std::tuple<Tensor,Tensor> std_mean(const Tensor& self, bool unbiased) {
  Tensor result1 = at::empty({0}, self.options());
  Tensor result2 = at::empty({0}, self.options());
  return at::native::std_mean_out(result1, result2, self, unbiased);
}

std::tuple<Tensor,Tensor> var_mean(const Tensor& self, bool unbiased) {
  Tensor result1 = at::empty({0}, self.options());
  Tensor result2 = at::empty({0}, self.options());
  return at::native::var_mean_out(result1, result2, self, unbiased);
}

Tensor var(const Tensor& self, bool unbiased) {
  TORCH_CHECK(self.device().type() == DeviceType::CPU || self.device().type() == DeviceType::CUDA,
              "var only supports CPU AND CUDA device type, got: ", self.device().type());
  TORCH_CHECK(self.layout() == Layout::Strided,
              "var only supports strided layout, got: ", self.layout());
  TORCH_CHECK(at::isFloatingType(self.scalar_type()) || at::isComplexType(self.scalar_type()),
              "var only supports floating-point dtypes");
  auto trivial_return = _allreduce_return_trivial(self, std::numeric_limits<double>::quiet_NaN());
  if (trivial_return.has_value()) {
    return trivial_return.value();
  }

  // NOTE: CPU performance significantly regressed when attempting to port to ATen,
  //   so this dispatches differently based on device type.
  //   See https://github.com/pytorch/pytorch/pull/43858.
  if (self.device().type() == kCPU) {
    return at::_var(self, unbiased);
  }

  Tensor result = at::empty({0}, self.options());
  return std_var_out(result, self, std::vector<int64_t>{}, unbiased, false, false);
}

Tensor var(const Tensor& self, IntArrayRef dim, bool unbiased, bool keepdim) {
  Tensor result = at::empty({0}, self.options());
  return at::native::var_out(result, self, dim, unbiased, keepdim);
}

Tensor& var_out(Tensor& result, const Tensor& self, IntArrayRef dim, bool unbiased, bool keepdim) {
  return std_var_out(result, self, dim, unbiased, keepdim, false);
}

Tensor std(const Tensor& self, bool unbiased) {
  TORCH_CHECK(self.device().type() == DeviceType::CPU || self.device().type() == DeviceType::CUDA,
              "std only supports CPU AND CUDA device type, got: ", self.device().type());
  TORCH_CHECK(self.layout() == Layout::Strided,
              "std only supports strided layout, got: ", self.layout());
  TORCH_CHECK(at::isFloatingType(self.scalar_type()) || at::isComplexType(self.scalar_type()),
              "std only supports floating-point dtypes");
  auto trivial_return = _allreduce_return_trivial(self, std::numeric_limits<double>::quiet_NaN());
  if (trivial_return.has_value()) {
    return trivial_return.value();
  }

  // NOTE: CPU performance significantly regressed when attempting to port to ATen,
  //   so this dispatches differently based on device type.
  //   See https://github.com/pytorch/pytorch/pull/43858.
  if (self.device().type() == kCPU) {
    return at::_std(self, unbiased);
  }

  Tensor result = at::empty({0}, self.options());
  return std_var_out(result, self, std::vector<int64_t>{}, unbiased, false, true);
}

Tensor std(const Tensor& self, IntArrayRef dim, bool unbiased, bool keepdim) {
  Tensor result = at::empty({0}, self.options());
  return at::native::std_out(result, self, dim, unbiased, keepdim);
}

Tensor& std_out(Tensor& result, const Tensor& self, IntArrayRef dim, bool unbiased, bool keepdim) {
  return std_var_out(result, self, dim, unbiased, keepdim, true);
}

Tensor std(const Tensor& self, DimnameList dim, bool unbiased, bool keepdim) {
  return  at::std(self, dimnames_to_positions(self, dim), unbiased, keepdim);
}

Tensor& std_out(Tensor& result, const Tensor& self, DimnameList dim, bool unbiased, bool keepdim) {
  return at::std_out(result, self, dimnames_to_positions(self, dim), unbiased, keepdim);
}

Tensor var(const Tensor& self, DimnameList dim, bool unbiased, bool keepdim) {
  return  at::var(self, dimnames_to_positions(self, dim), unbiased, keepdim);
}

Tensor& var_out(Tensor& result, const Tensor& self, DimnameList dim, bool unbiased, bool keepdim) {
  return at::std_out(result, self, dimnames_to_positions(self, dim), unbiased, keepdim);
}

std::tuple<Tensor,Tensor> var_mean(const Tensor& self, DimnameList dim, bool unbiased, bool keepdim) {
  return at::var_mean(self, dimnames_to_positions(self, dim), unbiased, keepdim);
}

std::tuple<Tensor,Tensor> std_mean(const Tensor& self, DimnameList dim, bool unbiased, bool keepdim) {
  return at::std_mean(self, dimnames_to_positions(self, dim), unbiased, keepdim);
}

Tensor& norm_out(Tensor& result, const Tensor& self, optional<Scalar> p, DimnameList dim, bool keepdim, ScalarType dtype) {
  return at::norm_out(result, self, p, dimnames_to_positions(self, dim), keepdim, dtype);
}

Tensor& norm_out(Tensor& result, const Tensor& self, optional<Scalar> p, DimnameList dim, bool keepdim) {
  return at::norm_out(result, self, p, dimnames_to_positions(self, dim), keepdim);
}

Tensor norm(const Tensor& self, optional<Scalar> p, DimnameList dim, bool keepdim, ScalarType dtype) {
  return at::norm(self, p, dimnames_to_positions(self, dim), keepdim, dtype);
}

Tensor norm(const Tensor& self, optional<Scalar> p, DimnameList dim, bool keepdim) {
  return at::norm(self, p, dimnames_to_positions(self, dim), keepdim);
}

Tensor any(const Tensor& self, Dimname dim, bool keepdim) {
  reportNYIDimnameOverload("any");
}
Tensor& any_out(Tensor& result, const Tensor &self, Dimname dim, bool keepdim) {
  reportNYIDimnameOverload("any");
}
Tensor all(const Tensor& self, Dimname dim, bool keepdim) {
  reportNYIDimnameOverload("all");
}
Tensor& all_out(Tensor& result, const Tensor &self, Dimname dim, bool keepdim) {
  reportNYIDimnameOverload("all");
}
Tensor logcumsumexp(const Tensor& self, Dimname dim) {
  return at::logcumsumexp(self, dimname_to_position(self, dim));
}
Tensor& logcumsumexp_out(Tensor& result, const Tensor& self, Dimname dim) {
  return at::logcumsumexp_out(result, self, dimname_to_position(self, dim));
}
Tensor cumsum(const Tensor& self, Dimname dim, c10::optional<ScalarType> dtype) {
  return at::cumsum(self, dimname_to_position(self, dim), dtype);
}
Tensor& cumsum_(Tensor& self, Dimname dim, c10::optional<ScalarType> dtype) {
    return native::cumsum_(self, dimname_to_position(self, dim), dtype);
}
Tensor& cumsum_out(Tensor& result, const Tensor& self, Dimname dim, c10::optional<ScalarType> dtype) {
  return at::cumsum_out(result, self, dimname_to_position(self, dim), dtype);
}
Tensor cumprod(const Tensor& self, Dimname dim, c10::optional<ScalarType> dtype) {
  return at::cumprod(self, dimname_to_position(self, dim), dtype);
}
Tensor& cumprod_(Tensor& self, Dimname dim, c10::optional<ScalarType> dtype) {
    return native::cumprod_(self, dimname_to_position(self, dim), dtype);
}
Tensor& cumprod_out(Tensor& result, const Tensor& self, Dimname dim, c10::optional<ScalarType> dtype) {
  return at::cumprod_out(result, self, dimname_to_position(self, dim), dtype);
}
std::tuple<Tensor, Tensor> cummax(const Tensor& self, Dimname dim) {
  return at::cummax(self, dimname_to_position(self, dim));
}
std::tuple<Tensor&, Tensor&> cummax_out(Tensor& values, Tensor& indices, const Tensor& self, Dimname dim) {
  return at::cummax_out(values, indices, self, dimname_to_position(self, dim));
}
std::tuple<Tensor, Tensor> cummin(const Tensor& self, Dimname dim) {
  return at::cummin(self, dimname_to_position(self, dim));
}
std::tuple<Tensor&, Tensor&> cummin_out(Tensor& values, Tensor& indices, const Tensor& self, Dimname dim) {
  return at::cummin_out(values, indices, self, dimname_to_position(self, dim));
}

Tensor dist(const Tensor &self, const Tensor& other, Scalar p){
  return at::norm(self - other, p);
}

Tensor count_nonzero(const Tensor& self, IntArrayRef dims){
  auto mask = (self != 0);
  return mask.sum(dims);
}

Tensor count_nonzero(const Tensor& self, c10::optional<int64_t> dim){
  if (dim){
    auto wrap_dim = maybe_wrap_dim(dim.value(), self.dim());
    return at::count_nonzero(self, IntArrayRef{wrap_dim});
  }
  return at::count_nonzero(self, IntArrayRef{});
}

bool cpu_equal(const Tensor& self, const Tensor& other) {
  if (!at::namedinference::are_names_equal(
        self.unsafeGetTensorImpl(), other.unsafeGetTensorImpl())) {
    return false;
  }
  at::NoNamesGuard guard;
  TORCH_CHECK(self.device() == other.device(), "Cannot compare two tensors on "
              "different devices. Got: ", self.device(), " and ", other.device());
  TORCH_CHECK(self.dtype() == other.dtype(),
              "Expected object of scalar type ", self.dtype(), " but got scalar type ",
              other.dtype(), " for argument 'other'");
  if (!self.is_same_size(other)) {
    return false;
  }
  std::atomic<bool> result{true};
  auto iter = TensorIteratorConfig()
    .add_input(self)
    .add_input(other)
    .allow_cpu_scalars(true)
    .promote_inputs_to_common_dtype(true)
    .build();

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(kBool, kBFloat16, kHalf, iter.input_dtype(), "equal_cpu", [&] {
    iter.for_each([&](char** data, const int64_t *strides, int64_t dim_size) {
      if (!result) {
          return;
      }
      char* self_data = data[0];
      char* other_data = data[1];
      for (int64_t i = 0; i < dim_size; ++i) {
        if (*((scalar_t*)self_data) != *((scalar_t*)other_data)) {
          result = false;
          return;
        }
        self_data += strides[0];
        other_data += strides[1];
      }
    });
  });
  return result.load();
}

// max(dim), min(dim), topk(dim), mode(dim), are examples of reduction
// functions that select values. value_selecting_reduction_backward is the
// backward function for those operators; it propagates the grad to the
// specific value locations referred to at `indices`.
Tensor value_selecting_reduction_backward(const Tensor& grad, int64_t dim, const Tensor& indices, IntArrayRef sizes, bool keepdim) {
  if (!keepdim && sizes.size() > 0) {
    auto grad_ = grad.unsqueeze(dim);
    auto indices_ = indices.unsqueeze(dim);
    return at::zeros(sizes, grad_.options()).scatter_(dim, indices_, grad_);
  }
  return at::zeros(sizes, grad.options()).scatter_(dim, indices, grad);
}

}} // namespace at::native
