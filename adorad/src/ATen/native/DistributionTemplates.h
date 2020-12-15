#pragma once

#include <ATen/Dispatch.h>
#include <ATen/Generator.h>
#include <ATen/Tensor.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/native/TensorIterator.h>
#include <c10/util/Optional.h>
#include <limits>
#include <cmath>

namespace at {
namespace native {
namespace templates {

// ==================================================== Random ========================================================

// The purpose of `update_from` and `update_to` is to find the closest valid int64_t number that can be used as actual `from`.
// The current implementation of `random_` uses uint64_t arithmetics and casts the result to the target dtype(scalar_t).
// This casting can result in generating numbers that happen to be greater or equal to `to` value. For instance:
//
//    auto actual = torch::empty({3, 3}, torch::half);
//    actual.random_(0, 65504);
//
// If random's uint64_t arithmetics produces 65503 as a random value after casting to torch::half it becomes 65504
// and violates the requirement that random value must be less than `to`. To resolve this issue `update_from` and `update_to`
// moves `from` to the left and `to` to the right to the next closest value that won't go outside [from, to) after casting to
// the target dtype. For `to` = 65504 it moves left for (1 << (log2(to) - 11 + 1)) = 32 and becomes 65472, which is previous
// available number for torch::half dtype.
template<typename scalar_t>
int64_t update_from(int64_t from) {
  static_assert(
    std::is_floating_point<scalar_t>::value ||
    std::is_same<scalar_t, at::Half>::value ||
    std::is_same<scalar_t, at::BFloat16>::value, "scalar_t must be floating-point type");
  const auto from_plus_1 = static_cast<int64_t>(static_cast<scalar_t>(from + 1));
  if (from_plus_1 < from) {
    int64_t from_ = std::abs(from + 1);
    int n = 0;
    while (from_ >>= 1) ++n;
    from = from_plus_1 + (1LL << (n - std::numeric_limits<scalar_t>::digits + 1));
  }
  return from;
}

template<typename scalar_t>
int64_t update_to(int64_t to) {
  static_assert(
    std::is_floating_point<scalar_t>::value ||
    std::is_same<scalar_t, at::Half>::value ||
    std::is_same<scalar_t, at::BFloat16>::value, "scalar_t must be floating-point type");
  const auto to_minus_1 = static_cast<int64_t>(static_cast<scalar_t>(to - 1));
  if (to_minus_1 >= to) {
    int64_t to_ = std::abs(to - 1);
    int n = 0;
    while (to_ >>= 1) ++n;
    to = to_minus_1 - (1LL << (n - std::numeric_limits<scalar_t>::digits + 1));
  }
  return to;
}

template<template<typename> class random_kernel, typename RNG>
at::Tensor& random_impl(at::Tensor& self, c10::optional<Generator> generator) {
  auto iter = at::TensorIterator::nullary_op(self);
  random_kernel<RNG>()(iter, generator);
  return self;
}

#define CHECK_OUT_OF_BOUNDS(var, name, min, max, dtype) \
  TORCH_CHECK(var >= min && var <= max, name , " is out of bounds for ", dtype); \

#define WARN_OUT_OF_BOUNDS(var, name, digits, dtype) \
  if (var < -(1LL << digits) || var > (1LL << digits)) { \
    TORCH_WARN(name , " is out of bounds [-(2^", digits, "), 2^", digits, "]. ", \
      "Due to precision limitations ", dtype, " can support discrete uniform distribution only within this range. ", \
      "This warning will become an error in version 1.7 release, please fix the code in advance"); \
  }

static void check_from_to_in_range(int64_t from, int64_t to_inc, caffe2::TypeMeta dtype) {
  const auto scalar_type = typeMetaToScalarType(dtype);
  if (isFloatingType(scalar_type)) {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, scalar_type, "check_random_fp_bounds", [&] {
      const auto min = static_cast<double>(std::numeric_limits<scalar_t>::lowest());
      const auto max = static_cast<double>(std::numeric_limits<scalar_t>::max());
      CHECK_OUT_OF_BOUNDS(from, "from", min, max, dtype);
      CHECK_OUT_OF_BOUNDS(to_inc, "to - 1", min, max, dtype);

      constexpr auto digits = std::numeric_limits<scalar_t>::digits;
      WARN_OUT_OF_BOUNDS(from, "from", digits, dtype);
      WARN_OUT_OF_BOUNDS(to_inc, "to - 1", digits, dtype);
    });
  } else if (isIntegralType(scalar_type, /*includeBool=*/true)) {
    AT_DISPATCH_INTEGRAL_TYPES_AND(at::ScalarType::Bool, scalar_type, "check_random_integral_bounds", [&]() {
      const auto min = static_cast<int64_t>(std::numeric_limits<scalar_t>::lowest());
      const auto max = static_cast<int64_t>(std::numeric_limits<scalar_t>::max());
      CHECK_OUT_OF_BOUNDS(from, "from", min, max, dtype);
      CHECK_OUT_OF_BOUNDS(to_inc, "to - 1", min, max, dtype);
    });
  } else {
    TORCH_CHECK(false, "check_random_bounds handles only integral, floating-point and boolean types");
  }
}

template<template<typename> class random_from_to_kernel, typename RNG>
at::Tensor& random_from_to_impl(at::Tensor& self, int64_t from, c10::optional<int64_t> to_opt, c10::optional<Generator> generator) {
  uint64_t range = 0;
  auto iter = at::TensorIterator::nullary_op(self);
  if (to_opt.has_value()) {
    // [from, to)
    int64_t to = *to_opt;
    TORCH_CHECK(from < to, "random_ expects 'from' to be less than 'to', but got from=", from, " >= to=", to);
    if (isFloatingType(iter.dtype())) {
      AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, self.scalar_type(), "random_update_from_to", [&] {
        from = update_from<scalar_t>(from);
        to = update_to<scalar_t>(to);
        TORCH_CHECK(from < to, "random_ expects 'from' casted to dtype to be less than 'to' casted to dtype, but got from=", from, " >= to=", to);
      });
    }
    check_from_to_in_range(from, to - 1, self.dtype());
    range = static_cast<uint64_t>(to) - static_cast<uint64_t>(from);
    random_from_to_kernel<RNG>()(iter, range, from, generator);
  } else if (from != std::numeric_limits<int64_t>::lowest()) {
    // [from, std::numeric_limits<int64_t>::max()]
    int64_t to_inc = 0;
    if (isFloatingType(iter.dtype())) {
      AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, self.scalar_type(), "random_from_to_range_calc", [&] {
        constexpr int64_t scalar_t_max = static_cast<int64_t>(1) << std::numeric_limits<scalar_t>::digits;
        to_inc = scalar_t_max > std::numeric_limits<int64_t>::max() ? std::numeric_limits<int64_t>::max() : static_cast<int64_t>(scalar_t_max);
        from = update_from<scalar_t>(from);
        TORCH_CHECK(from < to_inc, "random_ expects 'from' casted to dtype to be less than or equal to 'to_inc' casted to dtype, but got from=", from, " > to_inc=", to_inc);
      });
    } else if (isIntegralType(iter.dtype(), /*includeBool=*/true)) {
      AT_DISPATCH_INTEGRAL_TYPES_AND(at::ScalarType::Bool, self.scalar_type(), "random_from_to_range_calc", [&] {
        if (std::is_same<scalar_t, bool>::value) {
          to_inc = static_cast<int64_t>(true);
        } else {
          to_inc = static_cast<int64_t>(std::numeric_limits<scalar_t>::max());
        }
      });
    } else {
      TORCH_CHECK(false, "random_from_to_impl handles only integral, floating-point and boolean types");
    }
    check_from_to_in_range(from, to_inc, self.dtype());
    range = static_cast<uint64_t>(to_inc) - static_cast<uint64_t>(from) + 1;
    random_from_to_kernel<RNG>()(iter, range, from, generator);
  } else {
    // [std::numeric_limits<int64_t>::lowest(), std::numeric_limits<int64_t>::max()]
    // range = 2^64
    random_from_to_kernel<RNG>()(iter, generator);
  }
  return self;
}

// ==================================================== Normal ========================================================

// This function computes broadcasted size of mean and std, resize the output to the broadcasted size if it was empty
// [Note] The following features will be deprecated in version 1.6 release and function signature will be changed after
//   When mean and std are not broadcastable but have same number of elements:
//     This function will resize the output to the size of mean if it was empty.
//     This function will reshape the std to the shape of mean.
//     This function will return true in deprecated case, false in broadcastable case and throw in all other cases before deprecation.
//     This function will not return and throw if mean and std are not broadcastable after deprecation
static bool resize_output_for_normal(at::Tensor& output, const at::Tensor& mean, const at::Tensor& std) {
  bool expandable = at::are_expandable(mean.sizes(), std.sizes());
  bool empty_output = output.numel() == 0;

  if (expandable) {
    auto shape = at::infer_size(mean.sizes(), std.sizes());
    TORCH_CHECK(
        empty_output || output.sizes().equals(shape),
        "inconsistent tensor, output size (", output.sizes(), ") is not the same as broadcasted mean and std size (", shape, ")");
    if (empty_output) {
      at::native::resize_(output, shape);
    }
    return false;
  }
  else {
    TORCH_CHECK(
        mean.numel() == std.numel(),
        "inconsistent tensor, std and mean are not broadcastable and have different number of elements, "
        "expected mean ", mean.sizes(), " and std ", std.sizes(), " to have same number of elements)");
    TORCH_CHECK(
        empty_output || output.sizes().equals(mean.sizes()),
        "inconsistent tensor, std and mean are not broadcastable, output size (", output.sizes(), ") is not the same as mean size (", mean.sizes(), ")");
    TORCH_WARN_ONCE(
        "std and mean have the same number of elements, but are not broadcastable. This was previously a "
        "supported mode of operation, but is now deprecated and the support will be removed in version 1.6 release. "
        "Note that the current implementation reshapes std to the shape of mean, which may be incur data copies. "
        "Please ensure that std and mean are broadcastable to avoid these issues.");
    if (empty_output) {
      at::native::resize_(output, mean.sizes());
    }
    return true;
  }
}

template<template<typename> class normal_kernel, typename RNG>
Tensor& normal_impl_(Tensor& self, double mean, double std, c10::optional<Generator> gen) {
  TORCH_CHECK(std > 0.0, "normal_ expects std > 0.0, but found std=", std);
  if (self.is_complex()) {
    auto float_tensor = at::view_as_real(self);
    // variance for normal distribution of the real and imaginary values
    // is half of the input variance
    normal_kernel<RNG>()(float_tensor, mean, std/(std::sqrt(2)), gen);
  } else {
    normal_kernel<RNG>()(self, mean, std, gen);
  }
  return self;
}

template<template<typename> class normal_kernel, typename RNG>
Tensor& normal_out_impl(Tensor& output, const Tensor& mean, double std, c10::optional<Generator> gen) {
  normal_impl_<normal_kernel, RNG>(output, 0, std, gen);
  output.add_(mean);
  return output;
}

template<template<typename> class normal_kernel, typename RNG>
Tensor& normal_out_impl(Tensor& output, double mean, const Tensor& std, c10::optional<Generator> gen) {
  TORCH_CHECK(!std.is_complex(), "normal expects standard deviation to be non-complex");
  normal_impl_<normal_kernel, RNG>(output, 0, 1, gen);
  auto mean_tensor = at::full({}, mean, output.options());
  // CUDA NB: addcmul_out copies the tensor to be added into the output.
  // Please look at aten/src/THC/generic/THCTensorMathPointwise.cu
  // The previous function here was addcmul_out(output, mean_tensor, output, std, 1);
  // The third argument is not a constant reference and hence the samples in output are overwritten.
  // Consequently, the computation performed is mean_tensor + mean_tensor * std instead of mean_tensor + output * std
  output.mul_(std).add_(mean_tensor);
  return output;
}

template<template<typename> class normal_kernel, typename RNG>
Tensor& normal_out_impl(Tensor& output, const Tensor& mean, const Tensor& std, c10::optional<Generator> gen) {
  TORCH_CHECK(!std.is_complex(), "normal expects standard deviation to be non-complex");
  bool is_deprecated_th_impl = resize_output_for_normal(output, mean, std);
  normal_impl_<normal_kernel, RNG>(output, 0, 1, gen);
  // CUDA NB: addcmul_out copies the tensor to be added into the output.
  // Please look at aten/src/THC/generic/THCTensorMathPointwise.cu
  // The previous function here was addcmul_out(output, mean, output, std, 1);
  // The third argument is not a constant reference and hence the samples in output are overwritten.
  // Consequently, the computation performed is mean + mean * std instead of mean + output * std
  if (is_deprecated_th_impl) {
    output.mul_(std.reshape(mean.sizes())).add_(mean);
  }
  else {
    output.mul_(std).add_(mean);
  }
  return output;
}

template<template<typename> class normal_kernel, typename RNG>
Tensor normal_impl(const Tensor& mean, double std, c10::optional<Generator> gen) {
  Tensor ret = at::empty_like(mean, MemoryFormat::Contiguous);
  normal_out_impl<normal_kernel, RNG>(ret, mean, std, gen);
  return ret;
}

template<template<typename> class normal_kernel, typename RNG>
Tensor normal_impl(double mean, const Tensor& std, c10::optional<Generator> gen) {
  Tensor ret = at::empty_like(std, MemoryFormat::Contiguous);
  normal_out_impl<normal_kernel, RNG>(ret, mean, std, gen);
  return ret;
}

template<template<typename> class normal_kernel, typename RNG>
Tensor normal_impl(const Tensor& mean, const Tensor& std, c10::optional<Generator> gen) {
  Tensor ret = at::empty({0}, mean.options(), MemoryFormat::Contiguous);
  normal_out_impl<normal_kernel, RNG>(ret, mean, std, gen);
  return ret;
}

// ==================================================== Uniform =======================================================

template<template<typename> class uniform_kernel, typename RNG>
at::Tensor& uniform_impl_(at::Tensor& self, double from, double to, c10::optional<Generator> generator) {
  if (self.is_complex()) {
    auto float_tensor = at::view_as_real(self);
    uniform_impl_<uniform_kernel, RNG>(float_tensor, from, to, generator);
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, self.scalar_type(), "check_uniform_bounds", [&] {
      const auto dtype = self.dtype();
      const auto min = static_cast<double>(std::numeric_limits<scalar_t>::lowest());
      const auto max = static_cast<double>(std::numeric_limits<scalar_t>::max());
      CHECK_OUT_OF_BOUNDS(from, "from", min, max, dtype);
      CHECK_OUT_OF_BOUNDS(to, "to", min, max, dtype);
      TORCH_CHECK(from <= to, "uniform_ expects to return a [from, to) range, but found from=", from, " > to=", to);
      TORCH_CHECK((to - from) <= std::numeric_limits<scalar_t>::max(),
            "uniform_ expects to-from <= std::numeric_limits<", toString(self.scalar_type()),
            ">::max(), but found to=", to, " and from=", from,
            " which result in to-from to exceed the limit");
      from = std::min(std::max(from, min), max);
      to = std::max(std::min(to, max), min);
    });
    auto iter = at::TensorIterator::nullary_op(self);
    uniform_kernel<RNG>()(iter, from, to, generator);
  }
  return self;
}

// ================================================== LogNormal =======================================================

template<template<typename> class log_normal_kernel, typename RNG>
at::Tensor& log_normal_impl_(at::Tensor& self, double mean, double std, c10::optional<Generator> gen) {
  TORCH_CHECK(std > 0.0, "log_normal_ expects std > 0.0, but found std=", std);
  auto iter = TensorIterator::nullary_op(self);
  log_normal_kernel<RNG>()(iter, mean, std, gen);
  return self;
}

// =================================================== Geometric ======================================================

template<template<typename> class geometric_kernel, typename RNG>
Tensor& geometric_impl_(Tensor& self, double p, c10::optional<Generator> gen) {
  TORCH_CHECK(0 < p && p < 1, "geometric_ expects p to be in (0, 1), but got p=", p);
  auto iter = TensorIterator::nullary_op(self);
  geometric_kernel<RNG>()(iter, p, gen);
  return self;
}

// ================================================== Exponential =====================================================

template<template<typename> class exponential_kernel, typename RNG>
Tensor& exponential_impl_(Tensor& self, double lambda, c10::optional<Generator> gen) {
  TORCH_CHECK(lambda >= 0.0, "exponential_ expects lambda >= 0.0, but found lambda=", lambda);
  auto iter = TensorIterator::nullary_op(self);
  exponential_kernel<RNG>()(iter, lambda, gen);
  return self;
}

// ==================================================== Cauchy ========================================================

template<template<typename> class cauchy_kernel, typename RNG>
Tensor& cauchy_impl_(Tensor& self, double median, double sigma, c10::optional<Generator> gen) {
  auto iter = TensorIterator::nullary_op(self);
  cauchy_kernel<RNG>()(iter, median, sigma, gen);
  return self;
}

// ==================================================== Bernoulli =====================================================

template<template<typename> class bernoulli_tensor_kernel, typename RNG>
Tensor& bernoulli_impl_(Tensor& self, const Tensor& p_, c10::optional<Generator> gen) {
  NoNamesGuard guard;
  at::assert_no_internal_overlap(self);
  bernoulli_tensor_kernel<RNG>()(self, p_, gen);
  return self;
}

template<template<typename> class bernoulli_scalar_kernel, typename RNG>
Tensor& bernoulli_impl_(Tensor& self, double p, c10::optional<Generator> gen) {
  TORCH_CHECK(0 <= p && p <= 1, "bernoulli_ expects p to be in [0, 1], but got p=", p);
  at::assert_no_internal_overlap(self);
  bernoulli_scalar_kernel<RNG>()(self, p, gen);
  return self;
}

template<template<typename> class bernoulli_tensor_kernel, typename RNG>
Tensor& bernoulli_out_impl(Tensor& result, const Tensor& self, c10::optional<Generator> gen) {
  // result.resize_as_(self) requires self to have same dtype as result, so we
  // use resize_ instead.
  // TODO: Fix resize_as_. See pytorch/pytorch#11665.
  result.resize_(self.sizes());
  bernoulli_impl_<bernoulli_tensor_kernel, RNG>(result, self, gen);
  namedinference::propagate_names(result, self);
  return result;
}

#undef CHECK_OUT_OF_BOUNDS
#undef WARN_OUT_OF_BOUNDS

}}}
