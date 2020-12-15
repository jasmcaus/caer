#pragma once
// Please note that this file is
// used across both CPU and GPU.

#include <type_traits>
#include <complex>
#include <c10/macros/Macros.h>
#include <ATen/detail/FunctionTraits.h>
#include <ATen/NumericUtils.h>
#if defined(__CUDACC__)
#include <THC/THCDeviceUtils.cuh>
#include <ATen/native/cuda/DeviceSqrt.cuh>
#elif defined(__HIPCC__)
#include <aten/src/THH/THHDeviceUtils.cuh>
#include <aten/src/ATen/native/hip/DeviceSqrt.cuh>
#endif
#if defined(__CUDACC__) || defined(__HIPCC__)
#include <thrust/pair.h>
#else
#include <cmath>
#define device_sqrt std::sqrt
#endif
#if defined(__CUDACC__) || defined(__HIPCC__)
#define MAX(X, Y) ::max(X,Y)
#define MIN(X, Y) ::min(X,Y)
#else
#define MAX(X, Y) max_impl(X,Y)
#define MIN(X, Y) min_impl(X,Y)
#endif

// ROCM hcc doesn't work well with using std:: in kernel functions
#if defined(__CUDA_ARCH__)
#include <c10/cuda/CUDAMathCompat.h>
#define compat_pow c10::cuda::compat::pow
#elif defined(__HIPCC__)
#include <c10/hip/HIPMathCompat.h>
#define compat_pow c10::hip::compat::pow
#else
#define compat_pow std::pow
#endif

namespace at { namespace native {

namespace detail {

#if defined(__CUDACC__) || defined(__HIPCC__)
template <typename T1, typename T2> using pair = thrust::pair<T1, T2>;
#else
template <typename T1, typename T2> using pair = std::pair<T1, T2>;
#endif

} // namespace detail

template <typename scalar_t, typename index_t, typename combine_t>
struct WelfordData {
  scalar_t mean;
  scalar_t m2;
  index_t n;
  combine_t nf;
  C10_HOST_DEVICE WelfordData() : mean(0), m2(0), n(0), nf(0)  {}
  C10_DEVICE WelfordData(scalar_t mean, scalar_t m2, index_t n, combine_t nf) : mean(mean), m2(m2), n(n), nf(nf) {}
};


template <typename scalar_t, typename acc_scalar_t, typename index_t, typename combine_t, typename res_t>
struct WelfordOps {
  bool unbiased;
  bool take_sqrt;
 public:
  using acc_t = WelfordData<acc_scalar_t, index_t, combine_t>;
  inline C10_DEVICE acc_t reduce(acc_t acc, scalar_t data, index_t /*idx*/) const {
    acc_scalar_t delta = data - acc.mean;
    // using acc.nf(combine_t) here, as acc.n(index_t) would still be converted
    // accumulation in reduce is done through index_T
    acc_scalar_t new_mean = acc.mean + delta / (acc.nf + 1);
    acc_scalar_t new_delta = data - new_mean;
    return {
      new_mean,
      acc.m2 + delta * new_delta,
      acc.n + 1,
      combine_t(acc.n + 1), // accumulate for combine_t uses index_t
    };
  }
  inline C10_DEVICE acc_t combine(acc_t a, acc_t b) const {
    if (a.nf == 0) {
      return b;
    }
    if (b.nf == 0) {
      return a;
    }
    acc_scalar_t delta = b.mean - a.mean;
    combine_t new_count = a.nf + b.nf;
    acc_scalar_t nb_over_n = b.nf / new_count;
    return {
      a.mean + delta * nb_over_n,
      a.m2 + b.m2 + delta * delta * a.nf * nb_over_n,
      // setting acc.n as -1 since acc.n might not be able to represent the count
      // correctly within its range, setting it to -1 to avoid confusion
      -1,
      new_count
    };
  }
  inline C10_DEVICE res_t project(acc_t acc) const {
    auto mean = acc.mean;
    combine_t divisor = unbiased ? (acc.nf - 1) : acc.nf;
    auto ret = (divisor > 0) ?
      (take_sqrt ? device_sqrt(acc.m2 / divisor) : (acc.m2 / divisor))
      : NAN;
    detail::pair<scalar_t, scalar_t> results{(scalar_t) ret, (scalar_t) mean};
    return results;
  }

  static C10_DEVICE acc_t translate_idx(acc_t acc, int64_t /*base_idx*/) {
    return acc;
  }

#if defined(__CUDACC__) || defined(__HIPCC__)
  inline __device__ acc_t warp_shfl_down(acc_t acc, int offset) const {
    return {
      WARP_SHFL_DOWN(acc.mean, offset)
      , WARP_SHFL_DOWN(acc.m2, offset)
      , WARP_SHFL_DOWN(acc.n, offset)
      , WARP_SHFL_DOWN(acc.nf, offset)
    };
  }
#endif
  WelfordOps(bool unbiased, bool take_sqrt)
    : unbiased(unbiased), take_sqrt(take_sqrt) {
  }
};

template <typename acc_t, typename factor_t>
struct MeanOps {
  factor_t factor;

  inline C10_DEVICE acc_t reduce(acc_t a, acc_t b, int64_t /*idx*/) const {
    return combine(a, b);
  }

  inline C10_DEVICE acc_t combine(acc_t a, acc_t b) const {
    return a + b;
  }

  inline C10_DEVICE acc_t project(acc_t a) const {
    return a * factor;
  }

  static C10_DEVICE acc_t translate_idx(acc_t acc, int64_t /*base_idx*/) {
    return acc;
  }

#if defined(__CUDACC__) || defined(__HIPCC__)
  inline C10_DEVICE acc_t warp_shfl_down(acc_t data, int offset) const {
    return WARP_SHFL_DOWN(data, offset);
  }
#endif

  MeanOps(factor_t factor): factor(factor) {
  }
};

// This accumulator template is used to calculate the minimum absolute value of
// a set of numbers.
// `scalar_t` is the type of the input and `acc_t` is the type of the accumulated
// value. These types differ for complex number input support.
template <typename scalar_t, typename acc_t=scalar_t>
struct AbsMinOps {

  inline C10_DEVICE acc_t reduce(acc_t acc, scalar_t data, int64_t /*idx*/) const {
    return MIN(acc, static_cast<acc_t>(std::abs(data)));
  }

  inline C10_DEVICE acc_t combine(acc_t a, acc_t b) const {
    return MIN(a, b);
  }

  inline C10_DEVICE acc_t project(acc_t a) const {
    return a;
  }

  static C10_DEVICE acc_t translate_idx(acc_t acc, int64_t /*base_idx*/) {
    return acc;
  }

#if defined(__CUDACC__) || defined(__HIPCC__)
  inline C10_DEVICE acc_t warp_shfl_down(acc_t acc, int offset) const {
    return WARP_SHFL_DOWN(acc, offset);
  }
#endif
};

// This accumulator template is used to calculate the maximum absolute value of
// a set of numbers.
// `scalar_t` is the type of the input and `acc_t` is the type of the accumulated
// value. These types differ for complex number input support.
template <typename scalar_t, typename acc_t=scalar_t>
struct AbsMaxOps {

  inline C10_DEVICE acc_t reduce(acc_t acc, scalar_t data, int64_t /*idx*/) const {
    return MAX(acc, static_cast<acc_t>(std::abs(data)));
  }

  inline C10_DEVICE acc_t combine(acc_t a, acc_t b) const {
    return MAX(a, b);
  }

  inline C10_DEVICE acc_t project(acc_t a) const {
    return a;
  }

  static C10_DEVICE acc_t translate_idx(acc_t acc, int64_t /*base_idx*/) {
    return acc;
  }

#if defined(__CUDACC__) || defined(__HIPCC__)
  inline C10_DEVICE acc_t warp_shfl_down(acc_t acc, int offset) const {
    return WARP_SHFL_DOWN(acc, offset);
  }
#endif
};

// This accumulator template is used to calculate the norm of the absolute value
// of a set of numbers.
// `scalar_t` is the type of the input and `acc_t` is the type of the accumulated
// value. These types differ for complex number input support.
template <typename scalar_t, typename acc_t=scalar_t>
struct NormOps {
  acc_t norm_;

  inline C10_DEVICE acc_t reduce(acc_t acc, scalar_t data, int64_t /*idx*/) const {
    return acc + compat_pow(static_cast<acc_t>(std::abs(data)), norm_);
  }

  inline C10_DEVICE acc_t combine(acc_t a, acc_t b) const {
    return a + b;
  }

  inline C10_DEVICE acc_t project(acc_t a) const {
    return compat_pow(a, static_cast<acc_t>(1.0) / norm_);
  }

  static C10_DEVICE acc_t translate_idx(acc_t acc, int64_t /*base_idx*/) {
    return acc;
  }

#if defined(__CUDACC__) || defined(__HIPCC__)
  inline C10_DEVICE acc_t warp_shfl_down(acc_t acc, int offset) const {
    return WARP_SHFL_DOWN(acc, offset);
  }
#endif

  NormOps(acc_t norm_): norm_(norm_) {
  }
};

// This accumulator template is used to calculate the order zero norm of the
// absolute value of a set of numbers.
// `scalar_t` is the type of the input and `acc_t` is the type of the accumulated
// value. These types differ for complex number input support.
template <typename scalar_t, typename acc_t=scalar_t>
struct NormZeroOps {
  inline C10_DEVICE acc_t reduce(acc_t acc, scalar_t data, int64_t /*idx*/) const {
    return acc + (data == static_cast<scalar_t>(0) ? static_cast<acc_t>(0) : static_cast<acc_t>(1));
  }

  inline C10_DEVICE acc_t combine(acc_t a, acc_t b) const {
    return a + b;
  }

  inline C10_DEVICE acc_t project(acc_t a) const {
    return a;
  }

  static C10_DEVICE acc_t translate_idx(acc_t acc, int64_t /*base_idx*/) {
    return acc;
  }


#if defined(__CUDACC__) || defined(__HIPCC__)
  inline C10_DEVICE acc_t warp_shfl_down(acc_t acc, int offset) const {
    return WARP_SHFL_DOWN(acc, offset);
  }
#endif
};

// This accumulator template is used to calculate the order one norm of the
// absolute value of a set of numbers.
// `scalar_t` is the type of the input and `acc_t` is the type of the accumulated
// value. These types differ for complex number input support.
template <typename scalar_t, typename acc_t=scalar_t>
struct NormOneOps {
  inline C10_DEVICE acc_t reduce(acc_t acc, scalar_t data, int64_t /*idx*/) const {
    return acc + static_cast<acc_t>(std::abs(data));
  }

  inline C10_DEVICE acc_t combine(acc_t a, acc_t b) const {
    return a + b;
  }

  inline C10_DEVICE acc_t project(acc_t a) const {
    return a;
  }

  static C10_DEVICE acc_t translate_idx(acc_t acc, int64_t /*base_idx*/) {
    return acc;
  }

#if defined(__CUDACC__) || defined(__HIPCC__)
  inline C10_DEVICE acc_t warp_shfl_down(acc_t acc, int offset) const {
    return WARP_SHFL_DOWN(acc, offset);
  }
#endif
};


template<typename acc_t>
struct AbsSwitch {};

template<typename scalar_t, typename acc_t>
inline C10_DEVICE acc_t abs_if_complex(scalar_t data, AbsSwitch<acc_t> s) {
  return static_cast<acc_t>(data);
}

template<typename scalar_t, typename acc_t>
inline C10_DEVICE acc_t abs_if_complex(std::complex<scalar_t> data, AbsSwitch<acc_t> s) {
  return static_cast<acc_t>(std::abs(data));
}

template<typename scalar_t, typename acc_t>
inline C10_DEVICE acc_t abs_if_complex(c10::complex<scalar_t> data, AbsSwitch<acc_t> s) {
  return static_cast<acc_t>(std::abs(data));
}

// This accumulator template is used to calculate the order two norm of the
// absolute value of a set of numbers.
// `scalar_t` is the type of the input and `acc_t` is the type of the accumulated
// value. These types differ for complex number input support.
template <typename scalar_t, typename acc_t=scalar_t>
struct NormTwoOps {
  inline C10_DEVICE acc_t reduce(acc_t acc, scalar_t data, int64_t /*idx*/) const {
    acc_t data_ = abs_if_complex(data, AbsSwitch<acc_t>());
    return acc + data_ * data_;
  }

  inline C10_DEVICE acc_t combine(acc_t a, acc_t b) const {
    return a + b;
  }

  inline C10_DEVICE acc_t project(acc_t a) const {
    return device_sqrt(a);
  }

  static C10_DEVICE acc_t translate_idx(acc_t acc, int64_t /*base_idx*/) {
    return acc;
  }

#if defined(__CUDACC__) || defined(__HIPCC__)
  inline C10_DEVICE acc_t warp_shfl_down(acc_t acc, int offset) const {
    return WARP_SHFL_DOWN(acc, offset);
  }
#endif
};

template <typename acc_t, typename data_t>
struct NanSumOps {
  inline C10_DEVICE acc_t reduce(acc_t a, data_t b, int64_t /*idx*/) const {
    return a + (at::_isnan(b) ? acc_t{0.} : acc_t{b});
  }

  inline C10_DEVICE acc_t combine(acc_t a, acc_t b) const {
    return  a + b;
  }

  inline C10_DEVICE data_t project(acc_t a) const {
    return data_t{a};
  }

  static C10_DEVICE acc_t translate_idx(acc_t acc, int64_t /*base_idx*/) {
    return acc;
  }

#if defined(__CUDACC__) || defined(__HIPCC__)
  inline C10_DEVICE acc_t warp_shfl_down(acc_t data, int offset) const {
    return WARP_SHFL_DOWN(data, offset);
  }
#endif
};

template <typename acc_t>
struct AndOps {
  inline C10_DEVICE acc_t reduce(acc_t a, acc_t b, int64_t /*idx*/) const {
    return static_cast<bool>(a) && static_cast<bool>(b);
  }

  inline C10_DEVICE acc_t combine(acc_t a, acc_t b) const {
    return static_cast<bool>(a) && static_cast<bool>(b);
  }

  inline C10_DEVICE acc_t project(acc_t a) const {
    return a;
  }

  static C10_DEVICE acc_t translate_idx(acc_t acc, int64_t /*base_idx*/) {
    return acc;
  }

#if defined(__CUDACC__) || defined(__HIPCC__)
  inline C10_DEVICE acc_t warp_shfl_down(acc_t data, int offset) const {
    return WARP_SHFL_DOWN(data, offset);
  }
#endif
};

template <typename acc_t>
struct OrOps {
  inline C10_DEVICE acc_t reduce(acc_t a, acc_t b, int64_t /*idx*/) const {
    return static_cast<bool>(a) || static_cast<bool>(b);
  }

  inline C10_DEVICE acc_t combine(acc_t a, acc_t b) const {
    return static_cast<bool>(a) || static_cast<bool>(b);
  }

  inline C10_DEVICE acc_t project(acc_t a) const {
    return a;
  }

  static C10_DEVICE acc_t translate_idx(acc_t acc, int64_t /*base_idx*/) {
    return acc;
  }

#if defined(__CUDACC__) || defined(__HIPCC__)
  inline C10_DEVICE acc_t warp_shfl_down(acc_t data, int offset) const {
    return WARP_SHFL_DOWN(data, offset);
  }
#endif
};

namespace detail {

template <typename scalar_t>
struct LessOrNan {
  C10_DEVICE bool operator () (scalar_t a, scalar_t b, int64_t idx_a, int64_t idx_b) const {
    // If (a == b), then choose the one with lower idx, else min(a, b)
    if (at::_isnan(a)) {
      if (at::_isnan(b)) {
        return idx_a < idx_b;
      }
      return true;
    }
    return (a == b) ? idx_a < idx_b : (a < b);
  }
};

template <typename scalar_t>
struct GreaterOrNan {
  C10_DEVICE bool operator () (scalar_t a, scalar_t b, int64_t idx_a, int64_t idx_b) const {
    // If (a == b), then choose the one with lower idx, else max(a, b)
    if (at::_isnan(a)) {
      if (at::_isnan(b)) {
        return idx_a < idx_b;
      }
      return true;
    }
    return (a == b) ? idx_a < idx_b : (a > b);
  }
};

template <typename comp_t>
struct MinMaxReductionOps {
  using scalar_t = typename binary_function_traits<comp_t>::arg1_t;
  using index_t = int64_t;
  using arg_t = detail::pair<scalar_t, index_t>;

  static C10_DEVICE arg_t project(arg_t arg) {
    return arg;
  }

  static C10_DEVICE arg_t reduce(arg_t arg, scalar_t val, int64_t idx) {
    return comp_t{}(arg.first, val, arg.second, idx) ? arg : arg_t(val, idx);
  }

  static C10_DEVICE arg_t combine(arg_t a, arg_t b) {
    return comp_t{}(a.first, b.first, a.second, b.second) ? a : b;
  }

  static C10_DEVICE arg_t translate_idx(arg_t a, int64_t base_idx) {
    return {a.first, a.second + base_idx};
  }

#if defined(__CUDACC__) || defined(__HIPCC__)
  static C10_DEVICE arg_t warp_shfl_down(arg_t arg, int offset) {
    return arg_t(WARP_SHFL_DOWN(arg.first, offset),
                 WARP_SHFL_DOWN(arg.second, offset));
  }
#endif
};

template <typename comp_t>
struct ArgReductionOps : public MinMaxReductionOps<comp_t> {
  using typename MinMaxReductionOps<comp_t>::scalar_t;
  using typename MinMaxReductionOps<comp_t>::index_t;
  using typename MinMaxReductionOps<comp_t>::arg_t;

  static C10_DEVICE index_t project(arg_t arg) {
    return arg.second;
  }
};

} // namespace detail

template <typename scalar_t>
struct ArgMaxOps :
  public detail::ArgReductionOps<detail::GreaterOrNan<scalar_t>> {
};

template <typename scalar_t>
struct ArgMinOps :
  public detail::ArgReductionOps<detail::LessOrNan<scalar_t>> {
};

template <typename scalar_t>
struct MinOps :
  public detail::MinMaxReductionOps<detail::LessOrNan<scalar_t>> {
};

template <typename scalar_t>
struct MaxOps :
  public detail::MinMaxReductionOps<detail::GreaterOrNan<scalar_t>> {
};

template <typename scalar_t, typename acc_scalar_t, typename index_t>
struct MinMaxOps {
  using acc_t = detail::pair<acc_scalar_t, acc_scalar_t>;
  inline C10_DEVICE acc_t reduce(acc_t acc, scalar_t data, index_t /*idx*/) const {
    return combine(acc, {data, data});
  }

  inline C10_DEVICE acc_t combine(acc_t a, acc_t b) const {
    auto min_val = (at::_isnan(a.first) || a.first < b.first) ? a.first : b.first;
    auto max_val = (at::_isnan(a.second) || a.second > b.second) ? a.second : b.second;

    return {min_val, max_val};
  }

  inline C10_DEVICE acc_t project(acc_t acc) const {
    return acc;
  }

  static C10_DEVICE acc_t translate_idx(acc_t acc, int64_t /*base_idx*/) {
    return acc;
  }

#if defined(__CUDACC__) || defined(__HIPCC__)
  inline C10_DEVICE acc_t warp_shfl_down(acc_t acc, int offset) const {
    return {
      WARP_SHFL_DOWN(acc.first, offset), WARP_SHFL_DOWN(acc.second, offset)
    };
  }
#endif
};

}} // namespace at::native

#undef MAX
#undef MIN
