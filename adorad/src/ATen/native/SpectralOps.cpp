// define constants like M_PI and C keywords for MSVC
#ifdef _MSC_VER
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <math.h>
#endif

#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <ATen/detail/CUDAHooksInterface.h>
#include <ATen/native/SpectralOpsUtils.h>
#include <ATen/native/TensorIterator.h>

#include <algorithm>
#include <vector>
#include <cmath>

namespace at { namespace native {

namespace {

// Promote inputs to FFT functions
// * Integers are promoted to the default floating type
// * If require_complex=True, all types are promoted to complex
// * Raises an error for half-precision dtypes to allow future support
ScalarType promote_type_fft(ScalarType type, bool require_complex) {
  if (at::isComplexType(type)) {
    return type;
  }
  // Promote integral to default float type
  if (!at::isFloatingType(type)) {
    type = c10::typeMetaToScalarType(c10::get_default_dtype());
  }

  TORCH_CHECK(type == kFloat || type == kDouble, "Unsupported dtype ", type);

  if (!require_complex) {
    return type;
  }

  // Promote to complex
  switch (type) {
  case kFloat: return kComplexFloat;
  case kDouble: return kComplexDouble;
  default: TORCH_INTERNAL_ASSERT(false, "Unhandled dtype");
  }
}

// Promote a tensor's dtype according to promote_type_fft
Tensor promote_tensor_fft(const Tensor& t, bool require_complex=false) {
  auto cur_type = t.scalar_type();
  auto new_type = promote_type_fft(cur_type, require_complex);
  return (cur_type == new_type) ? t : t.to(new_type);
}

// Convert NumPy compatible normalization mode string to enum values
// NOTE: NumPy's normalization modes have direction-specific meanings. For example,
// "forward" translates to `by_n` for a forward transform and `none` for backward.
fft_norm_mode norm_from_string(c10::optional<std::string> norm, bool forward) {
  if (!norm || *norm == "backward") {
    return forward ? fft_norm_mode::none : fft_norm_mode::by_n;
  }

  if (*norm == "forward") {
    return forward ? fft_norm_mode::by_n : fft_norm_mode::none;
  }

  if (*norm == "ortho") {
    return fft_norm_mode::by_root_n;
  }

  TORCH_CHECK(false, "Invalid normalization mode: \"", *norm, "\"")
}

// Fixes the shape of x such that x.size(dims[i]) == sizes[i],
// either by zero-padding, or by slicing x starting from 0.
Tensor resize_fft_input(Tensor x, IntArrayRef dims, IntArrayRef sizes) {
  TORCH_INTERNAL_ASSERT(dims.size() == sizes.size());
  bool must_copy = false;
  auto x_sizes = x.sizes();
  DimVector pad_amount(x_sizes.size() * 2);
  for (int64_t i = 0; i < dims.size(); ++i) {
    if (sizes[i] == -1) {
      continue;
    }

    if (x_sizes[dims[i]] < sizes[i]) {
      must_copy = true;
      auto pad_idx = pad_amount.size() - 2 * dims[i] - 1;
      pad_amount[pad_idx] = sizes[i] - x_sizes[dims[i]];
    }

    if (x_sizes[dims[i]] > sizes[i]) {
      x = x.slice(dims[i], 0, sizes[i]);
    }
  }

  // Only call pad if necessary since pad copies the entire tensor
  return must_copy ? at::constant_pad_nd(x, pad_amount) : x;
}

// Complex to real FFT
Tensor fft_c2r(Tensor input, c10::optional<int64_t> n_opt,
               int64_t unwrapped_dim, c10::optional<std::string> norm_str,
               bool forward) {
  input = promote_tensor_fft(input, /*require_complex=*/true);
  const auto input_dim = input.dim();
  const auto dim = maybe_wrap_dim(unwrapped_dim, input_dim);
  const auto n = n_opt.value_or(2*(input.sizes()[dim] - 1));
  TORCH_CHECK(n >= 1, "Invalid number of data points (", n, ") specified");
  if (n_opt) {
    input = resize_fft_input(input, dim, n/2 + 1);
  }
  const auto norm = norm_from_string(norm_str, forward);
  if (forward) {
    // FIXME: _fft does not support complex_output=false with inverse=false
    input = at::conj(input);
  }
  return at::_fft_c2r(input, dim, static_cast<int64_t>(norm), n);
}

// Real to complex FFT
Tensor fft_r2c(Tensor input, c10::optional<int64_t> n_opt,
               int64_t unwrapped_dim, c10::optional<std::string> norm_str,
               bool forward, bool onesided) {
  TORCH_CHECK(!input.is_complex(), "Expected a real input tensor to FFT");
  input = promote_tensor_fft(input);
  const auto input_dim = input.dim();
  const auto dim = maybe_wrap_dim(unwrapped_dim, input_dim);
  const auto n = n_opt.value_or(input.sizes()[dim]);
  TORCH_CHECK(n >= 1, "Invalid number of data points (", n, ") specified");
  if (n_opt) {
    input = resize_fft_input(input, dim, n);
  }

  const auto norm = norm_from_string(norm_str, forward);
  auto out = at::_fft_r2c(input, dim, static_cast<int64_t>(norm), onesided);
  if (!forward) {
    // FIXME: _fft_r2c doesn't support native r2c IFFT
    out = at::conj(out);
  }
  return out;
}

// Complex to complex FFT
Tensor fft_c2c(Tensor input, c10::optional<int64_t> n_opt,
               int64_t unwrapped_dim, c10::optional<std::string> norm_str,
               bool forward) {
  TORCH_CHECK(input.is_complex(), "Expected a complex input tensor to FFT");
  const auto input_dim = input.dim();
  const auto dim = maybe_wrap_dim(unwrapped_dim, input_dim);
  const auto n = n_opt.value_or(input.sizes()[dim]);
  TORCH_CHECK(n >= 1, "Invalid number of data points (", n, ") specified");
  if (n_opt) {
    input = resize_fft_input(input, dim, n);
  }
  const auto norm = norm_from_string(norm_str, forward);
  return at::_fft_c2c(input, dim, static_cast<int64_t>(norm), forward);
}

// Dimensions to transform, and the signal shape in those dimensions
struct ShapeAndDims {
  DimVector shape, dim;
};

// Pre-process n-dimensional fft's `s` and `dim` arguments.
// Wraps dimensions and applies defaulting behavior.
// Also checks transform dims are unique and transform shape is non-empty.
ShapeAndDims canonicalize_fft_shape_and_dim_args(
    Tensor input, c10::optional<IntArrayRef> shape, c10::optional<IntArrayRef> dim) {
  const int64_t input_dim = input.dim();
  const IntArrayRef input_sizes = input.sizes();
  ShapeAndDims ret;

  if (dim) {
    ret.dim.resize(dim->size());
    std::copy(dim->begin(), dim->end(), ret.dim.begin());
    maybe_wrap_dims(ret.dim, input_dim);

    // Check dims are unique
    DimVector copy = ret.dim;
    std::sort(copy.begin(), copy.end());
    auto duplicate = std::adjacent_find(copy.begin(), copy.end());
    TORCH_CHECK(duplicate == copy.end(), "FFT dims must be unique");
  }

  if (shape) {
    // Has shape, may have dim
    TORCH_CHECK(!dim || dim->size() == shape->size(),
                "When given, dim and shape arguments must have the same length");
    TORCH_CHECK(shape->size() <= input_dim,
                "Got shape with ", shape->size(), " values but input tensor "
                "only has ", input_dim, " dimensions.");
    const int64_t transform_ndim = shape->size();
    // If shape is given, dims defaults to the last shape.size() dimensions
    if (!dim) {
      ret.dim.resize(transform_ndim);
      std::iota(ret.dim.begin(), ret.dim.end(), input_dim - transform_ndim);
    }

    // Translate shape of -1 to the default length
    ret.shape.resize(transform_ndim);
    for (int64_t i = 0; i < transform_ndim; ++i) {
      const auto n = (*shape)[i];
      ret.shape[i] = n == -1 ? input_sizes[ret.dim[i]] : n;
    }
  } else if (!dim) {
    // No shape, no dim
    ret.dim.resize(input_dim);
    std::iota(ret.dim.begin(), ret.dim.end(), int64_t{0});
    ret.shape.resize(input_dim);
    std::copy(input_sizes.begin(), input_sizes.end(), ret.shape.begin());
  } else {
    // No shape, has dim
    ret.shape.resize(ret.dim.size());
    for (int64_t i = 0; i < ret.dim.size(); ++i) {
      ret.shape[i] = input_sizes[ret.dim[i]];
    }
  }

  for (int64_t i = 0; i < ret.shape.size(); ++i) {
    TORCH_CHECK(ret.shape[i] > 0,
                "Invalid number of data points (", ret.shape[i], ") specified");
  }

  return ret;
}

// Complex to complex n-dimensional fft
Tensor fftn_c2c(
    const Tensor& input, IntArrayRef shape, IntArrayRef dim,
    c10::optional<std::string> norm_str, bool forward) {
  TORCH_CHECK(input.is_complex(), "Expected a complex input tensor to FFT");
  Tensor x = resize_fft_input(input, dim, shape);
  const auto norm = norm_from_string(norm_str, forward);
  return at::_fft_c2c(x, dim, static_cast<int64_t>(norm), forward);
}

}  // namespace (anonymous)

// torch.fft.fft, analogous to NumPy's numpy.fft.fft
Tensor fft_fft(const Tensor& self, c10::optional<int64_t> n, int64_t dim,
               c10::optional<std::string> norm) {
  return self.is_complex() ?
    fft_c2c(self, n, dim, norm, /*forward=*/true) :
    fft_r2c(self, n, dim, norm, /*forward=*/true, /*onesided=*/false);
}

Tensor fft_ifft(const Tensor& self, c10::optional<int64_t> n, int64_t dim,
                c10::optional<std::string> norm) {
  return self.is_complex() ?
    fft_c2c(self, n, dim, norm, /*forward=*/false) :
    fft_r2c(self, n, dim, norm, /*forward=*/false, /*onesided=*/false);
}

Tensor fft_rfft(const Tensor& self, c10::optional<int64_t> n, int64_t dim,
                c10::optional<std::string> norm) {
  return fft_r2c(self, n, dim, norm, /*forward=*/true, /*onesided=*/true);
}

Tensor fft_irfft(const Tensor& self, c10::optional<int64_t> n, int64_t dim,
                 c10::optional<std::string> norm) {
  return fft_c2r(self, n, dim, norm, /*forward=*/false);
}

Tensor fft_hfft(const Tensor& self, c10::optional<int64_t> n, int64_t dim,
                c10::optional<std::string> norm) {
  return fft_c2r(self, n, dim, norm, /*forward=*/true);
}

Tensor fft_ihfft(const Tensor& self, c10::optional<int64_t> n, int64_t dim,
                 c10::optional<std::string> norm) {
  return fft_r2c(self, n, dim, norm, /*forward=*/false, /*onesided=*/true);
}

Tensor fft_fftn(const Tensor& self, c10::optional<IntArrayRef> s,
                c10::optional<IntArrayRef> dim,
                c10::optional<std::string> norm) {
  auto desc = canonicalize_fft_shape_and_dim_args(self, s, dim);
  // TODO: For real input, perform rfftn then mirror with conjugate symmetry
  Tensor input = promote_tensor_fft(self, /*require_complex=*/true);
  return fftn_c2c(input, desc.shape, desc.dim, norm, /*forward=*/true);
}

Tensor fft_ifftn(const Tensor& self, c10::optional<IntArrayRef> s,
                c10::optional<IntArrayRef> dim,
                c10::optional<std::string> norm) {
  auto desc = canonicalize_fft_shape_and_dim_args(self, s, dim);
  Tensor input = promote_tensor_fft(self, /*require_complex=*/true);
  return fftn_c2c(input, desc.shape, desc.dim, norm, /*forward=*/false);
}

Tensor fft_rfftn(const Tensor& self, c10::optional<IntArrayRef> s,
                c10::optional<IntArrayRef> dim,
                c10::optional<std::string> norm_str) {
  TORCH_CHECK(!self.is_complex(), "rfftn expects a real-valued input tensor, but got ", self.scalar_type());
  auto desc = canonicalize_fft_shape_and_dim_args(self, s, dim);
  TORCH_CHECK(desc.shape.size() > 0, "rfftn must transform at least one axis");
  Tensor input = promote_tensor_fft(self, /*require_complex=*/false);
  Tensor x = resize_fft_input(input, desc.dim, desc.shape);
  const auto norm = norm_from_string(norm_str, /*forward=*/true);
  return at::_fft_r2c(x, desc.dim, static_cast<int64_t>(norm), /*onesided=*/true);
}

Tensor fft_irfftn(const Tensor& self, c10::optional<IntArrayRef> s,
                c10::optional<IntArrayRef> dim,
                c10::optional<std::string> norm_str) {
  auto desc = canonicalize_fft_shape_and_dim_args(self, s, dim);
  TORCH_CHECK(desc.shape.size() > 0, "irfftn must transform at least one axis");

  const auto last_dim_size = [&] {
    // Fixup default shape handling in the last dimension,
    if (!s.has_value() || (s->back() == -1)) {
      const auto last_dim = desc.dim.back();
      return 2 * (self.sizes()[last_dim] - 1);
    }
    return desc.shape.back();
  }();
  desc.shape.back() = last_dim_size / 2 + 1;

  Tensor input = promote_tensor_fft(self, /*require_complex=*/true);
  Tensor x = resize_fft_input(input, desc.dim, desc.shape);
  const auto norm = norm_from_string(norm_str, /*forward=*/false);
  return at::_fft_c2r(x, desc.dim, static_cast<int64_t>(norm), last_dim_size);
}

Tensor fft_fft2(const Tensor& self, c10::optional<IntArrayRef> s,
                IntArrayRef dim, c10::optional<std::string> norm) {
  return native::fft_fftn(self, s, dim, std::move(norm));
}

Tensor fft_ifft2(const Tensor& self, c10::optional<IntArrayRef> s,
                IntArrayRef dim, c10::optional<std::string> norm) {
  return native::fft_ifftn(self, s, dim, std::move(norm));
}

Tensor fft_rfft2(const Tensor& self, c10::optional<IntArrayRef> s,
                IntArrayRef dim, c10::optional<std::string> norm) {
  return native::fft_rfftn(self, s, dim, std::move(norm));
}

Tensor fft_irfft2(const Tensor& self, c10::optional<IntArrayRef> s,
                  IntArrayRef dim, c10::optional<std::string> norm) {
  return native::fft_irfftn(self, s, dim, std::move(norm));
}

Tensor fft_fftfreq(int64_t n, double d, const TensorOptions& options) {
  ScalarType dtype = typeMetaToScalarType(options.dtype());
  TORCH_CHECK(at::isFloatingType(dtype) || at::isComplexType(dtype),
              "fftfreq requires a floating point or complex dtype");
  // TODO: arange doesn't have complex support
  Tensor result = native::arange(n, options);
  auto right_slice = result.slice(0, (n + 1) / 2, 0);
  at::arange_out(right_slice, -(n/2), 0, 1);
  result.mul_(1.0 / (n * d));  // Slightly faster than div_(n*d)
  return result;
}

Tensor fft_rfftfreq(int64_t n, double d, const TensorOptions& options) {
  ScalarType dtype = typeMetaToScalarType(options.dtype());
  TORCH_CHECK(at::isFloatingType(dtype) || at::isComplexType(dtype),
              "rfftfreq requires a floating point or complex dtype");
  // TODO: arange doesn't have complex support
  Tensor result = native::arange(n/2 + 1, options);
  result.mul_(1.0 / (n * d));  // Slightly faster than div_(n*d)
  return result;
}

// If an array dim is specified, wraps them according to self.dim().
// Otherwise returns a vector of all dims.
DimVector default_alldims(const Tensor& self, c10::optional<IntArrayRef> dim_opt) {
  DimVector dim;
  if (dim_opt) {
    IntArrayRef dim_unwrapped = *dim_opt;
    dim.resize(dim_unwrapped.size());
    for (int64_t i = 0; i < dim.size(); ++i) {
      dim[i] = maybe_wrap_dim(dim_unwrapped[i], self.dim());
    }
  } else {
    dim.resize(self.dim());
    std::iota(dim.begin(), dim.end(), 0);
  }
  return dim;
}

Tensor fft_fftshift(const Tensor& x, c10::optional<IntArrayRef> dim_opt) {
  auto dim = default_alldims(x, dim_opt);

  IntArrayRef x_sizes = x.sizes();
  DimVector shift(dim.size());
  for (int64_t i = 0; i < dim.size(); ++i) {
    shift[i] = x_sizes[dim[i]] / 2;
  }

  return at::roll(x, shift, dim);
}

Tensor fft_ifftshift(const Tensor& x, c10::optional<IntArrayRef> dim_opt) {
  auto dim = default_alldims(x, dim_opt);

  IntArrayRef x_sizes = x.sizes();
  DimVector shift(dim.size());
  for (int64_t i = 0; i < dim.size(); ++i) {
    shift[i] = (x_sizes[dim[i]] + 1) / 2;
  }

  return at::roll(x, shift, dim);
}


// We call the following methods via CUDA hooks because they are really only
// valid when CUDA is available. See native/cuda/CuFFTPlanCache.h for more details.
int64_t _cufft_get_plan_cache_max_size(int64_t device_index) {
  return detail::getCUDAHooks().cuFFTGetPlanCacheMaxSize(device_index);
}

void _cufft_set_plan_cache_max_size(int64_t device_index, int64_t max_size) {
  detail::getCUDAHooks().cuFFTSetPlanCacheMaxSize(device_index, max_size);
}

int64_t _cufft_get_plan_cache_size(int64_t device_index) {
  return detail::getCUDAHooks().cuFFTGetPlanCacheSize(device_index);
}

void _cufft_clear_plan_cache(int64_t device_index) {
  detail::getCUDAHooks().cuFFTClearPlanCache(device_index);
}

template <typename Stream, typename T>
static Stream& write_opt(Stream& SS, const optional<T>& value) {
  if (value) {
    SS << *value;
  } else {
    SS << "None";
  }
  return SS;
}

/* Short-time Fourier Transform, for signal analysis.
 *
 * This is modeled after librosa but with support for complex time-domain
 * signals and complex windows.
 *
 * NOTE: librosa's center and pad_mode arguments are currently only implemented
 * in python because it uses torch.nn.functional.pad which is python-only.
 */
Tensor stft(const Tensor& self, const int64_t n_fft, const optional<int64_t> hop_lengthOpt,
            const optional<int64_t> win_lengthOpt, const Tensor& window,
            const bool normalized, const optional<bool> onesidedOpt,
            const optional<bool> return_complexOpt) {
  #define REPR(SS) \
    SS << "stft(" << self.toString() << self.sizes() << ", n_fft=" << n_fft \
       << ", hop_length=" << hop_length << ", win_length=" << win_length \
       << ", window="; \
    if (window.defined()) { \
      SS << window.toString() << "{" << window.sizes() << "}"; \
    } else { \
      SS << "None"; \
    } \
    SS << ", normalized=" << normalized << ", onesided="; \
    write_opt(SS, onesidedOpt) << ", return_complex="; \
    write_opt(SS, return_complexOpt) << ") "

  // default_init hop_length and win_length
  auto hop_length = hop_lengthOpt.value_or(n_fft >> 2);
  auto win_length = win_lengthOpt.value_or(n_fft);
  const bool return_complex = return_complexOpt.value_or(
      self.is_complex() || (window.defined() && window.is_complex()));
  if (!return_complexOpt && !return_complex) {
    TORCH_WARN_ONCE("stft will require the return_complex parameter be explicitly "
                    " specified in a future PyTorch release. Use return_complex=False "
                    " to preserve the current behavior or return_complex=True to return "
                    " a complex output.");
  }

  if (!at::isFloatingType(self.scalar_type()) && !at::isComplexType(self.scalar_type())) {
    std::ostringstream ss;
    REPR(ss) << ": expected a tensor of floating point or complex values";
    AT_ERROR(ss.str());
  }
  if (self.dim() > 2 || self.dim() < 1) {
    std::ostringstream ss;
    REPR(ss) << ": expected a 1D or 2D tensor";
    AT_ERROR(ss.str());
  }
  Tensor input = self;
  if (self.dim() == 1) {
    input = input.unsqueeze(0);
  }
  int64_t batch = input.size(0);
  int64_t len = input.size(1);
  if (n_fft <= 0 || n_fft > len) {
    std::ostringstream ss;
    REPR(ss) << ": expected 0 < n_fft < " << len
             << ", but got n_fft=" << win_length;
    AT_ERROR(ss.str());
  }
  if (hop_length <= 0) {
    std::ostringstream ss;
    REPR(ss) << ": expected hop_length > 0, but got hop_length=" << hop_length;
    AT_ERROR(ss.str());
  }
  if (win_length <= 0 || win_length > n_fft) {
    std::ostringstream ss;
    REPR(ss) << ": expected 0 < win_length <= n_fft, but got win_length="
             << win_length;
    AT_ERROR(ss.str());
  }
  if (window.defined() && (window.dim() != 1 || window.size(0) != win_length)) {
    std::ostringstream ss;
    REPR(ss) << ": expected a 1D window tensor of size equal to win_length="
             << win_length << ", but got window with size " << window.sizes();
    AT_ERROR(ss.str());
  }
  #undef REPR
  auto window_ = window;
  if (win_length < n_fft) {
    // pad center
    auto left = (n_fft - win_length) / 2;
    if (window.defined()) {
      window_ = at::zeros({n_fft}, window.options());
      window_.narrow(0, left, win_length).copy_(window);
    } else {
      window_ = at::zeros({n_fft}, self.options());
      window_.narrow(0, left, win_length).fill_(1);
    }
  }
  int64_t n_frames = 1 + (len - n_fft) / hop_length;
  // time2col
  input = input.as_strided(
    {batch, n_frames, n_fft},
    {input.stride(0), hop_length * input.stride(1), input.stride(1)}
  );
  if (window_.defined()) {
    input = input.mul(window_);
  }

  // FFT and transpose to get (batch x fft_size x num_frames)
  const bool complex_fft = input.is_complex();
  const auto onesided = onesidedOpt.value_or(!complex_fft);

  const fft_norm_mode norm = normalized ? fft_norm_mode::by_root_n : fft_norm_mode::none;
  Tensor out;
  if (complex_fft) {
    TORCH_CHECK(!onesided, "Cannot have onesided output if window or input is complex");
    out = at::_fft_c2c(input, input.dim() - 1, static_cast<int64_t>(norm), /*forward=*/true);
  } else {
    out = at::_fft_r2c(input, input.dim() - 1, static_cast<int64_t>(norm), onesided);
  }
  out.transpose_(1, 2);

  if (self.dim() == 1) {
    out.squeeze_(0);
  }

  if (return_complex) {
    return out;
  } else {
    return at::view_as_real(out);
  }
}

// Create complex tensor from the old style of real tensor with size=(..., 2)
// This is to support istft in the transition to requiring complex input.
// NOTE: This may return a view of the input tensor, or might clone if necessary
static Tensor as_complex(const Tensor& self) {
  const bool can_view_as_complex = [&]{
    auto strides = self.strides();
    for (int64_t i = 0; i + 1 < strides.size(); ++i) {
      if (strides[i] % 2 != 0) {
        return false;
      }
    }
    return strides.back() == 1 && self.storage_offset() % 2 == 0;
  }();
  return at::view_as_complex(can_view_as_complex ? self : self.clone(MemoryFormat::Contiguous));
}

/* Inverse Short-time Fourier Transform
 *
 * This is modeled after librosa but with support for complex time-domain
 * signals and complex windows.
 */
Tensor istft(const Tensor& self, const int64_t n_fft, const optional<int64_t> hop_lengthOpt,
             const optional<int64_t> win_lengthOpt, const Tensor& window,
             const bool center, const bool normalized, const c10::optional<bool> onesidedOpt,
             const optional<int64_t> lengthOpt, const bool return_complex) {
  #define REPR(SS) \
    SS << "istft(" << self.toString() << self.sizes() << ", n_fft=" << n_fft \
       << ", hop_length=" << hop_length << ", win_length=" << win_length \
       << ", window="; \
    if (window.defined()) { \
      SS << window.toString() << "{" << window.sizes() << "}"; \
    } else { \
      SS << "None"; \
    } \
    SS << ", center=" << center << ", normalized=" << normalized << ", onesided="; \
    write_opt(SS, onesidedOpt) << ", length="; \
    write_opt(SS, lengthOpt) << ", return_complex=" << return_complex << ") "

  // default_init hop_length and win_length
  const auto hop_length = hop_lengthOpt.value_or(n_fft >> 2);
  const auto win_length = win_lengthOpt.value_or(n_fft);

  if (!self.is_complex()) {
    TORCH_WARN_ONCE(
      "istft will require a complex-valued input tensor in a future PyTorch release. "
      "Matching the output from stft with return_complex=True. ");
  }
  Tensor input = self.is_complex() ? at::view_as_real(self) : self;
  const auto input_dim = input.dim();
  const auto n_frames = input.size(-2);
  const auto fft_size = input.size(-3);

  const auto expected_output_signal_len = n_fft + hop_length * (n_frames - 1);

  const auto options = at::device(input.device()).dtype(input.dtype());
  if (input.numel() == 0) {
    std::ostringstream ss;
    REPR(ss) << ": input tensor cannot be empty.";
    AT_ERROR(ss.str());
  }
  if (input_dim != 3 && input_dim != 4) {
    std::ostringstream ss;
    REPR(ss) << ": expected a tensor with 3 or 4 dimensions, but got " << input_dim;
    AT_ERROR(ss.str());
  }
  if (input.size(-1) != 2) {
    std::ostringstream ss;
    REPR(ss) << ": expected the last dimension to be 2 (corresponding to real and imaginary parts), but got " << self.size(-1);
    AT_ERROR(ss.str());
  }

  const bool onesided = onesidedOpt.value_or(fft_size != n_fft);
  if (onesided) {
    if (n_fft / 2 + 1 != fft_size) {
      std::ostringstream ss;
      REPR(ss) << ": expected the frequency dimension (3rd to the last) of the input tensor to match n_fft / 2 + 1 when onsided=True, but got " << fft_size;
      AT_ERROR(ss.str());
    }
  } else {
    if (n_fft != fft_size) {
      std::ostringstream ss;
      REPR(ss) << ": expected the frequency dimension (3rd to the last) of the input tensor to match n_fft when onsided=False, but got " << fft_size;
      AT_ERROR(ss.str());
    }
  }

  if (!(0 < hop_length && hop_length <= win_length)) {
    std::ostringstream ss;
    REPR(ss) << ": expected 0 < hop_length <= win_length";
    AT_ERROR(ss.str());
  }

  if (!(0 < win_length && win_length <= n_fft)) {
    std::ostringstream ss;
    REPR(ss) << ": expected 0 < win_length <= n_fft";
    AT_ERROR(ss.str());
  }
  if (window.defined()) {
    if (window.dim() != 1 || window.size(0) != win_length) {
      std::ostringstream ss;
      REPR(ss) << ": Invalid window shape. window has to be 1D and length of `win_length`";
      AT_ERROR(ss.str());
    }
  }

  Tensor window_tmp = window.defined() ? window : at::ones({win_length,}, options);
  if (win_length != n_fft) {
    // center window by padding zeros on right and left side
    int64_t left = (n_fft - win_length) / 2;
    window_tmp = at::constant_pad_nd(window_tmp, {left, n_fft - win_length - left}, 0);
    TORCH_INTERNAL_ASSERT(window_tmp.size(0) == n_fft);
  }

  if (input_dim == 3) {
    input = input.unsqueeze(0);
  }

  input = as_complex(input.transpose(1, 2));  // size: (channel, n_frames, fft_size, 2)

  const fft_norm_mode norm = normalized ? fft_norm_mode::by_root_n : fft_norm_mode::by_n;
  if (return_complex) {
    TORCH_CHECK(!onesided, "Cannot have onesided output if window or input is complex");
    input = at::_fft_c2c(input, input.dim() - 1, static_cast<int64_t>(norm), /*forward=*/false);  // size: (channel, n_frames, n_fft)
  } else {
    TORCH_CHECK(!window.defined() || !window.is_complex(),
                "Complex windows are incompatible with return_complex=False");
    if (!onesided) {
      input = input.slice(-1, 0, n_fft / 2 + 1);
    }
    input = at::_fft_c2r(input, input.dim() - 1, static_cast<int64_t>(norm), n_fft);  // size: (channel, n_frames, n_fft)
  }
  TORCH_INTERNAL_ASSERT(input.size(2) == n_fft);

  Tensor y_tmp = input * window_tmp.view({1, 1, n_fft});  // size: (channel, n_frames, n_fft)
  y_tmp = y_tmp.transpose(1, 2);  // size: (channel, n_fft, frame)

  Tensor y = at::col2im(y_tmp,
                                  /*output_size*/ {1, (n_frames - 1) * hop_length + n_fft},
                                  /*kernel_size*/ {1, n_fft},
                                  /*dilation*/    {1, 1},
                                  /*padding*/     {0, 0},
                                  /*stride*/      {1, hop_length}
                                 ).squeeze(2);
  window_tmp = window_tmp.pow(2).view({n_fft, 1}).repeat({1, n_frames}).unsqueeze(0);  // size: (1, n_fft, n_frames)
  Tensor window_envelop = at::col2im(window_tmp,
                                  /*output_size*/ {1, (n_frames - 1) * hop_length + n_fft},
                                  /*kernel_size*/ {1, n_fft},
                                  /*dilation*/    {1, 1},
                                  /*padding*/     {0, 0},
                                  /*stride*/      {1, hop_length}
                                 ).squeeze(2); // size: (1, 1, expected_output_signal_len)

  TORCH_INTERNAL_ASSERT(expected_output_signal_len == y.size(2));
  TORCH_INTERNAL_ASSERT(expected_output_signal_len == window_envelop.size(2));

  // We need to trim the front padding away if centered
  const auto start = center ? n_fft / 2 : 0;
  const auto end = lengthOpt.has_value()? start + lengthOpt.value() : - n_fft / 2;

  y = y.slice(2, start, end, 1);
  window_envelop = window_envelop.slice(2, start, end, 1);
  const auto window_envelop_lowest = window_envelop.abs().min().item().toDouble();
  if (window_envelop_lowest < 1e-11) {
    std::ostringstream ss;
    REPR(ss) << "window overlap add min: " << window_envelop_lowest;
    AT_ERROR(ss.str());
  }

  y = (y / window_envelop).squeeze(1);  // size: (channel, expected_output_signal_len)
  if (input_dim == 3) {
    y = y.squeeze(0);
  }
  return y;

  #undef REPR
}

Tensor stft(const Tensor& self, const int64_t n_fft, const optional<int64_t> hop_lengthOpt,
            const optional<int64_t> win_lengthOpt, const Tensor& window,
            const bool normalized, const optional<bool> onesidedOpt) {
  return at::native::stft(
      self, n_fft, hop_lengthOpt, win_lengthOpt, window, normalized, onesidedOpt,
      /*return_complex=*/c10::nullopt);
}

Tensor istft(const Tensor& self, const int64_t n_fft, const optional<int64_t> hop_lengthOpt,
             const optional<int64_t> win_lengthOpt, const Tensor& window,
             const bool center, const bool normalized, const optional<bool> onesidedOpt,
             const optional<int64_t> lengthOpt) {
  return at::native::istft(
      self, n_fft, hop_lengthOpt, win_lengthOpt, window, center, normalized,
      onesidedOpt, lengthOpt, /*return_complex=*/false);
}

void _fft_fill_with_conjugate_symmetry_(const Tensor& input, IntArrayRef dim_) {
  const auto input_sizes = input.sizes();
  const auto input_strides = input.strides();
  TORCH_CHECK(dim_.size() > 0);
  DimVector dim(dim_.begin(), dim_.end());
  at::maybe_wrap_dims(dim, input_strides.size());

  if (input.numel() == 0 || input_sizes[dim.back()] <= 2) {
    return;  // No elements need writing
  }

  // Small dimensions may be treated as batch dims since they don't get mirrored
  dim.erase(
      std::remove_if(dim.begin(), dim.end(), [&](int64_t dim) {
        return (input_sizes[dim] <= 2);
      }),
      dim.end());

  // Use TensorIterator to coalesce batch dimensions
  // NOTE: Can't use TensorIterator loops because we need negative strides
  auto iter = TensorIteratorConfig()
      .add_output(input)
      .add_input(input)
      .resize_outputs(false)
      .declare_static_shape(input_sizes, dim)
      .build();

  const auto iter_strides = iter.strides(0);
  const auto iter_sizes = iter.shape();
  const auto ndim = iter_strides.size() + dim.size();
  DimVector in_strides(ndim), signal_half_sizes(ndim);
  // Take coalesced batch dimensions from TensorIterator
  std::copy(iter_strides.begin(), iter_strides.end(), in_strides.begin());
  std::copy(iter_sizes.begin(), iter_sizes.end(), signal_half_sizes.begin());

  // Take transformed dimensions directly from the input
  const auto element_size = iter.element_size(0);
  for (int64_t i = 0; i < dim.size(); ++i) {
    // Convert to byte strides to match TensorIterator
    in_strides[iter_strides.size() + i] = input_strides[dim[i]] * element_size;
    signal_half_sizes[iter_strides.size() + i] = input_sizes[dim[i]];
  }

  // For the last dimension, use negative strides to perform the mirroring
  signal_half_sizes.back() = (input_sizes[dim.back()] - 1) / 2;
  auto out_strides = in_strides;
  out_strides.back() *= -1;

  auto* data_ptr = static_cast<char*>(input.data_ptr());
  const auto* in_data = data_ptr + input_strides[dim.back()] * element_size;
  auto* out_data = data_ptr + (
      input_strides[dim.back()] * (input_sizes[dim.back()] - 1) * element_size);

  // Reorder dimensions by stride to maximize data locality
  DimVector dim_permute(ndim);
  std::iota(dim_permute.begin(), dim_permute.end(), 0);
  std::sort(dim_permute.begin(), dim_permute.end(),
      [&](auto dim1, auto dim2) {
        return in_strides[dim1] < in_strides[dim2];
      });

  DimVector temp(ndim);
  auto apply_permutation = [&] (DimVector & vec) {
    // Do permuted index copy into a temporary, then copy back
    for (int64_t i = 0; i < ndim; ++i) {
      temp[i] = vec[dim_permute[i]];
    }
    vec = temp;
  };
  apply_permutation(in_strides);
  apply_permutation(out_strides);
  apply_permutation(signal_half_sizes);

  // Find dims.slice(dims.size() - 1) in the new permuted order.
  // These are the dimensions that need explicit Hermitian mirroring
  DimVector mirror_dims;
  mirror_dims.reserve(dim.size() - 1);
  for (int64_t i = 0; i < ndim; ++i) {
    if (dim_permute[i] >= iter_strides.size() &&  // Not a batch dimension
        dim_permute[i] != ndim - 1) {  // Not the last dim, which is mirrored separately with negative strides
      mirror_dims.push_back(i);
    }
  }
  TORCH_INTERNAL_ASSERT(mirror_dims.size() == dim.size() - 1);

  // Dispatch to CPU or CUDA kernel to do the actual conjugate mirroring
  fft_fill_with_conjugate_symmetry_stub(
      input.device().type(), input.scalar_type(),
      mirror_dims, signal_half_sizes, in_strides, in_data, out_strides, out_data);
}

DEFINE_DISPATCH(fft_fill_with_conjugate_symmetry_stub);

}} // at::native
