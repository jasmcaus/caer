#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/TensorIterator.h>

#include <type_traits>

// NOTE: CUDA on Windows requires that the enclosing function
// of a __device__ lambda not have internal linkage.

namespace at { namespace native {

void remainder_kernel_cuda(TensorIterator& iter) {
  if (isIntegralType(iter.dtype(), /*includeBool*/ false)) {
    AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "remainder_cuda", [&]() {
      gpu_kernel_with_scalars(iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
        scalar_t r = a % b;
        if (!std::is_unsigned<scalar_t>::value && (r != 0) && ((r < 0) != (b < 0))) {
          r += b;
        }
        return r;
      });
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, iter.dtype(), "remainder_cuda", [&]() {
      gpu_kernel_with_scalars(iter,
        []GPU_LAMBDA(scalar_t a, scalar_t b) __ubsan_ignore_float_divide_by_zero__ -> scalar_t {
          auto mod = ::fmod(a, b);
          if (!std::is_unsigned<scalar_t>::value && (mod != 0) && ((b < 0) != (mod < 0))) {
            mod += b;
          }
          return mod;
        });
    });
  }
}

void fmod_kernel_cuda(TensorIterator& iter) {
  // Use the dtype of the first argument to retain BC,
  // change to common_dtype for type promotion in the future
  // Issue #47779: https://github.com/pytorch/pytorch/issues/47779
  if (isIntegralType(iter.dtype(), /*includeBool*/ false)) {
    AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "fmod_cuda", [&]() {
      gpu_kernel_with_scalars(iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
        return a % b;
      });
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND(kHalf, iter.dtype(), "fmod_cuda", [&]() {
      gpu_kernel_with_scalars(iter,
        []GPU_LAMBDA(scalar_t a, scalar_t b) __ubsan_ignore_float_divide_by_zero__ -> scalar_t {
          return ::fmod(a, b);
        });
    });
  }
}

REGISTER_DISPATCH(remainder_stub, &remainder_kernel_cuda);
REGISTER_DISPATCH(fmod_stub, &fmod_kernel_cuda);

}} // namespace at::native
