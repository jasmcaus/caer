#pragma once

// This file provides two functions to help write GPU elementwise kernels:
//
//   gpu_kernel(TensorIterator iter, <lambda>)
//   gpu_kernel_with_scalars(TensorIterator iter, <lambda>)
//
// The gpu_kernel_with_scalars generates specializations that support a
// single scalar CPU argument, such as from `cuda_tensor + 5`. The CPU scalar
// is lifted to a kernel parameter instead of copying to device memory.
// This should be  used in conjunction with TensorIterator::allow_cpu_scalars_,
// which is the default for TensorIterator::binary_op. Otherwise, all inputs
// and the output must be on the GPU.
//
// For example, to write a reciprocal kernel for GPU float Tensors:
//
//   gpu_kernel(iter, []GPU_LAMBDA(float a) {
//    return 1.0f / a;
//   });
//
// To write a multiplication kernel for GPU float Tensors where one argument
// may be a CPU scalar:
//
//   gpu_kernel_with_scalars(iter, []GPU_LAMBDA(float a, float b) {
//     return a * b;
//   });
//
// See BinaryOpsKernel.cu for the complete implementation
//

#include <type_traits>
#include <tuple>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/core/Array.h>
#include <ATen/detail/FunctionTraits.h>
#include <ATen/native/TensorIterator.h>
#include <c10/macros/Macros.h>
#include <c10/core/ScalarType.h>
#include <c10/util/TypeCast.h>
#include <c10/util/C++17.h>

// Marks a lambda as executable on both the host and device. The __host__
// attribute is important so that we can access static type information from
// the host, even if the function is typically only executed on the device.
#ifndef GPU_LAMBDA
#define GPU_LAMBDA __host__ __device__
#endif

#ifdef __NVCC__
#define ASSERT_HOST_DEVICE_LAMBDA(type) \
  static_assert(__nv_is_extended_host_device_lambda_closure_type(type), \
                #type " must be a __host__ __device__ lambda")
#else
#define ASSERT_HOST_DEVICE_LAMBDA(type)
#endif


namespace at { namespace native {

template<int vec_size, typename func_t, typename array_t>
C10_LAUNCH_BOUNDS_1(num_threads)
__global__ void vectorized_elementwise_kernel(int N, func_t f, array_t data) {
  using traits = function_traits<func_t>;
  int remaining = N - block_work_size * blockIdx.x;

  if (remaining < block_work_size) {  // if this block handles the reminder, just do a naive unrolled loop
    auto input_calc = TrivialOffsetCalculator<traits::arity>();
    auto output_calc = TrivialOffsetCalculator<1>();
    auto loader = memory::LoadWithoutCast();
    auto storer = memory::StoreWithoutCast();
    auto policy = memory::policies::unroll<array_t, decltype(input_calc), decltype(output_calc),
                                           memory::LoadWithoutCast, memory::StoreWithoutCast>(
      data, remaining, input_calc, output_calc, loader, storer);
    elementwise_kernel_helper(f, policy);
  } else {  // if this block has a full `block_work_size` data to handle, use vectorized memory access
    elementwise_kernel_helper(f, memory::policies::vectorized<vec_size, array_t>(data));
  }
}

template<typename func_t, typename array_t, typename inp_calc_t, typename out_calc_t, typename loader_t, typename storer_t>
C10_LAUNCH_BOUNDS_1(num_threads)
__global__ void unrolled_elementwise_kernel(int N, func_t f, array_t data,
                                            inp_calc_t ic, out_calc_t oc, loader_t l, storer_t s)
{
  int remaining = N - block_work_size * blockIdx.x;
  auto policy = memory::policies::unroll<array_t, inp_calc_t, out_calc_t, loader_t, storer_t>(data, remaining, ic, oc, l, s);
  elementwise_kernel_helper(f, policy);
}

// this function assume trivial 1d and no dynamic casting
template<typename func_t, typename array_t>
static inline void launch_vectorized_kernel(int64_t N, const func_t& f, array_t data) {
  TORCH_INTERNAL_ASSERT(N > 0 && N <= std::numeric_limits<int32_t>::max());
  using traits = function_traits<func_t>;
  int64_t grid = (N + block_work_size - 1) / block_work_size;
  auto stream = at::cuda::getCurrentCUDAStream();
  int vec_size = memory::can_vectorize_up_to<func_t>(data);

  switch (vec_size) {
  case 4:
    vectorized_elementwise_kernel<4, func_t, array_t><<<grid, num_threads, 0, stream>>>(N, f, data);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    break;
  case 2:
    vectorized_elementwise_kernel<2, func_t, array_t><<<grid, num_threads, 0, stream>>>(N, f, data);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    break;
  case 1: {
    auto input_calc = TrivialOffsetCalculator<traits::arity>();
    auto output_calc = TrivialOffsetCalculator<1>();
    auto loader = memory::LoadWithoutCast();
    auto storer = memory::StoreWithoutCast();
    unrolled_elementwise_kernel<func_t, array_t><<<grid, num_threads, 0, stream>>>(N, f, data, input_calc, output_calc, loader, storer);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    break;
  }
  default:
    TORCH_INTERNAL_ASSERT(false, "Unexpected vectorization size");
  }
}

template<typename func_t, typename array_t, typename inp_calc_t, typename out_calc_t, typename loader_t, typename storer_t>
static inline void launch_unrolled_kernel(int64_t N, const func_t& f, array_t data,
                                          inp_calc_t ic, out_calc_t oc, loader_t l, storer_t s)
{
  TORCH_INTERNAL_ASSERT(N > 0 && N <= std::numeric_limits<int32_t>::max());
  int64_t grid = (N + block_work_size - 1) / block_work_size;
  auto stream = at::cuda::getCurrentCUDAStream();
  unrolled_elementwise_kernel<func_t, array_t><<<grid, num_threads, 0, stream>>>(N, f, data, ic, oc, l, s);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename func_t>
void gpu_kernel_impl(TensorIteratorBase& iter, const func_t& f) {
  using traits = function_traits<func_t>;
  using arg0_t = typename traits::result_type;
  constexpr int ntensors = traits::arity + 1;

  TORCH_INTERNAL_ASSERT(iter.can_use_32bit_indexing());
  TORCH_INTERNAL_ASSERT(iter.ninputs() == traits::arity);
  TORCH_INTERNAL_ASSERT(iter.noutputs() == 1);

  at::detail::Array<char*, ntensors> data;
  for (int i = 0; i < ntensors; i++) {
    data[i] = (char*)iter.data_ptr(i);
  }

  int64_t numel = iter.numel();

  bool contiguous = iter.is_contiguous();
  bool dynamic_casting = needs_dynamic_casting<func_t>::check(iter);

  if (!dynamic_casting) {
    if (contiguous) {
      launch_vectorized_kernel(numel, f, data);
    } else {
      auto input_offset_calculator = make_input_offset_calculator<traits::arity>(iter);
      auto output_offset_calculator = make_output_offset_calculator(iter);
      auto loader = memory::LoadWithoutCast();
      auto storer = memory::StoreWithoutCast();
      launch_unrolled_kernel(numel, f, data, input_offset_calculator, output_offset_calculator, loader, storer);
    }
  } else {
    at::detail::Array<ScalarType, traits::arity> dtypes;
    for (int i = 0; i < traits::arity; i++) {
      dtypes[i] = iter.tensor(i + 1).scalar_type();
    }
    auto loader = memory::LoadWithCast<traits::arity>(dtypes);
    auto storer = memory::StoreWithCast(iter.tensor(0).scalar_type());
    if (contiguous) {
      auto input_offset_calculator = TrivialOffsetCalculator<traits::arity>();
      auto output_offset_calculator = TrivialOffsetCalculator<1>();
      launch_unrolled_kernel(numel, f, data, input_offset_calculator, output_offset_calculator, loader, storer);
    } else {
      auto input_offset_calculator = make_input_offset_calculator<traits::arity>(iter);
      auto output_offset_calculator = make_output_offset_calculator(iter);
      launch_unrolled_kernel(numel, f, data, input_offset_calculator, output_offset_calculator, loader, storer);
    }
  }
}

}} // namespace at::native
