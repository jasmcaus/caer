#include "caffe2/operators/utility_ops.h"

#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/unique.h>

#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/flatten_op.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <>
bool WeightedSumOp<CUDAContext>::RunOnDevice() {
  if (Input(0).IsType<float>()) {
    return DoRunWithType<float>();
  } else if (Input(0).IsType<at::Half>()) {
    return DoRunWithType<at::Half>();
  } else {
    CAFFE_THROW("Unsupported inputs");
  }
  return false;
}

template <>
bool SumOp<CUDAContext>::RunOnDevice() {
  return DispatchHelper<TensorTypes<float, at::Half, int32_t, int64_t>>::call(
      this, Input(0));
}

REGISTER_CUDA_OPERATOR(Print, PrintOp<CUDAContext>);
REGISTER_CUDA_OPERATOR(Flatten, FlattenOp<CUDAContext>);
REGISTER_CUDA_OPERATOR(FlattenToVec, FlattenToVecOp<CUDAContext>);
REGISTER_CUDA_OPERATOR(Alias, AliasOp<CUDAContext>);
REGISTER_CUDA_OPERATOR(ResizeLike, ResizeLikeOp<CUDAContext>);
REGISTER_CUDA_OPERATOR(Sum, SumOp<CUDAContext>);
REGISTER_CUDA_OPERATOR(WeightedSum, WeightedSumOp<CUDAContext>);

REGISTER_CUDA_OPERATOR(EnsureDense, EnsureDenseOp<CUDAContext>);

__global__ void NanCheckKernel(int N, const float* X, bool* result) {
  bool has_nan = false;
  CUDA_1D_KERNEL_LOOP(i, N) {
    // Note: we have no need to do early return, since only if this fails
    // will we not need to inspect all elements. No need to optimize the
    // case that will fail.
    has_nan = has_nan || isnan(X[i]) || isinf(X[i]);
  }
  __syncthreads();
  if (has_nan) {
    result[0] = true;
  }
}

template <>
bool NanCheckOp<CUDAContext>::RunOnDevice() {
  auto& X = Input(0);
  auto* Y = Output(0);
  const size_t N = X.numel();
  const float* data_ptr = X.data<float>();

  ReinitializeTensor(&scratch_, {1}, at::dtype<bool>().device(CUDA));
  math::Set<bool, CUDAContext>(
      1, false, scratch_.mutable_data<bool>(), &context_);
  NanCheckKernel<<<
      CAFFE_GET_BLOCKS(N),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      N, X.data<float>(), scratch_.mutable_data<bool>());

  bool result = false;
  {
    std::lock_guard<std::mutex> lock(CUDAContext::mutex());
    CUDA_ENFORCE(cudaMemcpyAsync(
        &result,
        scratch_.raw_data(),
        1,
        cudaMemcpyDefault,
        context_.cuda_stream()));
  }
  // Note: we must synchronize here so we can inspect the result
  context_.FinishDeviceComputation();

  // Print out diagnostic info if we have a NaN or inf
  if (result) {
    std::cerr << "Tensor contained NaN or inf: " << this->debug_def().input(0)
              << std::endl;

    for (int j = 0; j < InputSize(); j++) {
      Tensor cpu_X(CPU);
      cpu_X.ResizeLike(Input(j));
      // Hack to cause allocation happen here, so it won't happen
      // when we do CopyFrom. We need the mutex then because host->gpu
      // copies seem to possibly lock with NCCL.
      cpu_X.mutable_data<float>();

      {
        std::lock_guard<std::mutex> lock(CUDAContext::mutex());
        cpu_X.CopyFrom(Input(j)); // sync copy
      }
      std::cerr << "Input tensor: " << j << ": [" << this->debug_def().input(j)
                << "]" << std::endl;
      tensorPrinter_.Print<float>(cpu_X);

      if (j == 0) {
        std::cerr << "NaN idxs:" << std::endl;
        auto* cpu_X_data = cpu_X.data<float>();
        for (size_t i = 0; i < cpu_X.numel(); ++i) {
          if (std::isnan(cpu_X_data[i]) || std::isinf(cpu_X_data[i])) {
            std::cerr << i << " ";
          }
        }
      }
      std::cerr << std::endl;
    }
    return false;
  }

  // This op should act as an identity matrix if we don't find any NaNs/infs.
  // Copy over the data if we are not doing this in-place.
  if (&X != Y) {
    Y->CopyFrom(X, true /*async*/);
  }
  return true;
}

REGISTER_CUDA_OPERATOR(NanCheck, NanCheckOp<CUDAContext>);

/**
 * @brief Update slices of Y in-place with a batch of weighted X's.
 * Y[idx] = alpha[b] * X[b][i] + Y[idx]
 * i=0,...,N-1
 * b=0,...,B-1
 * idx=Indices[i]
 */
template <typename T_INDEX>
__global__ void AxpySliceKernel(
    const float* weight0,
    const int64_t N,
    const int64_t B,
    const int64_t slice_size,
    const float** alpha,
    const float** X,
    const T_INDEX* Indices,
    float* Y,
    const int64_t M) {
  // This implementation requires that the first weight is 1.0
  CUDA_KERNEL_ASSERT(weight0[0] == 1.0);
  for (int i = blockIdx.x; i < N; i += gridDim.x) {
    T_INDEX idx = Indices[i];
    float* y_offset = Y + (idx * slice_size);
    for (int b = 0; b < B; b++) {
      float a = *alpha[b];
      const float* x_offset = X[b] + (i * slice_size);
      for (int j = threadIdx.x; j < slice_size; j += blockDim.x) {
        atomicAdd(&y_offset[j], a * x_offset[j]);
      }
    }
  }
}

// this kernel is a custom version of AxpySliceKernel
// to be used when there is only one weighted X to update
// slice of Y.
template <typename T_INDEX>
__global__ void AxpySliceKernel2(
    const float* weight0,
    const int64_t N,
    const int64_t slice_size,
    const float* alpha,
    const float* X,
    const T_INDEX* Indices,
    float* Y,
    const int64_t M) {
  // This implementation requires that the first weight is 1.0
  CUDA_KERNEL_ASSERT(weight0[0] == 1.0);
  for (int i = blockIdx.x; i < N; i += gridDim.x) {
    T_INDEX idx = Indices[i];
    float* y_offset = Y + (idx * slice_size);
    for (int j = threadIdx.x; j < slice_size; j += blockDim.x) {
      atomicAdd(&y_offset[j], alpha[0] * X[(i * slice_size) + j]);
    }
  }
}

template <>
bool ScatterWeightedSumOp<float, CUDAContext>::RunOnDevice() {
  return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(this, Input(2));
}

template <>
template <typename Index>
bool ScatterWeightedSumOp<float, CUDAContext>::DoRunWithType() {
  CAFFE_ENFORCE_EQ(InputSize() % 2, 1);
  auto& X0 = Input(0);
  auto& weight0 = Input(1);
  auto& indices = Input(2);
  auto* output = Output(0);

  CAFFE_ENFORCE_EQ(&X0, output, "In place operation is required");
  CAFFE_ENFORCE_GT(X0.numel(), 0);
  CAFFE_ENFORCE_GT(X0.dim(), 0, "X0 has to be at least the vector");
  CAFFE_ENFORCE_EQ(weight0.numel(), 1);

  int64_t M = X0.numel();
  int64_t N = X0.dim(0);
  int64_t K = indices.numel();
  int64_t block_size = M / N;

  float* data = output->template mutable_data<float>();

  const int64_t B = (InputSize() - 3) / 2;
  if (B > 1) {
    // In order to have all device pointers of x_i (and weight_i similarly)
    // consecutively in device memory, copy pointers to a host vector and then
    // copy back into a device array.
    ReinitializeTensor(&x_data_host_, {B}, at::dtype<float*>().device(CPU));
    ReinitializeTensor(&weights_host_, {B}, at::dtype<float*>().device(CPU));
    ReinitializeTensor(&x_data_device_, {B}, at::dtype<float*>().device(CUDA));
    ReinitializeTensor(&weights_device_, {B}, at::dtype<float*>().device(CUDA));

    float** x_data_host = x_data_host_.mutable_data<float*>();
    float** weights_host = weights_host_.mutable_data<float*>();
    float** x_data_device = x_data_device_.mutable_data<float*>();
    float** weights_device = weights_device_.mutable_data<float*>();

    for (int inp = 3; inp < InputSize(); inp += 2) {
      int idx = (inp - 3) / 2;
      x_data_host[idx] = static_cast<float*>(Input(inp).raw_data());
      weights_host[idx] = static_cast<float*>(Input(inp + 1).raw_data());
    }
    context_.Copy<float*, CPUContext, CUDAContext>(
        B, x_data_host, x_data_device);
    context_.Copy<float*, CPUContext, CUDAContext>(
        B, weights_host, weights_device);

    AxpySliceKernel<<<
        std::min<int64_t>(K, CAFFE_MAXIMUM_NUM_BLOCKS),
        CAFFE_CUDA_NUM_THREADS,
        0,
        context_.cuda_stream()>>>(
        weight0.template data<float>(),
        K,
        B,
        block_size,
        const_cast<const float**>(weights_device),
        const_cast<const float**>(x_data_device),
        indices.template data<Index>(),
        data,
        M);
  } else {
    // when only one input exists to update data buffer,
    // avoid copying pointers to device array to prevent
    // copy overhead
    auto& X1 = Input(3);
    auto& weight1 = Input(4);
    AxpySliceKernel2<<<
        std::min<int64_t>(K, CAFFE_MAXIMUM_NUM_BLOCKS),
        CAFFE_CUDA_NUM_THREADS,
        0,
        context_.cuda_stream()>>>(
        weight0.template data<float>(),
        K,
        block_size,
        weight1.template data<float>(),
        X1.template data<float>(),
        indices.template data<Index>(),
        data,
        M);
  }
  return true;
}

REGISTER_CUDA_OPERATOR(
    ScatterWeightedSum,
    ScatterWeightedSumOp<float, CUDAContext>);

namespace {

template <typename Index, typename T>
__global__ void scatter_assign_kernel(
    T* data,
    const Index* idxs,
    const T* slicesData,
    int64_t N,
    int64_t K,
    int64_t block_size) {
  for (int64_t i = blockIdx.x; i < K; i += gridDim.x) {
    Index idx = idxs[i];
    CUDA_KERNEL_ASSERT(0 <= idx && idx < N);
    const T* src = slicesData + block_size * i;
    T* dest = data + block_size * idx;
    for (int64_t j = threadIdx.x; j < block_size; j += blockDim.x) {
      dest[j] = src[j];
    }
  }
}

} // namespace

template <>
template <typename Index, typename T>
void ScatterAssignOp<CUDAContext>::DoScatterAssign(
    T* data,
    const Index* idxs,
    const T* slicesData,
    int64_t N,
    int64_t K,
    int64_t block_size) {
  scatter_assign_kernel<<<
      std::min(K, static_cast<int64_t>(CAFFE_MAXIMUM_NUM_BLOCKS)),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(data, idxs, slicesData, N, K, block_size);
}

REGISTER_CUDA_OPERATOR(ScatterAssign, ScatterAssignOp<CUDAContext>);

REGISTER_CUDA_OPERATOR(Size, SizeOp<CUDAContext>);

template <typename T>
__global__ void RangeKernel(const int n, T* Y, T offset, T step) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    Y[index] = index * step + offset;
  }
}

template <>
template <typename T>
bool RangeOp<CUDAContext>::DoRunOnDevice(
    const T& start,
    const T& step,
    Tensor* output) {
  int N = output->numel();
  RangeKernel<<<
      CAFFE_GET_BLOCKS(N),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      N, output->template mutable_data<T>(), start, step);
  return true;
}

REGISTER_CUDA_OPERATOR(Range, RangeOp<CUDAContext>);
} // namespace caffe2
