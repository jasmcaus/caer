#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/TensorUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/WrapDimUtils.h>
#include <THC/THCTensorMathReduce.cuh>
#include <THC/THCTensorSort.cuh>
#include <THC/THCThrustAllocator.cuh>
#include <c10/macros/Macros.h>

#include <ATen/AccumulateType.h>
#include <ATen/cuda/NumericLimits.cuh>
#include <type_traits>

#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/MemoryAccess.cuh>
#include <ATen/native/cuda/PersistentSoftmax.cuh>

namespace at {
namespace native {

namespace {

constexpr int ALIGN_BYTES = 16;

template<typename T, typename AccumT, typename OutT>
struct LogSoftMaxForwardEpilogue {
  __device__ __forceinline__ LogSoftMaxForwardEpilogue(AccumT max_input, AccumT sum)
    : max_input(max_input),  logsum(std::log(sum)) {}

  __device__ __forceinline__ OutT operator()(T input) const {
    return static_cast<OutT>(input - max_input - logsum);
}

  const AccumT max_input;
  const AccumT logsum;
};

template<typename T, typename AccumT, typename OutT>
struct LogSoftMaxBackwardEpilogue {
  __device__ __forceinline__ LogSoftMaxBackwardEpilogue(AccumT sum)
    : sum(sum) {}

  __device__ __forceinline__ T operator()(OutT gradOutput, OutT output) const {
    return static_cast<T>(gradOutput - std::exp(static_cast<AccumT>(output)) * sum);
  }

  const AccumT sum;
};

template<typename T, typename AccumT, typename OutT>
struct SoftMaxForwardEpilogue {
  __device__ __forceinline__ SoftMaxForwardEpilogue(AccumT max_input, AccumT sum)
    : max_input(max_input)
    , sum(sum) {}

  __device__ __forceinline__ OutT operator()(T input) const {
    return static_cast<OutT>(std::exp(input - max_input) / sum);
  }

  const AccumT max_input;
  const AccumT sum;
};

template<typename T, typename AccumT, typename OutT>
struct SoftMaxBackwardEpilogue {
  __device__ __forceinline__ SoftMaxBackwardEpilogue(AccumT sum)
    : sum(sum) {}

  // XXX: gradOutput that we get here is really gradOutput * output
  // Look for cmul in SoftMax_updateGradInput
  __device__ __forceinline__ T operator()(OutT gradOutput, OutT output) const {
    return static_cast<T>(gradOutput - output * sum);
  }

  const AccumT sum;
};




////////////////////////////////////////////////////////////////////////////////
// Spatial kernel (fast with large inner_size and small dim_size)
////////////////////////////////////////////////////////////////////////////////
// Let's assume that our input has been flattened to have only three dimension:
//     outer x dim x inner
// The spatial algorithm tries to parallelize along all of them.
// Within a 2d block threadIdx.y parallelizes over dim slices, and threads that
// share it will speed up reductions over dim (along axis x).
// The 2d grid is used to parallelize inner dimension over y axis and outer over x.
inline dim3 SpatialSoftMax_getGridSize(
    dim3 block, uint32_t max_active_blocks,
    uint64_t outer_size, uint64_t dim_size, uint64_t inner_size) {
  // First, tile as many blocks as we can over the y axis
  uint32_t inner_blocks = (inner_size + block.y - 1) / block.y;
  if (inner_blocks > max_active_blocks)
    inner_blocks = max_active_blocks;
  // Fill the x axis with as many blocks as we can fit (a little more is ok too)
  uint32_t outer_blocks = (max_active_blocks + inner_blocks - 1) / inner_blocks;
  if (outer_blocks > outer_size)
    outer_blocks = outer_size;
  return dim3(outer_blocks, inner_blocks);
}

const int max_threads = 1024;

inline dim3 SpatialSoftMax_getBlockSize(
  uint64_t outer_size, uint64_t dim_size, uint64_t inner_size) {
  uint32_t inner_threads = inner_size;
  inner_threads = std::min(inner_threads, static_cast<uint32_t>(max_threads));
  uint32_t dim_threads = 1;
  if (inner_threads <= 64 && dim_size >= 64) {
    while (inner_threads * dim_threads <= max_threads && dim_threads <= dim_size)
      dim_threads *= 2;
    dim_threads /= 2;
  }
  return dim3(dim_threads, inner_threads);
}


template<typename accscalar_t, typename Kernel>
void SpatialSoftMax_getLaunchSizes(
    Kernel k,
    uint64_t outer_size, uint64_t dim_size, uint64_t inner_size,
    dim3& grid, dim3& block, uint32_t& smem_size) {
  block = SpatialSoftMax_getBlockSize(outer_size, dim_size, inner_size);
  uint32_t block_threads = block.x * block.y;
  smem_size = block.x == 1 ? 0 : block_threads * sizeof(accscalar_t);
  int max_active_blocks;
#if defined(__HIP_PLATFORM_HCC__) && HIP_VERSION < 305
  // HIP function signature is not compatible yet.
  uint32_t max_blocks;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks,
                                                k, block_threads, smem_size);
  max_active_blocks = max_blocks;
#else
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks,
                                                k, block_threads, smem_size);
#endif
  max_active_blocks *= at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  grid = SpatialSoftMax_getGridSize(block, max_active_blocks, outer_size, dim_size, inner_size);
}

inline dim3 SoftMax_getBlockSize(int ILP, uint64_t dim_size) {
  uint64_t block_size = 1;
  uint64_t max_block_size = std::min(dim_size / ILP, static_cast<uint64_t>(max_threads));

  // In the vectorized case we want to trade off allowing more of the buffers to be accessed
  // in a vectorized way against wanting a larger block size to get better utilisation.
  // In general with ILP you can have (ILP-1)/ILP of the buffer accessed vectorised, at the risk
  // of having a very small block size. We choose to keep >= 1/2 of the buffer vectorised while
  // allowing a larger block size.
  if (ILP > 1) {
    max_block_size /= 2;
  }

  while (block_size < (max_block_size)) block_size *= 2;
  // Launch at least a single warp - the kernel assumes that.
  block_size = std::max(block_size, static_cast<uint64_t>(C10_WARP_SIZE));
  return dim3(block_size);
}

template<typename T>
struct Add {
  __device__ __forceinline__ T operator()(T a, T b) const {
    return a + b;
  }
};

template<typename T>
struct Max {
  __device__ __forceinline__ T operator()(T a, T b) const {
    return a < b ? b : a;
  }
};

// Note that it's not a complete block-wide reduction.
// Only threads that share threadIdx.y reduce values.
template<typename T, template<typename> class ReduceOp>
__forceinline__ __device__
T spatialBlockReduceX(T *shared, T val) {
  ReduceOp<T> r;
  shared += threadIdx.y * blockDim.x;

  __syncthreads();

  shared[threadIdx.x] = val;

  // NOTE: loop starts with __syncthreads()
  int offset = blockDim.x / 2;
  while (offset > 0) {
    __syncthreads();
    if (threadIdx.x < offset)
      shared[threadIdx.x] = r(shared[threadIdx.x], shared[threadIdx.x + offset]);
    offset /= 2;
  }

  __syncthreads();

  return shared[0];
}

template <typename scalar_t, typename accscalar_t, typename outscalar_t, template<typename, typename, typename> class Epilogue>
__global__ void cunn_SpatialSoftMaxForward(
    outscalar_t *output, scalar_t *input,
    uint32_t outer_size, uint32_t dim_size, uint32_t inner_size)
{
  extern __shared__ unsigned char smem[];
  auto sdata = reinterpret_cast<accscalar_t*>(smem);
  const uint32_t outer_stride = inner_size * dim_size;
  const uint32_t dim_stride = inner_size;

  for (uint32_t outer_index = blockIdx.x; outer_index < outer_size; outer_index += gridDim.x) {
    const uint32_t outer_offset = outer_index * outer_stride;
    for (uint32_t inner_index = blockIdx.y * blockDim.y + threadIdx.y; inner_index < inner_size; inner_index += blockDim.y * gridDim.y) {
      const uint32_t data_offset = outer_offset + inner_index;
      ////////////////////////////////////////////////////////////
      // These two blocks are really equivalent, but specializing on
      // blockDim.x == 1 makes the kernel faster when it's unused.
      // I didn't want to thread an extra template parameter, and nvcc
      // seems to be smart enough to hoist the if outside of the loops.
      ////////////////////////////////////////////////////////////

      if (blockDim.x > 1) {
        accscalar_t max_input = at::numeric_limits<accscalar_t>::lowest();
        for (uint32_t d = threadIdx.x; d < dim_size; d += blockDim.x) {
          const accscalar_t value = static_cast<accscalar_t>(input[data_offset + d * dim_stride]);
          max_input = Max<accscalar_t>()(max_input, value);
        }
        max_input = spatialBlockReduceX<accscalar_t, Max>(sdata,max_input);

        accscalar_t sum = 0;
        for (uint32_t d = threadIdx.x; d < dim_size; d += blockDim.x)
          sum += std::exp(static_cast<accscalar_t>(input[data_offset + d * dim_stride])
                 - max_input);
        sum = spatialBlockReduceX<accscalar_t, Add>(sdata, sum);

        Epilogue<scalar_t, accscalar_t, outscalar_t> epilogue(max_input, sum);
        for (uint32_t d = threadIdx.x; d < dim_size; d += blockDim.x)
          output[data_offset + d * dim_stride] = epilogue(input[data_offset + d * dim_stride]);
      } else {
        accscalar_t max_input = at::numeric_limits<accscalar_t>::lowest();
        for (uint32_t d = threadIdx.x; d < dim_size; d += blockDim.x) {
          const accscalar_t value = static_cast<accscalar_t>(input[data_offset + d * dim_stride]);
          max_input = Max<accscalar_t>()(max_input, value);
        }
        accscalar_t sum = 0;
        for (uint32_t d = threadIdx.x; d < dim_size; d += blockDim.x)
          sum += std::exp(static_cast<accscalar_t>(input[data_offset + d * dim_stride])
                 - max_input);
        Epilogue<scalar_t, accscalar_t, outscalar_t> epilogue(max_input, sum);
        for (uint32_t d = threadIdx.x; d < dim_size; d += blockDim.x)
          output[data_offset + d * dim_stride] = epilogue(input[data_offset + d * dim_stride]);
      }
    }
  }
}



template <typename scalar_t, typename accscalar_t, typename outscalar_t, template<typename, typename, typename> class Epilogue>
__global__ void cunn_SpatialSoftMaxBackward(
    scalar_t *gradInput, outscalar_t *output, outscalar_t *gradOutput,
    uint32_t outer_size, uint32_t dim_size, uint32_t inner_size)
{
  extern __shared__ unsigned char smem[];
  auto sdata = reinterpret_cast<accscalar_t*>(smem);
  const uint32_t outer_stride = inner_size * dim_size;
  const uint32_t dim_stride = inner_size;

  for (uint32_t outer_index = blockIdx.x; outer_index < outer_size; outer_index += gridDim.x) {
    const uint32_t outer_offset = outer_index * outer_stride;
    for (uint32_t inner_index = blockIdx.y * blockDim.y + threadIdx.y; inner_index < inner_size; inner_index += blockDim.y * gridDim.y) {
      const uint32_t data_offset = outer_offset + inner_index;
      // See the comment in forward kernel
      if (blockDim.x > 1) {
        accscalar_t sum = 0;
        for (uint32_t d = threadIdx.x; d < dim_size; d += blockDim.x)
          sum += gradOutput[data_offset + d * dim_stride];
        sum = spatialBlockReduceX<accscalar_t, Add>(sdata, sum);

        Epilogue<scalar_t, accscalar_t, outscalar_t> epilogue(sum);
        for (uint32_t d = threadIdx.x; d < dim_size; d += blockDim.x) {
          gradInput[data_offset + d * dim_stride] =
            epilogue(gradOutput[data_offset + d * dim_stride],
                    output[data_offset + d * dim_stride]);
        }
      } else {
        accscalar_t sum = 0;
        for (uint32_t d = 0; d < dim_size; d++)
          sum += gradOutput[data_offset + d * dim_stride];

        Epilogue<scalar_t, accscalar_t, outscalar_t> epilogue(sum);
        for (uint32_t d = 0; d < dim_size; d++) {
          gradInput[data_offset + d * dim_stride] =
            epilogue(gradOutput[data_offset + d * dim_stride],
                    output[data_offset + d * dim_stride]);
        }
      }
    }
  }
}


////////////////////////////////////////////////////////////////////////////////
// Regular kernel (fast when dim_size is large; requires inner_size == 1)
////////////////////////////////////////////////////////////////////////////////


template <typename T, typename AccumT>
struct MaxFloat
{
  __device__ __forceinline__ AccumT operator()(AccumT max, T v) const {
    return ::max(max, (AccumT)v);
  }
};

template<typename T, typename AccumT>
struct AddFloat
{
  __device__ __forceinline__ AccumT operator()(AccumT sum, T v) const {
    return sum + v;
  }
};

template<typename T, typename AccumT>
struct SumExpFloat
{
  __device__ __forceinline__ SumExpFloat(AccumT v)
    : max_k(v) {}

  __device__ __forceinline__ AccumT operator()(AccumT sum, T v) const {
    return sum + std::exp(v - max_k);
  }

  const AccumT max_k;
};

template <template<typename> class Reduction, typename AccumT>
__device__ __forceinline__ AccumT
blockReduce(AccumT* smem, AccumT val,
            const Reduction<AccumT>& r,
            AccumT defaultVal)
{
  // To avoid RaW races from chaining blockReduce calls together, we need a sync here
  __syncthreads();

  smem[threadIdx.x] = val;

  __syncthreads();

  AccumT warpVal = defaultVal;

  // First warp will perform per-warp reductions for the remaining warps
  uint32_t mask = (((uint64_t)1) << (blockDim.x / C10_WARP_SIZE)) - 1;
  if (threadIdx.x < C10_WARP_SIZE) {
    int lane = threadIdx.x % C10_WARP_SIZE;
    if (lane < blockDim.x / C10_WARP_SIZE) {
#pragma unroll
      for (int i = 0; i < C10_WARP_SIZE; ++i) {
        warpVal = r(warpVal, smem[lane * C10_WARP_SIZE + i]);
      }
#ifndef __HIP_PLATFORM_HCC__
      __syncwarp(mask);
#endif
      smem[lane] = warpVal;
    }
  }

  __syncthreads();

  // First thread will perform a reduction of the above per-warp reductions
  AccumT blockVal = defaultVal;

  if (threadIdx.x == 0) {
    for (int i = 0; i < blockDim.x / C10_WARP_SIZE; ++i) {
      blockVal = r(blockVal, smem[i]);
    }
    smem[0] = blockVal;
  }

  // Sync and broadcast
  __syncthreads();
  return smem[0];
}

template <template<typename, typename> class Reduction, int ILP, typename T, typename AccumT>
__device__ __forceinline__ AccumT
ilpReduce(int shift,
          T* data,
          int size,
          const Reduction<T, AccumT>& r,
          AccumT defaultVal)
{
  using LoadT = at::native::memory::aligned_vector<T, ILP>;
  AccumT threadVal = defaultVal;
  int offset = threadIdx.x;

  // shift and do 1
  if(shift > 0){
    data -= shift;
    size += shift;
    if(threadIdx.x >= shift){
      threadVal = r(threadVal, data[offset]);
    }
    size -= blockDim.x;
    data += blockDim.x;
  }
  int last = size % (ILP * blockDim.x);

  T v[ILP];
  LoadT* value = reinterpret_cast<LoadT*>(&v);

  for (; offset * ILP < (size - last); offset += blockDim.x) {
    *value = reinterpret_cast<LoadT*>(data)[offset];

    #pragma unroll
    for (int j = 0; j < ILP; ++j) {
      threadVal = r(threadVal, v[j]);
    }
  }

  offset = size - last + threadIdx.x;
  // Epilogue
  for (; offset < size; offset += blockDim.x)
    threadVal = r(threadVal, data[offset]);

  return threadVal;
}

/**
 * This will apply the Epilogue with vectorized reads & writes when input & output have the same shift
 */
template <int ILP, typename scalar_t, typename accum_t, typename outscalar_t, template<typename, typename, typename> class Epilogue>
__device__ __forceinline__ void
WriteFpropResultsVectorized(
             int size,
             const int shift,
             scalar_t *input,
             outscalar_t *output,
             Epilogue<scalar_t, accum_t, outscalar_t> epilogue) {
  using LoadT = at::native::memory::aligned_vector<scalar_t, ILP>;
  using StoreT = at::native::memory::aligned_vector<outscalar_t, ILP>;

  int offset = threadIdx.x;

  // if unaligned, do one value / thread and move on, guaranteeing aligned reads/writes later
  if (shift > 0) {
    input -= shift;
    output -= shift;
    size += shift;

    if (threadIdx.x >= shift) {
      output[offset] = epilogue(input[offset]);
    }
    size -= blockDim.x;
    input += blockDim.x;
    output += blockDim.x;
  }

  const int last = size % (ILP * blockDim.x);

  scalar_t in_v[ILP];
  LoadT* in_value = reinterpret_cast<LoadT*>(&in_v);

  outscalar_t out_v[ILP];
  StoreT* out_value = reinterpret_cast<StoreT*>(&out_v);

  for (; offset * ILP < (size - last); offset += blockDim.x) {
    *in_value = reinterpret_cast<LoadT*>(input)[offset];

    #pragma unroll
    for (int j = 0; j < ILP; ++j) {
      out_v[j] = epilogue(in_v[j]);
    }

    reinterpret_cast<StoreT*>(output)[offset] = *out_value;
  }

  offset = size - last + threadIdx.x;
  // handle the tail
  for (; offset < size; offset += blockDim.x) {
    output[offset] = epilogue(input[offset]);
  }
}

template <int ILP, typename scalar_t, typename accum_t, typename outscalar_t, template<typename, typename, typename> class Epilogue>
__device__ __forceinline__ void
WriteBpropResultsVectorized(
             int size,
             const int shift,
             scalar_t *gradInput,
             outscalar_t *output,
             outscalar_t *gradOutput,
             Epilogue<scalar_t, accum_t, outscalar_t> epilogue) {
  using gradInputT = at::native::memory::aligned_vector<scalar_t, ILP>;
  using outputT = at::native::memory::aligned_vector<outscalar_t, ILP>;

  int offset = threadIdx.x;

  // if unaligned, do one value / thread and move on, guaranteeing aligned reads/writes later
  if (shift > 0) {
    gradInput -= shift;
    output -= shift;
    gradOutput -= shift;
    size += shift;

    if (threadIdx.x >= shift) {
      gradInput[offset] = epilogue(gradOutput[offset], output[offset]);
    }
    size -= blockDim.x;
    gradInput += blockDim.x;
    output += blockDim.x;
    gradOutput += blockDim.x;
  }

  const int last = size % (ILP * blockDim.x);

  scalar_t dX[ILP];
  gradInputT *dX_v = reinterpret_cast<gradInputT*>(&dX);

  outscalar_t Y[ILP];
  outputT *Y_v = reinterpret_cast<outputT*>(&Y);

  outscalar_t dY[ILP];
  outputT *dY_v = reinterpret_cast<outputT*>(&dY);

  for (; offset * ILP < (size - last); offset += blockDim.x) {
    *Y_v = reinterpret_cast<outputT*>(output)[offset];
    *dY_v = reinterpret_cast<outputT*>(gradOutput)[offset];

    #pragma unroll
    for (int j = 0; j < ILP; ++j) {
      dX[j] = epilogue(dY[j], Y[j]);
    }

    reinterpret_cast<gradInputT*>(gradInput)[offset] = *dX_v;
  }

  offset = size - last + threadIdx.x;
  for (; offset < size; offset += blockDim.x) {
    gradInput[offset] = epilogue(gradOutput[offset], output[offset]);
  }
}

/**
 * This will apply the Epilogue with non-vectrorized reads & writes for the general case
 */
template <int ILP, typename scalar_t, typename accum_t, typename outscalar_t, template<typename, typename, typename> class Epilogue>
__device__ __forceinline__ void
WriteFpropResults(
             int classes,
             scalar_t *input,
             outscalar_t *output,
             Epilogue<scalar_t, accum_t, outscalar_t> epilogue) {
  int offset = threadIdx.x;

  int last = classes % (ILP * blockDim.x);

  // Main bulk of loop with ILP
  for (; offset < classes - last; offset += blockDim.x * ILP) {
    scalar_t tmp[ILP];

    #pragma unroll
    for (int j = 0; j < ILP; ++j) {
      tmp[j] = input[offset + j * blockDim.x];
    }
    #pragma unroll
    for (int j = 0; j < ILP; ++j) {
      output[offset + j * blockDim.x] = epilogue(tmp[j]);
    }
  }

  // Remainder - no ILP
  for (; offset < classes; offset += blockDim.x) {
    output[offset] = epilogue(input[offset]);
  }
}

template <int ILP, typename scalar_t, typename accum_t, typename outscalar_t, template<typename, typename, typename> class Epilogue>
__device__ __forceinline__ void
WriteBpropResults(
             int classes,
             scalar_t *gradInput,
             outscalar_t *output,
             outscalar_t *gradOutput,
             Epilogue<scalar_t, accum_t, outscalar_t> epilogue) {

  int offset = threadIdx.x;

  int last = classes % (ILP * blockDim.x);

  for (; offset < classes - last; offset += blockDim.x * ILP) {
    outscalar_t tmpOutput[ILP];
    outscalar_t tmpGradOutput[ILP];

    #pragma unroll
    for (int j = 0; j < ILP; ++j) {
      tmpOutput[j] = output[offset + j * blockDim.x];
      tmpGradOutput[j] = gradOutput[offset + j * blockDim.x];
    }

    #pragma unroll
    for (int j = 0; j < ILP; ++j) {
      gradInput[offset + j * blockDim.x] = epilogue(tmpGradOutput[j], tmpOutput[j]);
    }
  }

  // Remainder - no ILP
  for (; offset < classes; offset += blockDim.x) {
    gradInput[offset] = epilogue(gradOutput[offset], output[offset]);
  }
}

template <int ILP, typename scalar_t, typename accscalar_t, typename outscalar_t, template <typename, typename, typename> class Epilogue>
__global__ void
cunn_SoftMaxForward(outscalar_t *output, scalar_t *input, int classes)
{
  extern __shared__ unsigned char smem[];
  auto sdata = reinterpret_cast<accscalar_t*>(smem);

  using LoadT = at::native::memory::aligned_vector<scalar_t, ILP>;
  using StoreT = at::native::memory::aligned_vector<outscalar_t, ILP>;

  // forward pointers to batch[blockIdx.x]
  // each block handles a sample in the mini-batch
  input += blockIdx.x * classes;
  output += blockIdx.x * classes;

  const int shift = ((uint64_t)input) % ALIGN_BYTES / sizeof(scalar_t);
  const int output_shift = ((uint64_t)output) % ALIGN_BYTES / sizeof(outscalar_t);

  // find the max
  accscalar_t threadMax = ilpReduce<MaxFloat, ILP, scalar_t, accscalar_t>(
      shift, input, classes, MaxFloat<scalar_t, accscalar_t>(), -at::numeric_limits<accscalar_t>::max());
  accscalar_t max_k = blockReduce<Max, accscalar_t>(
      sdata, threadMax, Max<accscalar_t>(), -at::numeric_limits<accscalar_t>::max());

  // reduce all values
  accscalar_t threadExp = ilpReduce<SumExpFloat, ILP, scalar_t, accscalar_t>(
      shift, input, classes, SumExpFloat<scalar_t, accscalar_t>(max_k), static_cast<accscalar_t>(0));
  accscalar_t sumAll = blockReduce<Add, accscalar_t>(
      sdata, threadExp, Add<accscalar_t>(), static_cast<accscalar_t>(0));

  Epilogue<scalar_t, accscalar_t, outscalar_t> epilogue(max_k, sumAll);

  if (shift == output_shift) {
    WriteFpropResultsVectorized<ILP, scalar_t, accscalar_t, outscalar_t, Epilogue>(classes, shift, input, output, epilogue);
  } else {
    WriteFpropResults<ILP, scalar_t, accscalar_t, outscalar_t, Epilogue>(classes, input, output, epilogue);
  }
}

template <int ILP, typename scalar_t, typename accscalar_t, typename outscalar_t, template<typename, typename, typename> class Epilogue>
__global__ void
cunn_SoftMaxBackward(scalar_t *gradInput, outscalar_t *output, outscalar_t *gradOutput, int classes)
{
  using LoadT = at::native::memory::aligned_vector<scalar_t, ILP>;
  using StoreT = at::native::memory::aligned_vector<outscalar_t, ILP>;

  extern __shared__ unsigned char smem[];
  auto sdata = reinterpret_cast<accscalar_t*>(smem);
  gradInput += blockIdx.x * classes;
  output += blockIdx.x * classes;
  gradOutput += blockIdx.x * classes;

  const int shift = ((uint64_t)gradInput) % ALIGN_BYTES / sizeof(scalar_t);
  const int output_shift = ((uint64_t)output) % ALIGN_BYTES / sizeof(outscalar_t);
  const int grad_output_shift = ((uint64_t)gradOutput) % ALIGN_BYTES / sizeof(outscalar_t);

  accscalar_t threadSum = ilpReduce<AddFloat, ILP, outscalar_t, accscalar_t>(
      shift, gradOutput, classes, AddFloat<outscalar_t, accscalar_t>(), accscalar_t(0));
  accscalar_t sum_k = blockReduce<Add, accscalar_t>(
        sdata, threadSum, Add<accscalar_t>(), accscalar_t(0));

  Epilogue<scalar_t, accscalar_t, outscalar_t> epilogue(sum_k);

  if (shift == output_shift && shift == grad_output_shift) {
    WriteBpropResultsVectorized<ILP, scalar_t, accscalar_t, outscalar_t, Epilogue>(classes, shift, gradInput, output, gradOutput, epilogue);
  } else {
    WriteBpropResults<ILP, scalar_t, accscalar_t, outscalar_t, Epilogue>(classes, gradInput, output, gradOutput, epilogue);
  }
}

template<template<typename, typename, typename> class Epilogue, bool is_log_softmax>
Tensor host_softmax(const Tensor & input_, const int64_t dim_, const bool half_to_float){
  if (half_to_float) {
    TORCH_CHECK(input_.scalar_type() == ScalarType::Half, "conversion is supported for Half type only");
  }
  auto input = input_.contiguous();
  Tensor output = half_to_float ? at::empty_like(input, input.options().dtype(ScalarType::Float), LEGACY_CONTIGUOUS_MEMORY_FORMAT) : at::empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  static_assert(std::is_same<acc_type<at::Half, true>, float>::value, "accscalar_t for half should be float");
  if (input.dim() == 0) input = input.view(1);
  int64_t dim = maybe_wrap_dim(dim_, input.dim());
  TORCH_CHECK(dim >=0 && dim < input.dim(), "dim must be non-negative and less than input dimensions");
  int64_t outer_size = 1;
  int64_t dim_size = input.size(dim);

  if (input.numel() > 0) {
    int64_t inner_size = 1;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    for (int64_t i = 0; i < dim; ++i)
      outer_size *= input.size(i);
    for (int64_t i = dim + 1; i < input.dim(); ++i)
      inner_size *= input.size(i);
    // This kernel spawns a block per each element in the batch.
    // XXX: it assumes that inner_size == 1

    if (inner_size == 1) {
      dim3 grid(outer_size);
      AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(), "host_softmax", [&] {
        using accscalar_t = acc_type<scalar_t, true>;
        if (!half_to_float) {
          if (dim_size <= 1024 && dim_size*sizeof(scalar_t) <= 4096) {
            dispatch_softmax_forward<scalar_t, scalar_t, accscalar_t, is_log_softmax>(
                output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), dim_size, dim_size, outer_size);
          } else {
            constexpr int ILP = sizeof(float4) / sizeof(scalar_t);
            dim3 block = SoftMax_getBlockSize(ILP, dim_size);
            cunn_SoftMaxForward<ILP, scalar_t, accscalar_t, scalar_t, Epilogue>
              <<<grid, block, block.x * sizeof(accscalar_t), stream>>>(
                output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), dim_size);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
          }
        } else {
          if (dim_size <= 1024 && dim_size*sizeof(scalar_t) <= 4096) {
            dispatch_softmax_forward<scalar_t, accscalar_t, accscalar_t, is_log_softmax>(
                output.data_ptr<accscalar_t>(), input.data_ptr<scalar_t>(), dim_size, dim_size, outer_size);
          } else {
            constexpr int ILP = sizeof(float4) / sizeof(accscalar_t);
            dim3 block = SoftMax_getBlockSize(ILP, dim_size);
            cunn_SoftMaxForward<ILP, scalar_t, accscalar_t, accscalar_t, Epilogue>
              <<<grid, block, block.x * sizeof(accscalar_t), stream>>>(
                output.data_ptr<accscalar_t>(), input.data_ptr<scalar_t>(), dim_size);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
          }
        }
      });
    // This kernel runs in a 2D grid, where each application along y dimension has a fixed
    // outer_size, and runs in parallel over inner_size. Dimension x is parallel over outer_size.
    // Reductions over dim are done in a single-threaded manner.
    } else {
      uint32_t smem_size;
      dim3 grid, block;
      AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(), "host_softmax", [&] {
        using accscalar_t = acc_type<scalar_t, true>;
        if (!half_to_float) {
            SpatialSoftMax_getLaunchSizes<accscalar_t>(
                &cunn_SpatialSoftMaxForward<scalar_t, accscalar_t, scalar_t, Epilogue>,
                outer_size, dim_size, inner_size,
                grid, block, smem_size);
            cunn_SpatialSoftMaxForward<scalar_t, accscalar_t, scalar_t, Epilogue>
              <<<grid, block, smem_size, stream>>>(
              output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), outer_size, dim_size, inner_size);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        } else {
            SpatialSoftMax_getLaunchSizes<accscalar_t>(
                &cunn_SpatialSoftMaxForward<scalar_t, accscalar_t, accscalar_t, Epilogue>,
                outer_size, dim_size, inner_size,
                grid, block, smem_size);
            cunn_SpatialSoftMaxForward<scalar_t, accscalar_t, accscalar_t, Epilogue>
              <<<grid, block, smem_size, stream>>>(
              output.data_ptr<accscalar_t>(), input.data_ptr<scalar_t>(), outer_size, dim_size, inner_size);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
      });
    }
  }
  return output;
}

template<template<typename, typename, typename> class Epilogue, bool is_log_softmax>
Tensor host_softmax_backward(const Tensor &grad_, const Tensor &output_, int64_t dim_, bool half_to_float){
  int64_t dim = maybe_wrap_dim(dim_, grad_.dim());
  Tensor gI = half_to_float ? at::empty_like(grad_, grad_.options().dtype(ScalarType::Half), LEGACY_CONTIGUOUS_MEMORY_FORMAT) : at::empty_like(grad_, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  if (grad_.numel() == 0) {
    return gI;
  }
  auto grad = grad_.contiguous();
  static_assert(std::is_same<acc_type<at::Half, true>, float>::value, "accscalar_t for half should be float");
  if (grad.dim() == 0) grad = grad.view(1);
  TORCH_CHECK(dim >=0 && dim < grad.dim(), "dim must be non-negative and less than input dimensions");
  auto output = output_.contiguous();
  if (output.dim() == 0) output = output.view(1);
  int64_t outer_size = 1;
  int64_t dim_size = output.size(dim);
  int64_t inner_size = 1;
  for (int64_t i = 0; i < dim; ++i)
    outer_size *= output.size(i);
  for (int64_t i = dim + 1; i < output.dim(); ++i)
    inner_size *= output.size(i);
// See descriptions of kernels above.
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (inner_size == 1) {
    dim3 grid(outer_size);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, gI.scalar_type(), "host_softmax_backward", [&] {
    using accscalar_t = acc_type<scalar_t, true>;
    if (!half_to_float) {
      if (dim_size <= 1024 && dim_size*sizeof(scalar_t) <= 4096) {
        dispatch_softmax_backward<scalar_t, scalar_t, accscalar_t, is_log_softmax>(
            gI.data_ptr<scalar_t>(), grad.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), dim_size, dim_size, outer_size);
      } else {
        constexpr int ILP = sizeof(float4) / sizeof(scalar_t);
        dim3 block = SoftMax_getBlockSize(ILP, dim_size);
        cunn_SoftMaxBackward<ILP, scalar_t, accscalar_t, scalar_t, Epilogue>
         <<<grid, block, block.x * sizeof(accscalar_t), stream>>>(
            gI.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), grad.data_ptr<scalar_t>(), dim_size
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }
    } else {
      if (dim_size <= 1024 && dim_size*sizeof(scalar_t) <= 4096) {
        dispatch_softmax_backward<accscalar_t, scalar_t, accscalar_t, is_log_softmax>(
            gI.data_ptr<scalar_t>(), grad.data_ptr<accscalar_t>(), output.data_ptr<accscalar_t>(), dim_size, dim_size, outer_size);
      } else {
        constexpr int ILP = sizeof(float4) / sizeof(accscalar_t);
        dim3 block = SoftMax_getBlockSize(ILP, dim_size);
        cunn_SoftMaxBackward<ILP, scalar_t, accscalar_t, accscalar_t, Epilogue>
         <<<grid, block, block.x * sizeof(accscalar_t), stream>>>(
            gI.data_ptr<scalar_t>(), output.data_ptr<accscalar_t>(), grad.data_ptr<accscalar_t>(), dim_size
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }
    }
    });
  } else {
    uint32_t smem_size;
    dim3 grid, block;
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, gI.scalar_type(), "host_softmax_backward", [&] {
      using accscalar_t = acc_type<scalar_t, true>;
      if (!half_to_float) {
          SpatialSoftMax_getLaunchSizes<accscalar_t>(
              &cunn_SpatialSoftMaxBackward<scalar_t, accscalar_t, scalar_t, Epilogue>,
              outer_size, dim_size, inner_size,
              grid, block, smem_size);

          cunn_SpatialSoftMaxBackward<scalar_t, accscalar_t, scalar_t, Epilogue>
            <<<grid, block, smem_size, stream>>>(
              gI.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), grad.data_ptr<scalar_t>(),
              outer_size, dim_size, inner_size
          );
          C10_CUDA_KERNEL_LAUNCH_CHECK();
      } else {
          SpatialSoftMax_getLaunchSizes<accscalar_t>(
              &cunn_SpatialSoftMaxBackward<scalar_t, accscalar_t, accscalar_t, Epilogue>,
              outer_size, dim_size, inner_size,
              grid, block, smem_size);

          cunn_SpatialSoftMaxBackward<scalar_t, accscalar_t, accscalar_t, Epilogue>
            <<<grid, block, smem_size, stream>>>(
              gI.data_ptr<scalar_t>(), output.data_ptr<accscalar_t>(), grad.data_ptr<accscalar_t>(),
              outer_size, dim_size, inner_size
          );
          C10_CUDA_KERNEL_LAUNCH_CHECK();
      }
    });
  }

  return gI;
}
}

Tensor log_softmax_cuda(const Tensor &input, const int64_t dim, const bool half_to_float){
  return host_softmax<LogSoftMaxForwardEpilogue,true>(input, dim, half_to_float);
}

Tensor log_softmax_backward_cuda(const Tensor &grad, const Tensor &output, int64_t dim, const Tensor &input){
  bool half_to_float = grad.scalar_type() != input.scalar_type();
  if (half_to_float) {
     TORCH_CHECK((grad.scalar_type() == ScalarType::Float && input.scalar_type() == ScalarType::Half),
                 "expected input and grad types to match, or input to be at::Half and grad to be at::Float");
  }
  return host_softmax_backward<LogSoftMaxBackwardEpilogue,true>(grad, output, dim, half_to_float);
}

Tensor softmax_cuda(const Tensor &input, const int64_t dim, const bool half_to_float){
  return host_softmax<SoftMaxForwardEpilogue,false>(input, dim, half_to_float);
}

Tensor softmax_backward_cuda(const Tensor &grad, const Tensor &output, int64_t dim, const Tensor &input){
  bool half_to_float = grad.scalar_type() != input.scalar_type();
  if (half_to_float) {
     TORCH_CHECK((grad.scalar_type() == ScalarType::Float && input.scalar_type() == ScalarType::Half),
                 "expected input and grad types to match, or input to be at::Half and grad to be at::Float");
  }
  Tensor tmp = grad * output;
  return host_softmax_backward<SoftMaxBackwardEpilogue,false>(tmp, output, dim, half_to_float);
}

}
}
