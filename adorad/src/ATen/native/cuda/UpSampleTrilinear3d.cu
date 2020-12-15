// Adapted from interp.cpp from Caffe util by Pauline Luc
// Originally developed by George Papandreou
#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/NativeFunctions.h>
#include <ATen/TensorUtils.h>
#include <ATen/Utils.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/native/cuda/UpSample.cuh>
#include <ATen/native/cuda/KernelUtils.cuh>
#include <THC/THCAtomics.cuh>

namespace at {
namespace native {
namespace {

__device__ __forceinline__ size_t
idx_3d(const size_t nc,
    const size_t depth,
    const size_t height,
    const size_t width,
    const size_t z,
    const size_t y,
    const size_t x) {
  return ((nc * depth + z) * height + y) * width + x;
}

template <typename scalar_t, typename accscalar_t>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void upsample_trilinear3d_out_frame(
    const int n,
    const accscalar_t rdepth,
    const accscalar_t rheight,
    const accscalar_t rwidth,
    const bool align_corners,
    const PackedTensorAccessor64<scalar_t, 5> idata,
    PackedTensorAccessor64<scalar_t, 5> odata) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;

  const int batchsize = idata.size(0);
  const int channels = idata.size(1);
  const int depth1 = idata.size(2);
  const int height1 = idata.size(3);
  const int width1 = idata.size(4);
  const int depth2 = odata.size(2);
  const int height2 = odata.size(3);
  const int width2 = odata.size(4);

  if (index < n) {
    const int w2 = (index % (height2 * width2)) % width2; // 0:width2-1
    const int h2 = (index % (height2 * width2)) / width2; // 0:height2-1
    const int t2 = index / (height2 * width2); // 0:depth2-1
    // special case: just copy
    if (depth1 == depth2 && height1 == height2 && width1 == width2) {
      const int t1 = t2;
      const int h1 = h2;
      const int w1 = w2;

      for (int n = 0; n < batchsize; n++) {
        for (int c = 0; c < channels; ++c) {
          const scalar_t val = idata[n][c][t1][h1][w1];
          odata[n][c][t2][h2][w2] = val;
        }
      }
      return;
    }
    //
    const accscalar_t t1r = area_pixel_compute_source_index<accscalar_t>(
        rdepth, t2, align_corners, /*cubic=*/false);
    const int t1 = t1r;
    const int t1p = (t1 < depth1 - 1) ? 1 : 0;
    const accscalar_t t1lambda = t1r - t1;
    const accscalar_t t0lambda = static_cast<accscalar_t>(1) - t1lambda;
    //
    const accscalar_t h1r = area_pixel_compute_source_index<accscalar_t>(
        rheight, h2, align_corners, /*cubic=*/false);
    const int h1 = h1r;
    const int h1p = (h1 < height1 - 1) ? 1 : 0;
    const accscalar_t h1lambda = h1r - h1;
    const accscalar_t h0lambda = static_cast<accscalar_t>(1) - h1lambda;
    //
    const accscalar_t w1r = area_pixel_compute_source_index<accscalar_t>(
        rwidth, w2, align_corners, /*cubic=*/false);
    const int w1 = w1r;
    const int w1p = (w1 < width1 - 1) ? 1 : 0;
    const accscalar_t w1lambda = w1r - w1;
    const accscalar_t w0lambda = static_cast<accscalar_t>(1) - w1lambda;
    //
    for (int n = 0; n < batchsize; n++) {
      for (int c = 0; c < channels; ++c) {
        const accscalar_t val = t0lambda *
                (h0lambda *
                     (w0lambda * idata[n][c][t1][h1][w1] +
                      w1lambda * idata[n][c][t1][h1][w1 + w1p]) +
                 h1lambda *
                     (w0lambda * idata[n][c][t1][h1 + h1p][w1] +
                      w1lambda * idata[n][c][t1][h1 + h1p][w1 + w1p])) +
            t1lambda *
                (h0lambda *
                     (w0lambda * idata[n][c][t1 + t1p][h1][w1] +
                      w1lambda * idata[n][c][t1 + t1p][h1][w1 + w1p]) +
                 h1lambda *
                     (w0lambda * idata[n][c][t1 + t1p][h1 + h1p][w1] +
                      w1lambda * idata[n][c][t1 + t1p][h1 + h1p][w1 + w1p]));
        odata[n][c][t2][h2][w2] = static_cast<scalar_t>(val);
      }
    }
  }
}

// Backward (adjoint) operation 1 <- 2 (accumulates)
template <typename scalar_t, typename accscalar_t>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void upsample_trilinear3d_backward_out_frame(
    const size_t nc_,
    const int depth1,
    const int height1,
    const int width1,
    const int depth2,
    const int height2,
    const int width2,
    const accscalar_t rdepth,
    const accscalar_t rheight,
    const accscalar_t rwidth,
    const bool align_corners,
    scalar_t* __restrict__ idata,
    const scalar_t* __restrict__ odata) {
  const size_t i_numel = nc_ * depth1 * height1 * width1;
  const size_t o_numel = nc_ * depth2 * height2 * width2;

  for (size_t index = blockDim.x * blockIdx.x + threadIdx.x; index < o_numel; index += blockDim.x * gridDim.x) {
    size_t index_temp = index;
    const int w2 = index_temp % width2;   // 0:width2-1
    index_temp /= width2;
    const int h2 = index_temp % height2;  // 0:height2-1
    index_temp /= height2;
    const int t2 = index_temp % depth2;   // 0:depth2-1
    const int nc = index_temp / depth2;

    const accscalar_t t1r = area_pixel_compute_source_index<accscalar_t>(
        rdepth, t2, align_corners, /*cubic=*/false);
    const int t1 = t1r;
    const int t1p = (t1 < depth1 - 1) ? 1 : 0;
    const accscalar_t t1lambda = t1r - t1;
    const accscalar_t t0lambda = static_cast<accscalar_t>(1) - t1lambda;
    //
    const accscalar_t h1r = area_pixel_compute_source_index<accscalar_t>(
        rheight, h2, align_corners, /*cubic=*/false);
    const int h1 = h1r;
    const int h1p = (h1 < height1 - 1) ? 1 : 0;
    const accscalar_t h1lambda = h1r - h1;
    const accscalar_t h0lambda = static_cast<accscalar_t>(1) - h1lambda;
    //
    const accscalar_t w1r = area_pixel_compute_source_index<accscalar_t>(
        rwidth, w2, align_corners, /*cubic=*/false);
    const int w1 = w1r;
    const int w1p = (w1 < width1 - 1) ? 1 : 0;
    const accscalar_t w1lambda = w1r - w1;
    const accscalar_t w0lambda = static_cast<accscalar_t>(1) - w1lambda;
    //
    const scalar_t d2val = odata[index];
    fastAtomicAdd(
      idata,
      idx_3d(nc, depth1, height1, width1, t1, h1, w1),
      i_numel,
      static_cast<scalar_t>(t0lambda * h0lambda * w0lambda * d2val),
      true);
    fastAtomicAdd(
      idata,
      idx_3d(nc, depth1, width1, height1, t1, h1, w1 + w1p),
      i_numel,
      static_cast<scalar_t>(t0lambda * h0lambda * w1lambda * d2val),
      true);
    fastAtomicAdd(
      idata,
      idx_3d(nc, depth1, width1, height1, t1, h1 + h1p, w1),
      i_numel,
      static_cast<scalar_t>(t0lambda * h1lambda * w0lambda * d2val),
      true);
    fastAtomicAdd(
      idata,
      idx_3d(nc, depth1, width1, height1, t1, h1 + h1p, w1 + w1p),
      i_numel,
      static_cast<scalar_t>(t0lambda * h1lambda * w1lambda * d2val),
      true);
    fastAtomicAdd(
      idata,
      idx_3d(nc, depth1, width1, height1, t1 + t1p, h1, w1),
      i_numel,
      static_cast<scalar_t>(t1lambda * h0lambda * w0lambda * d2val),
      true);
    fastAtomicAdd(
      idata,
      idx_3d(nc, depth1, width1, height1, t1 + t1p, h1, w1 + w1p),
      i_numel,
      static_cast<scalar_t>(t1lambda * h0lambda * w1lambda * d2val),
      true);
    fastAtomicAdd(
      idata,
      idx_3d(nc, depth1, width1, height1, t1 + t1p, h1 + h1p, w1),
      i_numel,
      static_cast<scalar_t>(t1lambda * h1lambda * w0lambda * d2val),
      true);
    fastAtomicAdd(
      idata,
      idx_3d(nc, depth1, width1, height1, t1 + t1p, h1 + h1p, w1 + w1p),
      i_numel,
      static_cast<scalar_t>(t1lambda * h1lambda * w1lambda * d2val),
      true);
  }
}

static void upsample_trilinear3d_out_cuda_template(
    Tensor& output,
    const Tensor& input,
    IntArrayRef output_size,
    bool align_corners,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  TensorArg input_arg{input, "input", 1}, output_arg{output, "output", 2};
  checkAllSameGPU("upsample_trilinear3d_out_cuda", {input_arg, output_arg});

  TORCH_CHECK(
      output_size.size() == 3,
      "It is expected output_size equals to 3, but got size ",
      output_size.size());

  int output_depth = output_size[0];
  int output_height = output_size[1];
  int output_width = output_size[2];

  int nbatch = input.size(0);
  int channels = input.size(1);
  int input_depth = input.size(2);
  int input_height = input.size(3);
  int input_width = input.size(4);

  upsample_3d_shape_check(
      input,
      Tensor(),
      nbatch,
      channels,
      input_depth,
      input_height,
      input_width,
      output_depth,
      output_height,
      output_width);

  output.resize_({input.size(0),
                  input.size(1),
                  output_depth,
                  output_height,
                  output_width});

  AT_ASSERT(
      input_depth > 0 && input_height > 0 && input_width > 0 &&
      output_depth > 0 && output_height > 0 && output_width > 0);

  const int num_kernels = output_depth * output_height * output_width;
  const int num_threads = std::min(
      at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock, 1024);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "upsample_trilinear3d_out_frame", [&] {
        using accscalar_t = at::acc_type<scalar_t, true>;

        auto idata = input.packed_accessor64<scalar_t, 5>();
        auto odata = output.packed_accessor64<scalar_t, 5>();

        const accscalar_t rdepth = area_pixel_compute_scale<accscalar_t>(
            input_depth, output_depth, align_corners, scales_d);
        const accscalar_t rheight = area_pixel_compute_scale<accscalar_t>(
            input_height, output_height, align_corners, scales_h);
        const accscalar_t rwidth = area_pixel_compute_scale<accscalar_t>(
            input_width, output_width, align_corners, scales_w);

        upsample_trilinear3d_out_frame<scalar_t, accscalar_t>
            <<<cuda::ATenCeilDiv(num_kernels, num_threads),
               num_threads,
               0,
               stream>>>(
                num_kernels,
                rdepth,
                rheight,
                rwidth,
                align_corners,
                idata,
                odata);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
}

static void upsample_trilinear3d_backward_out_cuda_template(
    Tensor& grad_input,
    const Tensor& grad_output_,
    IntArrayRef output_size,
    IntArrayRef input_size,
    bool align_corners,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  TensorArg grad_input_arg{grad_input, "grad_input", 1},
      grad_output_arg{grad_output_, "grad_output_", 2};
  checkAllSameGPU(
      "upsample_trilinear3d_backward_out_cuda",
      {grad_output_arg, grad_input_arg});

  TORCH_CHECK(
      output_size.size() == 3,
      "It is expected output_size equals to 3, but got size ",
      output_size.size());

  TORCH_CHECK(
      input_size.size() == 5,
      "It is expected input_size equals to 5, but got size ",
      input_size.size());

  int output_depth = output_size[0];
  int output_height = output_size[1];
  int output_width = output_size[2];

  int nbatch = input_size[0];
  int channels = input_size[1];
  int input_depth = input_size[2];
  int input_height = input_size[3];
  int input_width = input_size[4];

  upsample_3d_shape_check(
      Tensor(),
      grad_output_,
      nbatch,
      channels,
      input_depth,
      input_height,
      input_width,
      output_depth,
      output_height,
      output_width);
  Tensor grad_output = grad_output_.contiguous();

  grad_input.resize_(
      {nbatch, channels, input_depth, input_height, input_width});
  // A contiguous tensor is required for the kernel launch config
  grad_input.contiguous();
  // Numbers are added atomically to grad_input tensor from multiple threads,
  // so it has to be initialized to zero.
  grad_input.zero_();

  // const size_t num_kernels = nbatch * channels * output_depth * output_height * output_width;
  const size_t num_kernels = grad_output.numel();
  const int num_threads = std::min(
      at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock, 1024);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (num_kernels > 0) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad_output.scalar_type(),
      "upsample_trilinear3d_backward_out_frame",
      [&] {
        using accscalar_t = at::acc_type<scalar_t, true>;

        auto idata = grad_input.data_ptr<scalar_t>();
        auto odata = grad_output.data_ptr<scalar_t>();

        const accscalar_t rdepth = area_pixel_compute_scale<accscalar_t>(
            input_depth, output_depth, align_corners, scales_d);
        const accscalar_t rheight = area_pixel_compute_scale<accscalar_t>(
            input_height, output_height, align_corners, scales_h);
        const accscalar_t rwidth = area_pixel_compute_scale<accscalar_t>(
            input_width, output_width, align_corners, scales_w);

        upsample_trilinear3d_backward_out_frame<scalar_t, accscalar_t>
            <<<cuda::ATenCeilDiv(num_kernels, static_cast<size_t>(num_threads)),
               num_threads,
               0,
               stream>>>(
                nbatch * channels,
                input_depth,
                input_height,
                input_width,
                output_depth,
                output_height,
                output_width,
                rdepth,
                rheight,
                rwidth,
                align_corners,
                idata,
                odata);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  }
}

} // namespace

Tensor& upsample_trilinear3d_out_cuda(
    Tensor& output,
    const Tensor& input,
    IntArrayRef output_size,
    bool align_corners,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  upsample_trilinear3d_out_cuda_template(
      output, input, output_size, align_corners, scales_d, scales_h, scales_w);
  return output;
}

Tensor upsample_trilinear3d_cuda(
    const Tensor& input,
    IntArrayRef output_size,
    bool align_corners,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  Tensor output = at::empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  upsample_trilinear3d_out_cuda_template(
      output, input, output_size, align_corners, scales_d, scales_h, scales_w);
  return output;
}

Tensor& upsample_trilinear3d_backward_out_cuda(
    Tensor& grad_input,
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    bool align_corners,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  // See Note [Writing Nondeterministic Operations]
  // Nondeterministic because of atomicAdd usage
  globalContext().alertNotDeterministic("upsample_trilinear3d_backward_out_cuda");
  upsample_trilinear3d_backward_out_cuda_template(
      grad_input, grad_output, output_size, input_size, align_corners, scales_d, scales_h, scales_w);
  return grad_input;
}

Tensor upsample_trilinear3d_backward_cuda(
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    bool align_corners,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  // See Note [Writing Nondeterministic Operations]
  // Nondeterministic because of atomicAdd usage
  globalContext().alertNotDeterministic("upsample_trilinear3d_backward_cuda");
  Tensor grad_input = at::empty_like(grad_output, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  upsample_trilinear3d_backward_out_cuda_template(
      grad_input, grad_output, output_size, input_size, align_corners, scales_d, scales_h, scales_w);
  return grad_input;
}

using at::native::upsample::compute_output_size;
using at::native::upsample_cuda::get_scale_value;

Tensor upsample_trilinear3d_cuda(
    const Tensor& input,
    c10::optional<IntArrayRef> output_size,
    bool align_corners,
    c10::optional<ArrayRef<double>> scale_factors) {
  auto output = at::empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  auto osize = compute_output_size(input.sizes(), output_size, scale_factors);
  auto scale_d = get_scale_value(scale_factors, 0);
  auto scale_h = get_scale_value(scale_factors, 1);
  auto scale_w = get_scale_value(scale_factors, 2);
  upsample_trilinear3d_out_cuda_template(output, input, osize, align_corners, scale_d, scale_h, scale_w);
  return output;
}

Tensor upsample_trilinear3d_backward_cuda(
    const Tensor& grad_output,
    c10::optional<IntArrayRef> output_size,
    IntArrayRef input_size,
    bool align_corners,
    c10::optional<ArrayRef<double>> scale_factors) {
  // Nondeterministic because of atomicAdd usage
  globalContext().alertNotDeterministic("upsample_trilinear3d_backward_cuda");
  auto osize = compute_output_size(input_size, output_size, scale_factors);
  auto scale_d = get_scale_value(scale_factors, 0);
  auto scale_h = get_scale_value(scale_factors, 1);
  auto scale_w = get_scale_value(scale_factors, 2);
  auto grad_input = at::empty_like(grad_output, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  upsample_trilinear3d_backward_out_cuda_template(
      grad_input, grad_output, osize, input_size, align_corners, scale_d, scale_h, scale_w);
  return grad_input;
}

} // namespace native
} // namespace at
