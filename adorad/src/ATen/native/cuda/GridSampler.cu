#include <ATen/ATen.h>
#include <ATen/native/cuda/GridSampler.cuh>
#include <ATen/native/cuda/UpSample.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/KernelUtils.h>
#include <c10/macros/Macros.h>

namespace at { namespace native {

using namespace at::cuda::detail;

using at::native::detail::GridSamplerInterpolation;
using at::native::detail::GridSamplerPadding;

namespace {
  template <typename scalar_t, typename index_t>
  C10_LAUNCH_BOUNDS_1(1024)
  __global__ void grid_sampler_2d_kernel(
      const index_t nthreads,
      TensorInfo<scalar_t, index_t> input,
      TensorInfo<scalar_t, index_t> grid,
      TensorInfo<scalar_t, index_t> output,
      const GridSamplerInterpolation interpolation_mode,
      const GridSamplerPadding padding_mode,
      bool align_corners) {
    index_t C = input.sizes[1];
    index_t inp_H = input.sizes[2];
    index_t inp_W = input.sizes[3];
    index_t out_H = grid.sizes[1];
    index_t out_W = grid.sizes[2];
    index_t inp_sN = input.strides[0];
    index_t inp_sC = input.strides[1];
    index_t inp_sH = input.strides[2];
    index_t inp_sW = input.strides[3];
    index_t grid_sN = grid.strides[0];
    index_t grid_sH = grid.strides[1];
    index_t grid_sW = grid.strides[2];
    index_t grid_sCoor = grid.strides[3];
    index_t out_sN = output.strides[0];
    index_t out_sC = output.strides[1];
    index_t out_sH = output.strides[2];
    index_t out_sW = output.strides[3];

    CUDA_KERNEL_LOOP_TYPE(index, nthreads, index_t) {
      const index_t w = index % out_W;
      const index_t h = (index / out_W) % out_H;
      const index_t n = index / (out_H * out_W);
      const index_t grid_offset = n * grid_sN + h * grid_sH + w * grid_sW;

      // get the corresponding input x, y co-ordinates from grid
      scalar_t x = grid.data[grid_offset];
      scalar_t y = grid.data[grid_offset + grid_sCoor];

      scalar_t ix = grid_sampler_compute_source_index(x, inp_W, padding_mode, align_corners);
      scalar_t iy = grid_sampler_compute_source_index(y, inp_H, padding_mode, align_corners);

      if (interpolation_mode == GridSamplerInterpolation::Bilinear) {
        // get NE, NW, SE, SW pixel values from (x, y)
        index_t ix_nw = static_cast<index_t>(::floor(ix));
        index_t iy_nw = static_cast<index_t>(::floor(iy));
        index_t ix_ne = ix_nw + 1;
        index_t iy_ne = iy_nw;
        index_t ix_sw = ix_nw;
        index_t iy_sw = iy_nw + 1;
        index_t ix_se = ix_nw + 1;
        index_t iy_se = iy_nw + 1;

        // get surfaces to each neighbor:
        scalar_t nw = (ix_se - ix)    * (iy_se - iy);
        scalar_t ne = (ix    - ix_sw) * (iy_sw - iy);
        scalar_t sw = (ix_ne - ix)    * (iy    - iy_ne);
        scalar_t se = (ix    - ix_nw) * (iy    - iy_nw);

        // calculate bilinear weighted pixel value and set output pixel
        auto inp_ptr_NC = input.data + n * inp_sN;
        auto out_ptr_NCHW = output.data + n * out_sN + h * out_sH + w * out_sW;
        for (index_t c = 0; c < C; ++c, inp_ptr_NC += inp_sC, out_ptr_NCHW += out_sC) {
          *out_ptr_NCHW = static_cast<scalar_t>(0);
          if (within_bounds_2d(iy_nw, ix_nw, inp_H, inp_W)) {
            *out_ptr_NCHW += inp_ptr_NC[iy_nw * inp_sH + ix_nw * inp_sW] * nw;
          }
          if (within_bounds_2d(iy_ne, ix_ne, inp_H, inp_W)) {
            *out_ptr_NCHW += inp_ptr_NC[iy_ne * inp_sH + ix_ne * inp_sW] * ne;
          }
          if (within_bounds_2d(iy_sw, ix_sw, inp_H, inp_W)) {
            *out_ptr_NCHW += inp_ptr_NC[iy_sw * inp_sH + ix_sw * inp_sW] * sw;
          }
          if (within_bounds_2d(iy_se, ix_se, inp_H, inp_W)) {
            *out_ptr_NCHW += inp_ptr_NC[iy_se * inp_sH + ix_se * inp_sW] * se;
          }
        }
      } else if (interpolation_mode == GridSamplerInterpolation::Nearest) {
        index_t ix_nearest = static_cast<index_t>(::round(ix));
        index_t iy_nearest = static_cast<index_t>(::round(iy));

        // assign nearest neighor pixel value to output pixel
        auto inp_ptr_NC = input.data + n * inp_sN;
        auto out_ptr_NCHW = output.data + n * out_sN + h * out_sH + w * out_sW;
        for (index_t c = 0; c < C; ++c, inp_ptr_NC += inp_sC, out_ptr_NCHW += out_sC) {
          if (within_bounds_2d(iy_nearest, ix_nearest, inp_H, inp_W)) {
            *out_ptr_NCHW = inp_ptr_NC[iy_nearest * inp_sH + ix_nearest * inp_sW];
          } else {
            *out_ptr_NCHW = static_cast<scalar_t>(0);
          }
        }
      } else if (interpolation_mode == GridSamplerInterpolation::Bicubic) {

        ix = grid_sampler_unnormalize(x, inp_W, align_corners);
        iy = grid_sampler_unnormalize(y, inp_H, align_corners);

        scalar_t ix_nw = ::floor(ix);
        scalar_t iy_nw = ::floor(iy);

        const scalar_t tx = ix - ix_nw;
        const scalar_t ty = iy - iy_nw;

        auto inp_ptr_NC = input.data + n * inp_sN;
        auto out_ptr_NCHW = output.data + n * out_sN + h * out_sH + w * out_sW;
        for (index_t c = 0; c < C; ++c, inp_ptr_NC += inp_sC, out_ptr_NCHW += out_sC) {
          scalar_t coefficients[4];

          for (index_t i = 0; i < 4; ++i) {
            coefficients[i] = cubic_interp1d(
              get_value_bounded<scalar_t>(inp_ptr_NC, ix_nw - 1, iy_nw - 1 + i, inp_W, inp_H, inp_sW, inp_sH, padding_mode, align_corners),
              get_value_bounded<scalar_t>(inp_ptr_NC, ix_nw + 0, iy_nw - 1 + i, inp_W, inp_H, inp_sW, inp_sH, padding_mode, align_corners),
              get_value_bounded<scalar_t>(inp_ptr_NC, ix_nw + 1, iy_nw - 1 + i, inp_W, inp_H, inp_sW, inp_sH, padding_mode, align_corners),
              get_value_bounded<scalar_t>(inp_ptr_NC, ix_nw + 2, iy_nw - 1 + i, inp_W, inp_H, inp_sW, inp_sH, padding_mode, align_corners),
              tx);
          }

          *out_ptr_NCHW = cubic_interp1d(
            coefficients[0],
            coefficients[1],
            coefficients[2],
            coefficients[3],
            ty);
        }
      }
    }
  }

  template <typename scalar_t, typename index_t>
  C10_LAUNCH_BOUNDS_1(1024)
  __global__ void grid_sampler_3d_kernel(
      const index_t nthreads,
      TensorInfo<scalar_t, index_t> input,
      TensorInfo<scalar_t, index_t> grid,
      TensorInfo<scalar_t, index_t> output,
      const GridSamplerInterpolation interpolation_mode,
      const GridSamplerPadding padding_mode,
      bool align_corners) {

    index_t C = input.sizes[1];
    index_t inp_D = input.sizes[2];
    index_t inp_H = input.sizes[3];
    index_t inp_W = input.sizes[4];
    index_t out_D = grid.sizes[1];
    index_t out_H = grid.sizes[2];
    index_t out_W = grid.sizes[3];
    index_t inp_sN = input.strides[0];
    index_t inp_sC = input.strides[1];
    index_t inp_sD = input.strides[2];
    index_t inp_sH = input.strides[3];
    index_t inp_sW = input.strides[4];
    index_t grid_sN = grid.strides[0];
    index_t grid_sD = grid.strides[1];
    index_t grid_sH = grid.strides[2];
    index_t grid_sW = grid.strides[3];
    index_t grid_sCoor = grid.strides[4];
    index_t out_sN = output.strides[0];
    index_t out_sC = output.strides[1];
    index_t out_sD = output.strides[2];
    index_t out_sH = output.strides[3];
    index_t out_sW = output.strides[4];

    CUDA_KERNEL_LOOP_TYPE(index, nthreads, index_t) {
      const index_t w = index % out_W;
      const index_t h = (index / out_W) % out_H;
      const index_t d = (index / (out_H * out_W)) % out_D;
      const index_t n = index / (out_D * out_H * out_W);
      const index_t grid_offset = n * grid_sN + d * grid_sD + h * grid_sH + w * grid_sW;

      // get the corresponding input x, y, z co-ordinates from grid
      scalar_t ix = grid.data[grid_offset];
      scalar_t iy = grid.data[grid_offset + grid_sCoor];
      scalar_t iz = grid.data[grid_offset + 2 * grid_sCoor];

      ix = grid_sampler_compute_source_index(ix, inp_W, padding_mode, align_corners);
      iy = grid_sampler_compute_source_index(iy, inp_H, padding_mode, align_corners);
      iz = grid_sampler_compute_source_index(iz, inp_D, padding_mode, align_corners);

      if (interpolation_mode == GridSamplerInterpolation::Bilinear) {
        // get corner pixel values from (x, y, z)
        // for 4d, we used north-east-south-west
        // for 5d, we add top-bottom
        index_t ix_tnw = static_cast<index_t>(::floor(ix));
        index_t iy_tnw = static_cast<index_t>(::floor(iy));
        index_t iz_tnw = static_cast<index_t>(::floor(iz));

        index_t ix_tne = ix_tnw + 1;
        index_t iy_tne = iy_tnw;
        index_t iz_tne = iz_tnw;

        index_t ix_tsw = ix_tnw;
        index_t iy_tsw = iy_tnw + 1;
        index_t iz_tsw = iz_tnw;

        index_t ix_tse = ix_tnw + 1;
        index_t iy_tse = iy_tnw + 1;
        index_t iz_tse = iz_tnw;

        index_t ix_bnw = ix_tnw;
        index_t iy_bnw = iy_tnw;
        index_t iz_bnw = iz_tnw + 1;

        index_t ix_bne = ix_tnw + 1;
        index_t iy_bne = iy_tnw;
        index_t iz_bne = iz_tnw + 1;

        index_t ix_bsw = ix_tnw;
        index_t iy_bsw = iy_tnw + 1;
        index_t iz_bsw = iz_tnw + 1;

        index_t ix_bse = ix_tnw + 1;
        index_t iy_bse = iy_tnw + 1;
        index_t iz_bse = iz_tnw + 1;

        // get surfaces to each neighbor:
        scalar_t tnw = (ix_bse - ix)    * (iy_bse - iy)    * (iz_bse - iz);
        scalar_t tne = (ix    - ix_bsw) * (iy_bsw - iy)    * (iz_bsw - iz);
        scalar_t tsw = (ix_bne - ix)    * (iy    - iy_bne) * (iz_bne - iz);
        scalar_t tse = (ix    - ix_bnw) * (iy    - iy_bnw) * (iz_bnw - iz);
        scalar_t bnw = (ix_tse - ix)    * (iy_tse - iy)    * (iz - iz_tse);
        scalar_t bne = (ix    - ix_tsw) * (iy_tsw - iy)    * (iz - iz_tsw);
        scalar_t bsw = (ix_tne - ix)    * (iy    - iy_tne) * (iz - iz_tne);
        scalar_t bse = (ix    - ix_tnw) * (iy    - iy_tnw) * (iz - iz_tnw);

        auto inp_ptr_NC = input.data + n * inp_sN;
        auto out_ptr_NCDHW = output.data + n * out_sN + d * out_sD + h * out_sH + w * out_sW;
        for (index_t c = 0; c < C; ++c, inp_ptr_NC += inp_sC, out_ptr_NCDHW += out_sC) {
          //   (c, iz_tnw, iy_tnw, ix_tnw) * tnw + (c, iz_tne, iy_tne, ix_tne) * tne
          // + (c, iz_tsw, iy_tsw, ix_tsw) * tsw + (c, iz_tse, iy_tse, ix_tse) * tse
          // + (c, iz_bnw, iy_bnw, ix_bnw) * bnw + (c, iz_bne, iy_bne, ix_bne) * bne
          // + (c, iz_bsw, iy_bsw, ix_bsw) * bsw + (c, iz_bse, iy_bse, ix_bse) * bse
          *out_ptr_NCDHW = static_cast<scalar_t>(0);
          if (within_bounds_3d(iz_tnw, iy_tnw, ix_tnw, inp_D, inp_H, inp_W)) {
            *out_ptr_NCDHW += inp_ptr_NC[iz_tnw * inp_sD + iy_tnw * inp_sH + ix_tnw * inp_sW] * tnw;
          }
          if (within_bounds_3d(iz_tne, iy_tne, ix_tne, inp_D, inp_H, inp_W)) {
            *out_ptr_NCDHW += inp_ptr_NC[iz_tne * inp_sD + iy_tne * inp_sH + ix_tne * inp_sW] * tne;
          }
          if (within_bounds_3d(iz_tsw, iy_tsw, ix_tsw, inp_D, inp_H, inp_W)) {
            *out_ptr_NCDHW += inp_ptr_NC[iz_tsw * inp_sD + iy_tsw * inp_sH + ix_tsw * inp_sW] * tsw;
          }
          if (within_bounds_3d(iz_tse, iy_tse, ix_tse, inp_D, inp_H, inp_W)) {
            *out_ptr_NCDHW += inp_ptr_NC[iz_tse * inp_sD + iy_tse * inp_sH + ix_tse * inp_sW] * tse;
          }
          if (within_bounds_3d(iz_bnw, iy_bnw, ix_bnw, inp_D, inp_H, inp_W)) {
            *out_ptr_NCDHW += inp_ptr_NC[iz_bnw * inp_sD + iy_bnw * inp_sH + ix_bnw * inp_sW] * bnw;
          }
          if (within_bounds_3d(iz_bne, iy_bne, ix_bne, inp_D, inp_H, inp_W)) {
            *out_ptr_NCDHW += inp_ptr_NC[iz_bne * inp_sD + iy_bne * inp_sH + ix_bne * inp_sW] * bne;
          }
          if (within_bounds_3d(iz_bsw, iy_bsw, ix_bsw, inp_D, inp_H, inp_W)) {
            *out_ptr_NCDHW += inp_ptr_NC[iz_bsw * inp_sD + iy_bsw * inp_sH + ix_bsw * inp_sW] * bsw;
          }
          if (within_bounds_3d(iz_bse, iy_bse, ix_bse, inp_D, inp_H, inp_W)) {
            *out_ptr_NCDHW += inp_ptr_NC[iz_bse * inp_sD + iy_bse * inp_sH + ix_bse * inp_sW] * bse;
          }
        }
      } else if (interpolation_mode == GridSamplerInterpolation::Nearest) {
        index_t ix_nearest = static_cast<index_t>(::round(ix));
        index_t iy_nearest = static_cast<index_t>(::round(iy));
        index_t iz_nearest = static_cast<index_t>(::round(iz));

        // assign nearest neighor pixel value to output pixel
        auto inp_ptr_NC = input.data + n * inp_sN;
        auto out_ptr_NCDHW = output.data + n * out_sN + d * out_sD + h * out_sH + w * out_sW;
        for (index_t c = 0; c < C; ++c, inp_ptr_NC += inp_sC, out_ptr_NCDHW += out_sC) {
          if (within_bounds_3d(iz_nearest, iy_nearest, ix_nearest, inp_D, inp_H, inp_W)) {
            *out_ptr_NCDHW = inp_ptr_NC[iz_nearest * inp_sD + iy_nearest * inp_sH + ix_nearest * inp_sW];
          } else {
            *out_ptr_NCDHW = static_cast<scalar_t>(0);
          }
        }
      }
    }
  }

  template <typename scalar_t, typename index_t>
  C10_LAUNCH_BOUNDS_1(1024)
  __global__ void grid_sampler_2d_backward_kernel(
      const index_t nthreads,
      TensorInfo<scalar_t, index_t> grad_output,
      TensorInfo<scalar_t, index_t> input,
      TensorInfo<scalar_t, index_t> grid,
      TensorInfo<scalar_t, index_t> grad_input,  // initialized to zeros
      TensorInfo<scalar_t, index_t> grad_grid,   // initialized to empty
      const GridSamplerInterpolation interpolation_mode,
      const GridSamplerPadding padding_mode,
      bool align_corners) {

    index_t C = input.sizes[1];
    index_t inp_H = input.sizes[2];
    index_t inp_W = input.sizes[3];
    index_t out_H = grid.sizes[1];
    index_t out_W = grid.sizes[2];
    index_t inp_sN = input.strides[0];
    index_t inp_sC = input.strides[1];
    index_t inp_sH = input.strides[2];
    index_t inp_sW = input.strides[3];
    index_t grid_sN = grid.strides[0];
    index_t grid_sH = grid.strides[1];
    index_t grid_sW = grid.strides[2];
    index_t grid_sCoor = grid.strides[3];
    index_t gOut_sN = grad_output.strides[0];
    index_t gOut_sC = grad_output.strides[1];
    index_t gOut_sH = grad_output.strides[2];
    index_t gOut_sW = grad_output.strides[3];
    index_t gInp_sN = grad_input.strides[0];
    index_t gInp_sC = grad_input.strides[1];
    index_t gInp_sH = grad_input.strides[2];
    index_t gInp_sW = grad_input.strides[3];
    index_t gGrid_sW = grad_grid.strides[2];

    CUDA_KERNEL_LOOP_TYPE(index, nthreads, index_t) {
      const index_t w = index % out_W;
      const index_t h = (index / out_W) % out_H;
      const index_t n = index / (out_H * out_W);
      const auto grid_offset = n * grid_sN + h * grid_sH + w * grid_sW;

      // get the corresponding input x, y co-ordinates from grid
      scalar_t x = grid.data[grid_offset];
      scalar_t y = grid.data[grid_offset + grid_sCoor];

      // multipliers for gradients on ix and iy
      scalar_t gix_mult, giy_mult;
      scalar_t ix = grid_sampler_compute_source_index_set_grad(x, inp_W, padding_mode, align_corners, &gix_mult);
      scalar_t iy = grid_sampler_compute_source_index_set_grad(y, inp_H, padding_mode, align_corners, &giy_mult);

      if (interpolation_mode == GridSamplerInterpolation::Bilinear) {
        // get NE, NW, SE, SW pixel values from (x, y)
        index_t ix_nw = static_cast<index_t>(::floor(ix));
        index_t iy_nw = static_cast<index_t>(::floor(iy));
        index_t ix_ne = ix_nw + 1;
        index_t iy_ne = iy_nw;
        index_t ix_sw = ix_nw;
        index_t iy_sw = iy_nw + 1;
        index_t ix_se = ix_nw + 1;
        index_t iy_se = iy_nw + 1;

        // get surfaces to each neighbor:
        scalar_t nw = (ix_se - ix)    * (iy_se - iy);
        scalar_t ne = (ix    - ix_sw) * (iy_sw - iy);
        scalar_t sw = (ix_ne - ix)    * (iy    - iy_ne);
        scalar_t se = (ix    - ix_nw) * (iy    - iy_nw);

        scalar_t gix = static_cast<scalar_t>(0), giy = static_cast<scalar_t>(0);
        scalar_t *gOut_ptr_NCHW = grad_output.data + n * gOut_sN + h * gOut_sH + w * gOut_sW;
        scalar_t *gInp_ptr_NC = grad_input.data + n * gInp_sN;
        scalar_t *inp_ptr_NC = input.data + n * inp_sN;
        for (index_t c = 0; c < C; ++c, inp_ptr_NC += inp_sC, gInp_ptr_NC += gInp_sC, gOut_ptr_NCHW += gOut_sC) {
          scalar_t gOut = *gOut_ptr_NCHW;

          // calculate and set grad_input
          safe_add_2d(gInp_ptr_NC, iy_nw, ix_nw, gInp_sH, gInp_sW, inp_H, inp_W, nw * gOut);
          safe_add_2d(gInp_ptr_NC, iy_ne, ix_ne, gInp_sH, gInp_sW, inp_H, inp_W, ne * gOut);
          safe_add_2d(gInp_ptr_NC, iy_sw, ix_sw, gInp_sH, gInp_sW, inp_H, inp_W, sw * gOut);
          safe_add_2d(gInp_ptr_NC, iy_se, ix_se, gInp_sH, gInp_sW, inp_H, inp_W, se * gOut);

          // calculate grad_grid
          if (within_bounds_2d(iy_nw, ix_nw, inp_H, inp_W)) {
            scalar_t nw_val = inp_ptr_NC[iy_nw * inp_sH + ix_nw * inp_sW];
            gix -= nw_val * (iy_se - iy) * gOut;
            giy -= nw_val * (ix_se - ix) * gOut;
          }
          if (within_bounds_2d(iy_ne, ix_ne, inp_H, inp_W)) {
            scalar_t ne_val = inp_ptr_NC[iy_ne * inp_sH + ix_ne * inp_sW];
            gix += ne_val * (iy_sw - iy) * gOut;
            giy -= ne_val * (ix - ix_sw) * gOut;
          }
          if (within_bounds_2d(iy_sw, ix_sw, inp_H, inp_W)) {
            scalar_t sw_val = inp_ptr_NC[iy_sw * inp_sH + ix_sw * inp_sW];
            gix -= sw_val * (iy - iy_ne) * gOut;
            giy += sw_val * (ix_ne - ix) * gOut;
          }
          if (within_bounds_2d(iy_se, ix_se, inp_H, inp_W)) {
            scalar_t se_val = inp_ptr_NC[iy_se * inp_sH + ix_se * inp_sW];
            gix += se_val * (iy - iy_nw) * gOut;
            giy += se_val * (ix - ix_nw) * gOut;
          }
        }

        // assuming grad_grid is contiguous
        // thus we can
        //   1. use index with gGrid_sW to directly compute gGrid_ptr_NHW
        //   2. directly assign to gGrid_ptr_NHW[0], gGrid_ptr_NHW[1]
        scalar_t *gGrid_ptr_NHW = grad_grid.data + index * gGrid_sW;
        gGrid_ptr_NHW[0] = gix_mult * gix;
        gGrid_ptr_NHW[1] = giy_mult * giy;
      } else if (interpolation_mode == GridSamplerInterpolation::Nearest) {
        index_t ix_nearest = static_cast<index_t>(::round(ix));
        index_t iy_nearest = static_cast<index_t>(::round(iy));

        // assign nearest neighor pixel value to output pixel
        scalar_t *gOut_ptr_NCHW = grad_output.data + n * gOut_sN + h * gOut_sH + w * gOut_sW;
        scalar_t *gInp_ptr_NC = grad_input.data + n * gInp_sN;
        for (index_t c = 0; c < C; ++c, gInp_ptr_NC += gInp_sC, gOut_ptr_NCHW += gOut_sC) {
          // calculate and set grad_input
          safe_add_2d(gInp_ptr_NC, iy_nearest, ix_nearest, gInp_sH, gInp_sW, inp_H, inp_W, *gOut_ptr_NCHW);
        }

        // assuming grad_grid is contiguous
        // thus we can
        //   1. use index with gGrid_sW to directly compute gGrid_ptr_NHW
        //   2. directly assign to gGrid_ptr_NHW[0], gGrid_ptr_NHW[1]
        scalar_t *gGrid_ptr_NHW = grad_grid.data + index * gGrid_sW;
        gGrid_ptr_NHW[0] = static_cast<scalar_t>(0);
        gGrid_ptr_NHW[1] = static_cast<scalar_t>(0);
      } else if (interpolation_mode == GridSamplerInterpolation::Bicubic) {

        ix = grid_sampler_unnormalize_set_grad(x, inp_W, align_corners, &gix_mult);
        iy = grid_sampler_unnormalize_set_grad(y, inp_H, align_corners, &giy_mult);

        scalar_t ix_nw = ::floor(ix);
        scalar_t iy_nw = ::floor(iy);

        const scalar_t tx = ix - ix_nw;
        const scalar_t ty = iy - iy_nw;

        scalar_t x_coeffs[4];
        scalar_t y_coeffs[4];
        scalar_t x_coeffs_grad[4];
        scalar_t y_coeffs_grad[4];

        get_cubic_upsampling_coefficients<scalar_t>(x_coeffs, tx);
        get_cubic_upsampling_coefficients<scalar_t>(y_coeffs, ty);
        get_cubic_coefficients_grad<scalar_t>(x_coeffs_grad, tx);
        get_cubic_coefficients_grad<scalar_t>(y_coeffs_grad, ty);

        scalar_t gix = static_cast<scalar_t>(0);
        scalar_t giy = static_cast<scalar_t>(0);

        scalar_t *gOut_ptr_NCHW = grad_output.data + n * gOut_sN + h * gOut_sH + w * gOut_sW;
        scalar_t *gInp_ptr_NC = grad_input.data + n * gInp_sN;
        scalar_t *inp_ptr_NC = input.data + n * inp_sN;

        for (index_t c = 0; c < C; ++c, gOut_ptr_NCHW += gOut_sC, gInp_ptr_NC += gInp_sC, inp_ptr_NC+= inp_sC) {
          scalar_t gOut = *gOut_ptr_NCHW;

          for (index_t i = 0; i < 4; ++i) {
            for (index_t j = 0; j < 4; ++j) {

              // set input gradient
              add_value_bounded<scalar_t>(gInp_ptr_NC, ix_nw - 1 + i, iy_nw - 1 + j, inp_W, inp_H,
                gInp_sW, gInp_sH, gOut * x_coeffs[i] * y_coeffs[j], padding_mode, align_corners);

              // set grid gradient
              scalar_t val = get_value_bounded<scalar_t>(inp_ptr_NC, ix_nw - 1 + i, iy_nw - 1 + j,
                inp_W, inp_H, inp_sW, inp_sH, padding_mode, align_corners);

              gix -= val * x_coeffs_grad[i] * y_coeffs[j] * gOut;
              giy -= val * y_coeffs_grad[j] * x_coeffs[i] * gOut;
            }
          }
        }

        scalar_t *gGrid_ptr_NHW = grad_grid.data + index * gGrid_sW;
        gGrid_ptr_NHW[0] = gix_mult * gix;
        gGrid_ptr_NHW[1] = giy_mult * giy;
      }
    }
  }

  template <typename scalar_t, typename index_t>
  C10_LAUNCH_BOUNDS_1(1024)
  __global__ void grid_sampler_3d_backward_kernel(
      const index_t nthreads,
      TensorInfo<scalar_t, index_t> grad_output,
      TensorInfo<scalar_t, index_t> input,
      TensorInfo<scalar_t, index_t> grid,
      TensorInfo<scalar_t, index_t> grad_input,  // initialized to zeros
      TensorInfo<scalar_t, index_t> grad_grid,   // initialized to empty
      const GridSamplerInterpolation interpolation_mode,
      const GridSamplerPadding padding_mode,
      bool align_corners) {

    index_t C = input.sizes[1];
    index_t inp_D = input.sizes[2];
    index_t inp_H = input.sizes[3];
    index_t inp_W = input.sizes[4];
    index_t out_D = grid.sizes[1];
    index_t out_H = grid.sizes[2];
    index_t out_W = grid.sizes[3];
    index_t inp_sN = input.strides[0];
    index_t inp_sC = input.strides[1];
    index_t inp_sD = input.strides[2];
    index_t inp_sH = input.strides[3];
    index_t inp_sW = input.strides[4];
    index_t grid_sN = grid.strides[0];
    index_t grid_sD = grid.strides[1];
    index_t grid_sH = grid.strides[2];
    index_t grid_sW = grid.strides[3];
    index_t grid_sCoor = grid.strides[4];
    index_t gOut_sN = grad_output.strides[0];
    index_t gOut_sC = grad_output.strides[1];
    index_t gOut_sD = grad_output.strides[2];
    index_t gOut_sH = grad_output.strides[3];
    index_t gOut_sW = grad_output.strides[4];
    index_t gInp_sN = grad_input.strides[0];
    index_t gInp_sC = grad_input.strides[1];
    index_t gInp_sD = grad_input.strides[2];
    index_t gInp_sH = grad_input.strides[3];
    index_t gInp_sW = grad_input.strides[4];
    index_t gGrid_sW = grad_grid.strides[3];

    CUDA_KERNEL_LOOP_TYPE(index, nthreads, index_t) {
      const index_t w = index % out_W;
      const index_t h = (index / out_W) % out_H;
      const index_t d = (index / (out_H * out_W)) % out_D;
      const index_t n = index / (out_D * out_H * out_W);
      const auto grid_offset = n * grid_sN + d * grid_sD + h * grid_sH + w * grid_sW;

      // get the corresponding input x, y, z co-ordinates from grid
      scalar_t ix = grid.data[grid_offset];
      scalar_t iy = grid.data[grid_offset + grid_sCoor];
      scalar_t iz = grid.data[grid_offset + 2 * grid_sCoor];

      // multipliers for gradients on ix, iy, and iz
      scalar_t gix_mult, giy_mult, giz_mult;
      ix = grid_sampler_compute_source_index_set_grad(ix, inp_W, padding_mode, align_corners, &gix_mult);
      iy = grid_sampler_compute_source_index_set_grad(iy, inp_H, padding_mode, align_corners, &giy_mult);
      iz = grid_sampler_compute_source_index_set_grad(iz, inp_D, padding_mode, align_corners, &giz_mult);

      if (interpolation_mode == GridSamplerInterpolation::Bilinear) {
        // get corner pixel values from (x, y, z)
        // for 4d, we used north-east-south-west
        // for 5d, we add top-bottom
        index_t ix_tnw = static_cast<index_t>(::floor(ix));
        index_t iy_tnw = static_cast<index_t>(::floor(iy));
        index_t iz_tnw = static_cast<index_t>(::floor(iz));

        index_t ix_tne = ix_tnw + 1;
        index_t iy_tne = iy_tnw;
        index_t iz_tne = iz_tnw;

        index_t ix_tsw = ix_tnw;
        index_t iy_tsw = iy_tnw + 1;
        index_t iz_tsw = iz_tnw;

        index_t ix_tse = ix_tnw + 1;
        index_t iy_tse = iy_tnw + 1;
        index_t iz_tse = iz_tnw;

        index_t ix_bnw = ix_tnw;
        index_t iy_bnw = iy_tnw;
        index_t iz_bnw = iz_tnw + 1;

        index_t ix_bne = ix_tnw + 1;
        index_t iy_bne = iy_tnw;
        index_t iz_bne = iz_tnw + 1;

        index_t ix_bsw = ix_tnw;
        index_t iy_bsw = iy_tnw + 1;
        index_t iz_bsw = iz_tnw + 1;

        index_t ix_bse = ix_tnw + 1;
        index_t iy_bse = iy_tnw + 1;
        index_t iz_bse = iz_tnw + 1;

        // get surfaces to each neighbor:
        scalar_t tnw = (ix_bse - ix)    * (iy_bse - iy)    * (iz_bse - iz);
        scalar_t tne = (ix    - ix_bsw) * (iy_bsw - iy)    * (iz_bsw - iz);
        scalar_t tsw = (ix_bne - ix)    * (iy    - iy_bne) * (iz_bne - iz);
        scalar_t tse = (ix    - ix_bnw) * (iy    - iy_bnw) * (iz_bnw - iz);
        scalar_t bnw = (ix_tse - ix)    * (iy_tse - iy)    * (iz - iz_tse);
        scalar_t bne = (ix    - ix_tsw) * (iy_tsw - iy)    * (iz - iz_tsw);
        scalar_t bsw = (ix_tne - ix)    * (iy    - iy_tne) * (iz - iz_tne);
        scalar_t bse = (ix    - ix_tnw) * (iy    - iy_tnw) * (iz - iz_tnw);

        scalar_t gix = static_cast<scalar_t>(0), giy = static_cast<scalar_t>(0), giz = static_cast<scalar_t>(0);
        scalar_t *gOut_ptr_NCDHW = grad_output.data + n * gOut_sN + d * gOut_sD + h * gOut_sH + w * gOut_sW;
        scalar_t *gInp_ptr_NC = grad_input.data + n * gInp_sN;
        scalar_t *inp_ptr_NC = input.data + n * inp_sN;
        // calculate bilinear weighted pixel value and set output pixel
        for (index_t c = 0; c < C; ++c, gOut_ptr_NCDHW += gOut_sC, gInp_ptr_NC += gInp_sC, inp_ptr_NC += inp_sC) {
          scalar_t gOut = *gOut_ptr_NCDHW;

          // calculate and set grad_input
          safe_add_3d(gInp_ptr_NC, iz_tnw, iy_tnw, ix_tnw, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, tnw * gOut);
          safe_add_3d(gInp_ptr_NC, iz_tne, iy_tne, ix_tne, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, tne * gOut);
          safe_add_3d(gInp_ptr_NC, iz_tsw, iy_tsw, ix_tsw, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, tsw * gOut);
          safe_add_3d(gInp_ptr_NC, iz_tse, iy_tse, ix_tse, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, tse * gOut);
          safe_add_3d(gInp_ptr_NC, iz_bnw, iy_bnw, ix_bnw, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, bnw * gOut);
          safe_add_3d(gInp_ptr_NC, iz_bne, iy_bne, ix_bne, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, bne * gOut);
          safe_add_3d(gInp_ptr_NC, iz_bsw, iy_bsw, ix_bsw, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, bsw * gOut);
          safe_add_3d(gInp_ptr_NC, iz_bse, iy_bse, ix_bse, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, bse * gOut);

          // calculate grad_grid
          if (within_bounds_3d(iz_tnw, iy_tnw, ix_tnw, inp_D, inp_H, inp_W)) {
            scalar_t tnw_val = inp_ptr_NC[iz_tnw * inp_sD + iy_tnw * inp_sH + ix_tnw * inp_sW];
            gix -= tnw_val * (iy_bse - iy)    * (iz_bse - iz)    * gOut;
            giy -= tnw_val * (ix_bse - ix)    * (iz_bse - iz)    * gOut;
            giz -= tnw_val * (ix_bse - ix)    * (iy_bse - iy)    * gOut;
          }
          if (within_bounds_3d(iz_tne, iy_tne, ix_tne, inp_D, inp_H, inp_W)) {
            scalar_t tne_val = inp_ptr_NC[iz_tne * inp_sD + iy_tne * inp_sH + ix_tne * inp_sW];
            gix += tne_val * (iy_bsw - iy)    * (iz_bsw - iz)    * gOut;
            giy -= tne_val * (ix    - ix_bsw) * (iz_bsw - iz)    * gOut;
            giz -= tne_val * (ix    - ix_bsw) * (iy_bsw - iy)    * gOut;
          }
          if (within_bounds_3d(iz_tsw, iy_tsw, ix_tsw, inp_D, inp_H, inp_W)) {
            scalar_t tsw_val = inp_ptr_NC[iz_tsw * inp_sD + iy_tsw * inp_sH + ix_tsw * inp_sW];
            gix -= tsw_val * (iy - iy_bne)    * (iz_bne - iz)    * gOut;
            giy += tsw_val * (ix_bne - ix)    * (iz_bne - iz)    * gOut;
            giz -= tsw_val * (ix_bne - ix)    * (iy    - iy_bne) * gOut;
          }
          if (within_bounds_3d(iz_tse, iy_tse, ix_tse, inp_D, inp_H, inp_W)) {
            scalar_t tse_val = inp_ptr_NC[iz_tse * inp_sD + iy_tse * inp_sH + ix_tse * inp_sW];
            gix += tse_val * (iy - iy_bnw)    * (iz_bnw - iz)    * gOut;
            giy += tse_val * (ix    - ix_bnw) * (iz_bnw - iz)    * gOut;
            giz -= tse_val * (ix    - ix_bnw) * (iy    - iy_bnw) * gOut;
          }
          if (within_bounds_3d(iz_bnw, iy_bnw, ix_bnw, inp_D, inp_H, inp_W)) {
            scalar_t bnw_val = inp_ptr_NC[iz_bnw * inp_sD + iy_bnw * inp_sH + ix_bnw * inp_sW];
            gix -= bnw_val * (iy_tse - iy)    * (iz - iz_tse)    * gOut;
            giy -= bnw_val * (ix_tse - ix)    * (iz - iz_tse)    * gOut;
            giz += bnw_val * (ix_tse - ix)    * (iy_tse - iy)    * gOut;
          }
          if (within_bounds_3d(iz_bne, iy_bne, ix_bne, inp_D, inp_H, inp_W)) {
            scalar_t bne_val = inp_ptr_NC[iz_bne * inp_sD + iy_bne * inp_sH + ix_bne * inp_sW];
            gix += bne_val * (iy_tsw - iy)    * (iz - iz_tsw)    * gOut;
            giy -= bne_val * (ix    - ix_tsw) * (iz - iz_tsw)    * gOut;
            giz += bne_val * (ix    - ix_tsw) * (iy_tsw - iy)    * gOut;
          }
          if (within_bounds_3d(iz_bsw, iy_bsw, ix_bsw, inp_D, inp_H, inp_W)) {
            scalar_t bsw_val = inp_ptr_NC[iz_bsw * inp_sD + iy_bsw * inp_sH + ix_bsw * inp_sW];
            gix -= bsw_val * (iy - iy_tne)    * (iz - iz_tne)    * gOut;
            giy += bsw_val * (ix_tne - ix)    * (iz - iz_tne)    * gOut;
            giz += bsw_val * (ix_tne - ix)    * (iy    - iy_tne) * gOut;
          }
          if (within_bounds_3d(iz_bse, iy_bse, ix_bse, inp_D, inp_H, inp_W)) {
            scalar_t bse_val = inp_ptr_NC[iz_bse * inp_sD + iy_bse * inp_sH + ix_bse * inp_sW];
            gix += bse_val * (iy - iy_tnw)    * (iz - iz_tnw)    * gOut;
            giy += bse_val * (ix    - ix_tnw) * (iz - iz_tnw)    * gOut;
            giz += bse_val * (ix    - ix_tnw) * (iy    - iy_tnw) * gOut;
          }
        }

        // assuming grad_grid is contiguous
        // thus we can
        //   1. use index with gGrid_sW to directly compute gGrid_ptr_NDHW
        //   2. directly assign to gGrid_ptr_NDHW[0], gGrid_ptr_NDHW[1], gGrid_ptr_NDHW[2]
        scalar_t *gGrid_ptr_NDHW = grad_grid.data + index * gGrid_sW;
        gGrid_ptr_NDHW[0] = gix_mult * gix;
        gGrid_ptr_NDHW[1] = giy_mult * giy;
        gGrid_ptr_NDHW[2] = giz_mult * giz;
      } else if (interpolation_mode == GridSamplerInterpolation::Nearest) {
        auto ix_nearest = static_cast<index_t>(::round(ix));
        auto iy_nearest = static_cast<index_t>(::round(iy));
        auto iz_nearest = static_cast<index_t>(::round(iz));

        // assign nearest neighor pixel value to output pixel
        scalar_t *gOut_ptr_NCDHW = grad_output.data + n * gOut_sN + d * gOut_sD + h * gOut_sH + w * gOut_sW;
        scalar_t *gInp_ptr_NC = grad_input.data + n * gInp_sN;
        for (index_t c = 0; c < C; ++c, gOut_ptr_NCDHW += gOut_sC, gInp_ptr_NC += gInp_sC) {
          // calculate and set grad_input
          safe_add_3d(gInp_ptr_NC, iz_nearest, iy_nearest, ix_nearest,
                      gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, *gOut_ptr_NCDHW);
        }

        // assuming grad_grid is contiguous
        // thus we can
        //   1. use index with gGrid_sW to directly compute gGrid_ptr_NDHW
        //   2. directly assign to gGrid_ptr_NDHW[0], gGrid_ptr_NDHW[1], gGrid_ptr_NDHW[2]
        scalar_t *gGrid_ptr_NDHW = grad_grid.data + index * gGrid_sW;
        gGrid_ptr_NDHW[0] = static_cast<scalar_t>(0);
        gGrid_ptr_NDHW[1] = static_cast<scalar_t>(0);
        gGrid_ptr_NDHW[2] = static_cast<scalar_t>(0);
      }
    }
  }
}  // namespace

// No shape checking needed here. See # NOTE [ grid_sampler Native Functions ].
Tensor grid_sampler_2d_cuda(const Tensor& input, const Tensor& grid,
                            int64_t interpolation_mode, int64_t padding_mode,
                            bool align_corners) {
  auto N = input.size(0);
  auto C = input.size(1);
  auto H = grid.size(1);
  auto W = grid.size(2);
  auto output = at::empty({N, C, H, W}, input.options());
  int64_t count = N * H * W;
  if (count > 0) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "grid_sampler_2d_cuda", [&] {
      if (canUse32BitIndexMath(input) && canUse32BitIndexMath(grid) &&
          canUse32BitIndexMath(output)) {
        grid_sampler_2d_kernel<scalar_t>
          <<<GET_BLOCKS(count), CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
            static_cast<int>(count),
            getTensorInfo<scalar_t, int>(input),
            getTensorInfo<scalar_t, int>(grid),
            getTensorInfo<scalar_t, int>(output),
            static_cast<GridSamplerInterpolation>(interpolation_mode),
            static_cast<GridSamplerPadding>(padding_mode),
            align_corners);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      } else {
        grid_sampler_2d_kernel<scalar_t>
          <<<GET_BLOCKS(count), CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
            count,
            getTensorInfo<scalar_t, int64_t>(input),
            getTensorInfo<scalar_t, int64_t>(grid),
            getTensorInfo<scalar_t, int64_t>(output),
            static_cast<GridSamplerInterpolation>(interpolation_mode),
            static_cast<GridSamplerPadding>(padding_mode),
            align_corners);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }
    });
  }
  return output;
}

// No shape checking needed here. See # NOTE [ grid_sampler Native Functions ].
Tensor grid_sampler_3d_cuda(const Tensor& input, const Tensor& grid,
                            int64_t interpolation_mode, int64_t padding_mode,
                            bool align_corners) {
  auto N = input.size(0);
  auto D = grid.size(1);
  auto H = grid.size(2);
  auto W = grid.size(3);
  auto output = at::empty({N, input.size(1), D, H, W}, input.options());
  int64_t count = N * D * H * W;
  if (count > 0) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "grid_sampler_2d_cuda", [&] {
      if (canUse32BitIndexMath(input) && canUse32BitIndexMath(grid) &&
          canUse32BitIndexMath(output)) {
        grid_sampler_3d_kernel<scalar_t>
          <<<GET_BLOCKS(count), CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
            static_cast<int>(count),
            getTensorInfo<scalar_t, int>(input),
            getTensorInfo<scalar_t, int>(grid),
            getTensorInfo<scalar_t, int>(output),
            static_cast<GridSamplerInterpolation>(interpolation_mode),
            static_cast<GridSamplerPadding>(padding_mode),
            align_corners);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      } else {
        grid_sampler_3d_kernel<scalar_t>
          <<<GET_BLOCKS(count), CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
            count,
            getTensorInfo<scalar_t, int64_t>(input),
            getTensorInfo<scalar_t, int64_t>(grid),
            getTensorInfo<scalar_t, int64_t>(output),
            static_cast<GridSamplerInterpolation>(interpolation_mode),
            static_cast<GridSamplerPadding>(padding_mode),
            align_corners);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }
    });
  }
  return output;
}

// No shape checking needed here. See # NOTE [ grid_sampler Native Functions ].
std::tuple<Tensor, Tensor>
grid_sampler_2d_backward_cuda(const Tensor& grad_output, const Tensor& input,
                              const Tensor& grid, int64_t interpolation_mode,
                              int64_t padding_mode, bool align_corners) {
  // See Note [Writing Nondeterministic Operations]
  // Nondeterministic because of atomicAdd usage
  globalContext().alertNotDeterministic("grid_sampler_2d_backward_cuda");
  auto N = input.size(0);
  auto H = grid.size(1);
  auto W = grid.size(2);
  auto grad_input = at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  auto grad_grid = at::empty_like(grid, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  int64_t count = N * H * W;
  if (count > 0) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "grid_sampler_2d_backward_cuda", [&] {
      if (canUse32BitIndexMath(input) && canUse32BitIndexMath(grid) &&
          canUse32BitIndexMath(grad_output)) {
        grid_sampler_2d_backward_kernel<scalar_t>
          <<<GET_BLOCKS(count), CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
            static_cast<int>(count),
            getTensorInfo<scalar_t, int>(grad_output),
            getTensorInfo<scalar_t, int>(input),
            getTensorInfo<scalar_t, int>(grid),
            getTensorInfo<scalar_t, int>(grad_input),
            getTensorInfo<scalar_t, int>(grad_grid),
            static_cast<GridSamplerInterpolation>(interpolation_mode),
            static_cast<GridSamplerPadding>(padding_mode),
            align_corners);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      } else {
        grid_sampler_2d_backward_kernel<scalar_t>
          <<<GET_BLOCKS(count), CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
            count,
            getTensorInfo<scalar_t, int64_t>(grad_output),
            getTensorInfo<scalar_t, int64_t>(input),
            getTensorInfo<scalar_t, int64_t>(grid),
            getTensorInfo<scalar_t, int64_t>(grad_input),
            getTensorInfo<scalar_t, int64_t>(grad_grid),
            static_cast<GridSamplerInterpolation>(interpolation_mode),
            static_cast<GridSamplerPadding>(padding_mode),
            align_corners);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }
    });
  }
  return std::make_tuple(grad_input, grad_grid);
}

// No shape checking needed here. See # NOTE [ grid_sampler Native Functions ].
std::tuple<Tensor, Tensor>
grid_sampler_3d_backward_cuda(const Tensor& grad_output, const Tensor& input,
                              const Tensor& grid, int64_t interpolation_mode, int64_t padding_mode,
                              bool align_corners) {
  // See Note [Writing Nondeterministic Operations]
  // Nondeterministic because of atomicAdd usage
  globalContext().alertNotDeterministic("grid_sampler_3d_backward_cuda");
  auto N = input.size(0);
  auto D = grid.size(1);
  auto H = grid.size(2);
  auto W = grid.size(3);
  auto grad_input = at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  auto grad_grid = at::empty_like(grid, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  int64_t count = N * D * H * W;
  if (count > 0) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "grid_sampler_3d_backward_cuda", [&] {
      if (canUse32BitIndexMath(input) && canUse32BitIndexMath(grid) &&
          canUse32BitIndexMath(grad_output)) {
        grid_sampler_3d_backward_kernel<scalar_t>
          <<<GET_BLOCKS(count), CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
            static_cast<int>(count),
            getTensorInfo<scalar_t, int>(grad_output),
            getTensorInfo<scalar_t, int>(input),
            getTensorInfo<scalar_t, int>(grid),
            getTensorInfo<scalar_t, int>(grad_input),
            getTensorInfo<scalar_t, int>(grad_grid),
            static_cast<GridSamplerInterpolation>(interpolation_mode),
            static_cast<GridSamplerPadding>(padding_mode),
            align_corners);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      } else {
        grid_sampler_3d_backward_kernel<scalar_t>
          <<<GET_BLOCKS(count), CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
            count,
            getTensorInfo<scalar_t, int64_t>(grad_output),
            getTensorInfo<scalar_t, int64_t>(input),
            getTensorInfo<scalar_t, int64_t>(grid),
            getTensorInfo<scalar_t, int64_t>(grad_input),
            getTensorInfo<scalar_t, int64_t>(grad_grid),
            static_cast<GridSamplerInterpolation>(interpolation_mode),
            static_cast<GridSamplerPadding>(padding_mode),
            align_corners);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }
    });
  }
  return std::make_tuple(grad_input, grad_grid);
}

}}  // namespace at::native
