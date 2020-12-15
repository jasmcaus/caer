#import <Accelerate/Accelerate.h>
#import <Foundation/Foundation.h>

#import <ATen/native/metal/MetalUtils.h>

#include <c10/util/Exception.h>
#include <c10/util/Logging.h>
#include <iostream>

namespace at {
namespace native {
namespace metal {

std::vector<uint16_t> fp32_to_fp16(const std::vector<float>& src) {
  unsigned long count = src.size();
  std::vector<uint16_t> output(count, 0);
  vImage_Buffer float32{(void*)src.data(), 1, count, count * sizeof(float)};
  vImage_Buffer float16{
      (void*)output.data(), 1, count, count * sizeof(uint16_t)};
  if (vImageConvert_PlanarFtoPlanar16F(&float32, &float16, 0) !=
      kvImageNoError) {
    TORCH_CHECK(false, "fp32_to_fp16 failed");
    return {};
  }

  return output;
}

std::vector<float> fp16_to_fp32(const std::vector<uint16_t>& src) {
  unsigned long count = src.size();
  std::vector<float> output(count, 0);
  vImage_Buffer float16{(void*)src.data(), 1, count, count * sizeof(uint16_t)};
  vImage_Buffer float32{(void*)output.data(), 1, count, count * sizeof(float)};
  if (vImageConvert_Planar16FtoPlanarF(&float16, &float32, 0) !=
      kvImageNoError) {
    TORCH_CHECK(false, "fp16_to_fp32 failed");
    return {};
  }
  return output;
}

std::vector<float> NCHW_to_NC4(
    const float* src,
    const std::vector<int64_t>& sizes) {
  int64_t N = sizes[0];
  int64_t C = sizes[1];
  int64_t H = sizes[2];
  int64_t W = sizes[3];
  int64_t src_image_count = C * H * W;
  int64_t src_count = N * src_image_count;
  int64_t slices = (C + 3) / 4;
  int64_t numComponents = C < 3 ? C : 4;
  int64_t dst_image_count = slices * numComponents * W * H;
  int64_t dst_count = N * dst_image_count;
  std::vector<float> output(dst_count, 0.0f);
  for (int n = 0; n < N; ++n) {
    int64_t src_image = n * src_image_count;
    int64_t dst_image = n * dst_image_count;
    for (int i = 0; i < slices; ++i) {
      int64_t slice = i * W * H * numComponents;
      for (int j = 0; j < W * H; ++j) {
        for (int k = 0; k < numComponents; ++k) {
          int ii = src_image + slice + k * W * H + j;
          int oi = dst_image + slice + j * numComponents + k;
          if (k < C && ii < src_count) {
            output[oi] = src[ii];
          }
        }
      }
    }
  }

  return output;
}

std::vector<float> NC4_to_NCHW(
    const float* src,
    const std::vector<int64_t>& sizes) {
  int64_t N = sizes[0];
  int64_t C = sizes[1];
  int64_t H = sizes[2];
  int64_t W = sizes[3];
  int64_t slices = (C + 3) / 4;
  int64_t numComponents = C < 3 ? C : 4;
  int64_t src_image_count = slices * numComponents * W * H;
  int64_t dst_image_count = C * H * W;
  int64_t dst_count = N * dst_image_count;
  std::vector<float> output(dst_count, 0.0f);
  for (int n = 0; n < N; ++n) {
    int64_t src_image = n * src_image_count;
    int64_t dst_image = n * dst_image_count;
    for (int i = 0; i < slices; ++i) {
      int64_t slice = i * W * H * numComponents;
      for (int j = 0; j < numComponents; ++j) {
        for (int k = 0; k < W * H; ++k) {
          int ii = src_image + slice + k * numComponents + j;
          int oi = dst_image + slice + j * W * H + k;
          if (j < C && oi < dst_count) {
            output[oi] = src[ii];
          }
        }
      }
    }
  }
  return output;
}

}
}
}
