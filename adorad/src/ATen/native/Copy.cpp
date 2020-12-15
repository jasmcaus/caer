#include <ATen/native/Copy.h>

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/quantized/Copy.h>
#include <ATen/native/vulkan/ops/Copy.h>
#include <ATen/quantized/Quantizer.h>
#include <ATen/vulkan/Context.h>
#include <ATen/metal/Context.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/Parallel.h>
#include <torch/library.h>

#ifdef USE_FBGEMM
#include <fbgemm/Fbgemm.h>
#include <fbgemm/FbgemmConvert.h>
#endif

namespace {

using namespace at;

bool copy_transpose_valid(const Tensor& self, const Tensor& src) {
  const int MIN_SZ = 60 * 60;
  return self.is_contiguous() && src.numel() != 0 && src.dim() == 2 &&
      src.stride(0) == 1 && src.stride(1) == src.size(0) &&
      self.scalar_type() == src.scalar_type() &&
      self.numel() >= MIN_SZ;
}

// special case copy where tensor is contiguous and src is a transposed matrix
// This can be generalized to most copies, but it's trickier
void copy_same_type_transpose_(Tensor& self, const Tensor& src) {
  int64_t BLOCK_SZ;
  if (self.scalar_type() == kByte) {
    BLOCK_SZ = 120;
  } else {
    BLOCK_SZ = 60;
  }
  Tensor buf = empty({BLOCK_SZ, BLOCK_SZ}, self.options());

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(kHalf, kBool, kBFloat16, self.scalar_type(), "copy_", [&] {
    scalar_t* sp = src.data_ptr<scalar_t>();
    scalar_t* rp = self.data_ptr<scalar_t>();
    scalar_t* bp = buf.data_ptr<scalar_t>();

    int64_t NR = src.size(0);
    int64_t NC = src.size(1);
    for (int64_t R = 0; R < NR; R += BLOCK_SZ) {
      for (int64_t C = 0; C < NC; C += BLOCK_SZ) {
        scalar_t* spo = sp + R + C * NR;
        scalar_t* rpo = rp + C + R * NC;

        int nr = std::min(NR - R, BLOCK_SZ);
        int nc = std::min(NC - C, BLOCK_SZ);

        // 1. copy columns from src to buf
        for (int c = 0; c < nc; c++) {
          memcpy(bp + c * BLOCK_SZ, spo + c * NR, nr * sizeof(scalar_t));
        }

        // 2. transpose buf in place
        int rc_max = std::max(nr, nc);
        int rc_min = std::min(nr, nc);
        for (int r = 0; r < rc_max; r++) {
          int end = std::min(r, rc_min);
          for (int c = 0; c < end; c++) {
            scalar_t tmp = bp[r + BLOCK_SZ * c];
            bp[r + BLOCK_SZ * c] = bp[r * BLOCK_SZ + c];
            bp[r * BLOCK_SZ + c] = tmp;
          }
        }

        // 3. copy rows from buf to dst
        for (int r = 0; r < nr; r++) {
          memcpy(rpo + r * NC, bp + r * BLOCK_SZ, nc * sizeof(scalar_t));
        }
      }
    }
  });
}

// Devices directly supported by this copy implementation. Other device types
// (e.g. XLA) may be supported by overriding copy_ and _copy_from.
bool is_supported_device(Device device) {
  DeviceType device_type = device.type();
  return device_type == kCPU || device_type == kCUDA || device_type == kHIP || device_type == kVulkan || device_type == kMetal;
}

} // namespace

namespace at {
namespace native {

static Tensor & copy_impl(Tensor & self, const Tensor & src, bool non_blocking) {
  // TODO: this should be handled during dispatch, but that's missing...
  TORCH_CHECK(self.defined(), "self is undefined");
  TORCH_CHECK(src.defined(), "src is undefined");

  // FBGeMM kernel support exists only for the following case,
  // 1. Memory Format for source and destination tensors is contiguous.
  // 2. Device for both the source and destination tensor is CPU.
  // 3. dtype conversion between FP32->FP16 and FP16->FP32.
  #ifdef USE_FBGEMM
    if (((self.dtype() == at::kFloat && src.dtype() == at::kHalf) ||
         (self.dtype() == at::kHalf && src.dtype() == at::kFloat)) &&
        (self.device().is_cpu() && src.device().is_cpu()) &&
        !self.is_sparse() && !src.is_sparse() &&
        ((self.is_contiguous() && src.is_contiguous()) ||
         (self.is_non_overlapping_and_dense() && self.strides() == src.strides()))) {
      if (src.dtype() == at::kFloat && self.dtype() == at::kHalf) {
        auto* output_ptr =
            reinterpret_cast<fbgemm::float16*>(self.data_ptr<at::Half>());
        at::parallel_for(
            0,
            self.numel(),
            at::internal::GRAIN_SIZE,
            [&](int64_t begin, int64_t end) {
              fbgemm::FloatToFloat16_simd(
                  src.data_ptr<float>() + begin,
                  output_ptr + begin,
                  end - begin);
            });
      } else {
        auto in_data = reinterpret_cast<fbgemm::float16*>(
            src.data_ptr<at::Half>());
        auto* output_ptr = self.data_ptr<float>();
        at::parallel_for(
            0,
            self.numel(),
            at::internal::GRAIN_SIZE,
            [&](int64_t begin, int64_t end) {
              fbgemm::Float16ToFloat_simd(
                  in_data + begin, output_ptr + begin, end - begin);
            });
      }
      return self;
    }
  #endif

  if (self.is_sparse() && src.is_sparse()) {
    return at::copy_sparse_to_sparse_(self, src, non_blocking);
  } else if (self.is_sparse() || src.is_sparse()) {
    AT_ERROR("copy_() between dense and sparse Tensors is not implemented! Found self type = ",
             self.toString(), " and src type = ", src.toString());
  }

  if (self.is_same(src)) {
    return self;
  }

  // Re-dispatch copies when either src or self device not implemented here (e.g. XLA).
  // _copy_from has a proper device dispatch setup.
  // This includes:
  //   cpu_tensor.copy_(xla_tensor) => xla_tensor._copy_from(cpu_tensor)
  //   xla_tensor.copy_(cpu_tensor) => cpu_tensor._copy_from(xla_tensor)
  // Both the _copy_from calls above will be dispatched to XLA's _copy_from kernels.
  if (!is_supported_device(src.device()) || !is_supported_device(self.device())) {
    at::_copy_from(src, self, non_blocking);
    return self;
  }

  if (self.is_quantized() && !src.is_quantized()) {
    return quantized_copy_from_float_cpu_(self, src);
  }

  if (self.is_quantized() && src.is_quantized()) {
    TORCH_CHECK(self.qscheme() == src.qscheme(),
                "Quantized Copy only works with same qscheme");
    TORCH_CHECK(self.scalar_type() == src.scalar_type());
    self.set_quantizer_(src.quantizer());
  }

  if (!self.is_quantized() && src.is_quantized()) {
    TORCH_CHECK(false, "Copying from quantized Tensor to non-quantized Tensor is not allowed, please use dequantize to get a float Tensor from a quantized Tensor");
  }

  if (self.device().type() == at::kVulkan || src.device().type() == at::kVulkan) {
  #ifdef USE_VULKAN_API
    return vulkan::ops::copy_(self, src);
  #else
    return at::vulkan::vulkan_copy_(self, src);
  #endif
  }

  if (self.device().type() == at::kMetal || src.device().type() == at::kMetal) {
    return at::metal::metal_copy_(self, src);
  }

  auto iter = TensorIteratorConfig()
    .add_output(self)
    .add_input(src)
    .resize_outputs(false)
    .check_all_same_dtype(false)
    .check_all_same_device(false)
    .build();

  if (iter.numel() == 0) {
    return self;
  }

  DeviceType device_type = iter.device_type(0);
  if (iter.device_type(1) == kCUDA) {
    device_type = kCUDA;
  } else if (iter.device_type(1) == kHIP) {
    device_type = kHIP;
  }

  // TODO: if we need to, we can also enable this path for quantized tensor
  if (device_type == kCPU && copy_transpose_valid(self, src) && !self.is_quantized()) {
    copy_same_type_transpose_(self, src);
    return self;
  }

  if(!self.is_complex() && src.is_complex()) {
    TORCH_WARN_ONCE("Casting complex values to real discards the imaginary part");
  }

  copy_stub(device_type, iter, non_blocking);
  return self;
}

Tensor& copy_(Tensor& self, const Tensor& src, bool non_blocking) {
  auto maybe_outnames = namedinference::compute_broadcast_outnames(self, src);
  {
    NoNamesGuard guard;
    copy_impl(self, src, non_blocking);
  }
  namedinference::propagate_names_if_nonempty(self, maybe_outnames);
  return self;
}

DEFINE_DISPATCH(copy_stub);

} // namespace native
} // namespace at
