#include <c10/cuda/CUDAFunctions.h>

#include <limits>

namespace c10 {
namespace cuda {

namespace {
// returns -1 on failure
int32_t driver_version() {
  int driver_version = -1;
  cudaError_t err = cudaDriverGetVersion(&driver_version);
  if (err != cudaSuccess) {
    cudaError_t last_err C10_UNUSED = cudaGetLastError();
  }
  return driver_version;
}

int device_count_impl() {
  int count;
  auto err = cudaGetDeviceCount(&count);
  if (err == cudaSuccess) {
    return count;
  }
  // Clear out the error state, so we don't spuriously trigger someone else.
  // (This shouldn't really matter, since we won't be running very much CUDA
  // code in this regime.)
  cudaError_t last_err C10_UNUSED = cudaGetLastError();
  switch (err) {
    case cudaErrorNoDevice:
      // Zero devices is ok here
      count = 0;
      break;
    case cudaErrorInsufficientDriver: {
      auto version = driver_version();
      if (version <= 0) {
        TORCH_CHECK(
            false,
            "Found no NVIDIA driver on your system. Please check that you "
            "have an NVIDIA GPU and installed a driver from "
            "http://www.nvidia.com/Download/index.aspx");
      } else {
        TORCH_CHECK(
            false,
            "The NVIDIA driver on your system is too old (found version ",
            version,
            "). Please update your GPU driver by downloading and installing "
            "a new version from the URL: "
            "http://www.nvidia.com/Download/index.aspx Alternatively, go to: "
            "https://pytorch.org to install a PyTorch version that has been "
            "compiled with your version of the CUDA driver.");
      }
    } break;
    case cudaErrorInitializationError:
      TORCH_CHECK(
          false,
          "CUDA driver initialization failed, you might not "
          "have a CUDA gpu.");
      break;
    case cudaErrorUnknown:
      TORCH_CHECK(
          false,
          "CUDA unknown error - this may be due to an "
          "incorrectly set up environment, e.g. changing env "
          "variable CUDA_VISIBLE_DEVICES after program start. "
          "Setting the available devices to be zero.");
      break;
#if C10_ASAN_ENABLED
    case cudaErrorMemoryAllocation:
      // In ASAN mode, we know that a cudaErrorMemoryAllocation error will
      // pop up if compiled with NVCC (clang-cuda is fine)
      TORCH_CHECK(
          false,
          "Got 'out of memory' error while trying to initialize CUDA. "
          "CUDA with nvcc does not work well with ASAN and it's probably "
          "the reason. We will simply shut down CUDA support. If you "
          "would like to use GPUs, turn off ASAN.");
      break;
#endif // C10_ASAN_ENABLED
    default:
      TORCH_CHECK(
          false,
          "Unexpected error from cudaGetDeviceCount(). Did you run "
          "some cuda functions before calling NumCudaDevices() "
          "that might have already set an error? Error ",
          err,
          ": ",
          cudaGetErrorString(err));
  }
  return count;
}
} // namespace

DeviceIndex device_count() noexcept {
  // initialize number of devices only once
  static int count = []() {
    try {
      auto result = device_count_impl();
      TORCH_INTERNAL_ASSERT(result <= std::numeric_limits<DeviceIndex>::max(), "Too many CUDA devices, DeviceIndex overflowed");
      return device_count_impl();
    } catch (const c10::Error& ex) {
      // We don't want to fail, but still log the warning
      // msg() returns the message without the stack trace
      TORCH_WARN("CUDA initialization: ", ex.msg());
      return 0;
    }
  }();
  return static_cast<DeviceIndex>(count);
}

DeviceIndex device_count_ensure_non_zero() {
  // Call the implementation every time to throw the exception
  int count = device_count_impl();
  // Zero gpus doesn't produce a warning in `device_count` but we fail here
  TORCH_CHECK(count, "No CUDA GPUs are available");
  return static_cast<DeviceIndex>(count);
}

DeviceIndex current_device() {
  int cur_device;
  C10_CUDA_CHECK(cudaGetDevice(&cur_device));
  return static_cast<DeviceIndex>(cur_device);
}

void set_device(DeviceIndex device) {
  C10_CUDA_CHECK(cudaSetDevice(static_cast<int>(device)));
}

void device_synchronize() {
  C10_CUDA_CHECK(cudaDeviceSynchronize());
}

} // namespace cuda
} // namespace c10
