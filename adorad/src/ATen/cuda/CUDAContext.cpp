#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>

#include <ATen/cuda/CUDAConfig.h>
#include <mutex>
#include <deque>
#include <vector>

namespace at { namespace cuda {

namespace {

DeviceIndex num_gpus = -1;
std::once_flag init_flag;
std::deque<std::once_flag> device_flags;
std::vector<cudaDeviceProp> device_properties;

void initCUDAContextVectors() {
  num_gpus = c10::cuda::device_count();
  device_flags.resize(num_gpus);
  device_properties.resize(num_gpus);
}

void initDeviceProperty(DeviceIndex device_index) {
  cudaDeviceProp device_prop;
  AT_CUDA_CHECK(cudaGetDeviceProperties(&device_prop, device_index));
  device_properties[device_index] = device_prop;
}

} // anonymous namespace

// We need this function to force the linking against torch_cuda on Windows.
// If you need to modify this function, please specify a new function and apply the changes
// according to https://github.com/pytorch/pytorch/pull/34288.
// Related issue: https://github.com/pytorch/pytorch/issues/31611.
/* Device info */
int warp_size() {
  return getCurrentDeviceProperties()->warpSize;
}

cudaDeviceProp* getCurrentDeviceProperties() {
  auto device = c10::cuda::current_device();
  return getDeviceProperties(device);
}

cudaDeviceProp* getDeviceProperties(int64_t device) {
  std::call_once(init_flag, initCUDAContextVectors);
  if (device == -1) device = c10::cuda::current_device();
  AT_ASSERT(device >= 0 && device < num_gpus);
  std::call_once(device_flags[device], initDeviceProperty, device);
  return &device_properties[device];
}

Allocator* getCUDADeviceAllocator() {
  return c10::cuda::CUDACachingAllocator::get();
}

} // namespace cuda

} // namespace at
