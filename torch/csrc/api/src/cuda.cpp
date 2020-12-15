#include <torch/cuda.h>

#include <ATen/Context.h>

#include <cstddef>

namespace torch {
namespace cuda {

size_t device_count() {
  return at::detail::getCUDAHooks().getNumGPUs();
}

bool is_available() {
  // NB: the semantics of this are different from at::globalContext().hasCUDA();
  // ATen's function tells you if you have a working driver and CUDA build,
  // whereas this function also tells you if you actually have any GPUs.
  // This function matches the semantics of at::cuda::is_available()
  return cuda::device_count() > 0;
}

bool cudnn_is_available() {
  return is_available() && at::detail::getCUDAHooks().hasCuDNN();
}

/// Sets the seed for the current GPU.
void manual_seed(uint64_t seed) {
  if (is_available()) {
    auto index = at::detail::getCUDAHooks().current_device();
    auto gen = at::detail::getCUDAHooks().getDefaultCUDAGenerator(index);
    {
      // See Note [Acquire lock when using random generators]
      std::lock_guard<std::mutex> lock(gen.mutex());
      gen.set_current_seed(seed);
    }
  }
}

/// Sets the seed for all available GPUs.
void manual_seed_all(uint64_t seed) {
  auto num_gpu = device_count();
  for (size_t i = 0; i < num_gpu; ++i) {
    auto gen = at::detail::getCUDAHooks().getDefaultCUDAGenerator(i);
    {
      // See Note [Acquire lock when using random generators]
      std::lock_guard<std::mutex> lock(gen.mutex());
      gen.set_current_seed(seed);
    }
  }
}

} // namespace cuda
} // namespace torch
