#pragma once

#include <aten/src/ATen/Context.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

TORCH_CUDA_API void TypePropagate(std::shared_ptr<Graph>& graph);

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
