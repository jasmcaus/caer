#include <torch/csrc/distributed/autograd/rpc_messages/cleanup_autograd_context_resp.h>

namespace torch {
namespace distributed {
namespace autograd {

rpc::Message CleanupAutogradContextResp::toMessageImpl() && {
  std::vector<torch::Tensor> tensors;
  std::vector<char> payload;
  return rpc::Message(
      std::move(payload),
      std::move(tensors),
      rpc::MessageType::CLEANUP_AUTOGRAD_CONTEXT_RESP);
}

std::unique_ptr<CleanupAutogradContextResp> CleanupAutogradContextResp::
    fromMessage(const rpc::Message& message /* unused */) {
  return std::unique_ptr<CleanupAutogradContextResp>();
}

} // namespace autograd
} // namespace distributed
} // namespace torch
