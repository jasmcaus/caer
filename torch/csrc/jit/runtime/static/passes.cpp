#include <torch/csrc/jit/runtime/static/passes.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>

namespace torch {
namespace jit {

void ConcatAddMulReplaceNaNClip(std::shared_ptr<torch::jit::Graph>& graph) {
  // TODO:: check restrictions for inputs; outputs not used elsewhere
  std::string pattern = R"IR(
    graph(%a, %b, %c, %d, %e, %f, %g, %h, %i, %j):
        %y0 = aten::cat(%a, %b)
        %y1 = aten::add(%y0, %c, %d)
        %y2 = aten::mul(%y1, %e)
        %y3 = aten::nan_to_num(%y2, %f, %g, %h)
        %res = aten::clamp(%y3, %i, %j)
        return (%res))IR";
  std::string pattern2 = R"IR(
    graph(%a, %b, %c, %d, %e, %f, %g, %h, %i, %j):
        %y0 = aten::cat(%a, %b)
        %y1 = aten::add(%y0, %c, %d)
        %y2 = aten::mul(%y1, %e)
        %y3 = aten::nan_to_num_(%y2, %f, %g, %h)
        %res = aten::clamp(%y3, %i, %j)
        return (%res))IR";
  std::string fused_pattern = R"IR(
    graph(%a, %b, %c, %d, %e, %f, %g, %h, %i, %j):
        %res = fb::concat_add_mul_replacenan_clip(%c, %e, %a, %i, %j)
        return (%res))IR";

  SubgraphRewriter fuse;
  fuse.RegisterRewritePattern(pattern, fused_pattern);
  fuse.runOnGraph(graph);

  fuse.RegisterRewritePattern(pattern2, fused_pattern);
  fuse.runOnGraph(graph);
}

void CastedBatchOneHotLengths(std::shared_ptr<torch::jit::Graph>& graph) {
  // TODO:: check restrictions for inputs; outputs not used elsewhere
  std::string pattern = R"IR(
    graph(%a, %b, %c, %d, %e, %f, %g):
        %y0 : Tensor = aten::to(%a, %b, %c, %c, %d)
        %y1 : Tensor = fb::batch_one_hot_lengths(%y0, %e, %f)
        %res : Tensor = aten::to(%y1, %g, %c, %c, %d)
        return (%res))IR";
  std::string fused_pattern = R"IR(
    graph(%a, %b, %c, %d, %e, %f, %g):
        %res : Tensor = fb::casted_batch_one_hot_lengths(%a, %e, %f)
        return (%res))IR";
  SubgraphRewriter fuse;
  fuse.RegisterRewritePattern(pattern, fused_pattern);
  fuse.runOnGraph(graph);
}

void ConcatBatchMatMulBatchGather(std::shared_ptr<torch::jit::Graph>& graph) {
  // TODO:: check restrictions for inputs; outputs not used elsewhere
  std::string pattern = R"IR(
    graph(%a, %b, %c, %d, %e, %f):
        %y0 : Tensor = aten::stack(%a, %b)
        %y1 : Tensor = aten::transpose(%y0, %b, %c)
        %y2 : Tensor = aten::bmm(%y0, %y1)
        %y3 : Tensor = aten::flatten(%y2, %d, %e)
        %res : Tensor = aten::index_select(%y3, %b, %f)
        return (%res))IR";
  std::string fused_pattern = R"IR(
    graph(%a, %b, %c, %d, %e, %f):
        %res : Tensor = fb::concat_batch_matmul_batch_gather(%f, %a)
        return (%res))IR";
  SubgraphRewriter fuse;
  fuse.RegisterRewritePattern(pattern, fused_pattern);
  fuse.runOnGraph(graph);
}

void FuseInferenceOpsForSparseNN(std::shared_ptr<torch::jit::Graph>& graph) {
#ifdef FBCODE_CAFFE2
  ConcatAddMulReplaceNaNClip(graph);
  CastedBatchOneHotLengths(graph);
  ConcatBatchMatMulBatchGather(graph);
#endif
}

} // namespace jit
} // namespace torch
