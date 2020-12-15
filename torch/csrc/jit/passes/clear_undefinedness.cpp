#include <torch/csrc/jit/passes/clear_undefinedness.h>
#include <torch/csrc/jit/jit_log.h>

namespace torch {
namespace jit {

void clearUndefinedness(Value* o) {
  if (o->type()->kind() == TensorType::Kind) {
    o->setType(TensorType::get());
  } else if (
      o->type()->kind() == ListType::Kind &&
      o->type()->expect<ListType>()->getElementType()->kind() ==
          TensorType::Kind) {
    o->setType(ListType::create(TensorType::get()));
  }
}

void clearUndefinedness(Block* block) {
  for (auto n : block->nodes()) {
    for (auto o : n->outputs()) {
      clearUndefinedness(o);
    }
    for (auto ib : n->blocks()) {
      clearUndefinedness(ib);
    }
  }
}

void ClearUndefinedness(const std::shared_ptr<Graph>& graph) {
  for (auto i : graph->inputs()) {
    clearUndefinedness(i);
  }
  clearUndefinedness(graph->block());
  GRAPH_DUMP("After removeUndefinedness: ", graph);
}

} // namespace jit
} // namespace torch
