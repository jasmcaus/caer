#include <torch/csrc/jit/runtime/static/fusion.h>
#include <ATen/core/interned_strings.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/utils/subgraph_utils.h>
#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/runtime/static/impl.h>

namespace torch {
namespace jit {

void createFusionGroups(Block* block, AliasDb* aliasDb);

void fuseStaticSubgraphs(std::shared_ptr<Graph> graph) {
  PrepareGraphForStaticRuntime(graph);
  auto aliasDb = torch::make_unique<AliasDb>(graph);
  createFusionGroups(graph->block(), aliasDb.get());
  torch::jit::EliminateDeadCode(graph);
}

Operation createStaticSubgraphRuntime(const Node* node) {
  auto g = torch::jit::PrepareForStaticRuntime(node->g(attr::Subgraph));
  auto runtime = std::make_shared<torch::jit::StaticRuntime>(g);
  auto num_inputs = runtime->get_inference_module()->input_regs.size();
  return [runtime, num_inputs](Stack* stack) {
    RECORD_FUNCTION("Static Runtime", std::vector<c10::IValue>());
    auto inps = torch::jit::last(stack, num_inputs);
    std::vector<at::Tensor> t_inputs;
    t_inputs.reserve(num_inputs);
    for (const auto& inp : inps) {
      t_inputs.emplace_back(inp.toTensor());
    }
    torch::jit::drop(stack, num_inputs);
    auto outputs = runtime->run(t_inputs);
    for (auto& o : outputs) {
      push_one(*stack, std::move(o));
    }
    return 0;
  };
}

RegisterOperators StaticSubgraphOps({torch::jit::Operator(
    prim::StaticSubgraph,
    createStaticSubgraphRuntime,
    AliasAnalysisKind::INTERNAL_SPECIAL_CASE)});

#define REQ(cond)                           \
  if (!(cond)) {                            \
    GRAPH_DEBUG("Failed cond " #cond "\n"); \
    return false;                           \
  }

bool canHandle(Node* node) {
  for (Value* input : node->inputs()) {
    // TODO checks
  }

  auto kind = node->kind();
  if (kind.is_prim()) {
    REQ(kind == prim::TupleConstruct || kind == prim::ListConstruct ||
        kind == prim::StaticSubgraph);
    return true;
  }
  const Operator& op = node->getOperator();
  auto analysis = op.aliasAnalysisKind();
  if (AliasAnalysisKind::PURE_FUNCTION == analysis ||
      (AliasAnalysisKind::FROM_SCHEMA == analysis &&
       !node->schema().is_mutable())) {
    return true;
  }
  return false;
}

bool canMerge(Node* consumer, Node* producer, AliasDb* aliasDb) {
  // Only fuse within a block
  REQ(consumer->owningBlock() == producer->owningBlock());

  // Symbolic checks
  REQ(canHandle(producer) || producer->kind() == prim::StaticSubgraph);
  TORCH_INTERNAL_ASSERT(
      consumer->kind() == prim::StaticSubgraph || canHandle(consumer));

  // Alias checks
  REQ(aliasDb->couldMoveBeforeTopologically(producer, consumer));

  // Ops that return aliases can only be folded if this is the only use.
  if (producer->kind() == aten::slice || producer->kind() == aten::unsqueeze ||
      producer->kind() == prim::ConstantChunk) {
    for (auto& use : producer->output(0)->uses()) {
      REQ(use.user == consumer);
    }
  }

  return true;
}

Node* getOrCreateStaticSubgraph(Node* n, AliasDb* aliasDb) {
  if (n->hasAttribute(attr::Subgraph) && n->kind() == prim::StaticSubgraph) {
    return n;
  }
  GRAPH_UPDATE("Creating a static subgraph::Group node from: ", *n);
  return SubgraphUtils::createSingletonSubgraphAndUpdateAliasing(
      n, prim::StaticSubgraph, *aliasDb);
}

value_list sortReverseTopological(ArrayRef<Value*> inputs, Block* b) {
  value_list result;
  for (auto i : inputs) {
    if (i->node()->owningBlock() == b) {
      result.push_back(i);
    }
  }
  // Sort in reverse topological order
  std::sort(result.begin(), result.end(), [&](Value* a, Value* b) {
    return a->node()->isAfter(b->node());
  });
  return result;
}

static void debugDumpFusionGroup(const std::string& msg, Node* n) {
  GRAPH_DEBUG(msg, *n);
  if (n->kind() == prim::StaticSubgraph) {
    GRAPH_DEBUG(*n->g(attr::Subgraph));
  }
}

c10::optional<Node*> tryMerge(
    Node* fusion_group,
    Node* to_merge,
    AliasDb* aliasDb) {
  if (!canMerge(fusion_group, to_merge, aliasDb)) {
    return c10::nullopt;
  }

  std::vector<Node*> nodes_to_merge = {to_merge};

  if (to_merge->kind() == aten::cat) {
    Node* listconstruct = to_merge->input(0)->node();
    nodes_to_merge.push_back(listconstruct);
  }

  // First, try to move all the nodes we want to fuse next to the fusion
  // group.
  Node* move_point = fusion_group;
  for (auto n : nodes_to_merge) {
    GRAPH_UPDATE("Trying to move node next to fusion group: ", getHeader(n));
    if (!aliasDb->moveBeforeTopologicallyValid(n, move_point)) {
      GRAPH_UPDATE("Failed to move because of AliasDb checks!");
      return c10::nullopt;
    }
    move_point = n;
  }

  // Now all the nodes that we're going to fuse are moved next to the fusion
  // group, so we can safely merge them into the fusion group subgraph.
  fusion_group = getOrCreateStaticSubgraph(fusion_group, aliasDb);

  for (auto n : nodes_to_merge) {
    GRAPH_UPDATE("Merging ", getHeader(n));
    SubgraphUtils::mergeNodeIntoSubgraphAndUpdateAliasing(
        n, fusion_group, *aliasDb);
  }
  return fusion_group;
}

std::pair<graph_node_list::iterator, bool> createFusionGroup(
    Node* fusion_node,
    AliasDb* aliasDb) {
  fusion_node = getOrCreateStaticSubgraph(fusion_node, aliasDb);

  GRAPH_DEBUG("Iteratively pull input nodes into the fusion group...\n");
  auto inputs =
      sortReverseTopological(fusion_node->inputs(), fusion_node->owningBlock());
  for (auto input : inputs) {
    debugDumpFusionGroup("Current fusion group: ", fusion_node);
    GRAPH_DEBUG("Trying to merge: ", *input->node());
    if (auto maybe_fusion_group =
            tryMerge(fusion_node, input->node(), aliasDb)) {
      // we successfully merged, so the new group's `inputs` may have
      // changed. So rescan the new group for more merging opportunities.
      return std::make_pair(
          maybe_fusion_group.value()->reverseIterator(), true);
    }
  }

  return std::make_pair(++fusion_node->reverseIterator(), false);
}

std::pair<graph_node_list::iterator, bool> scanNode(Node* n, AliasDb* aliasDb) {
  GRAPH_DEBUG("Considering node:", *n);

  if (!canHandle(n)) {
    return std::make_pair(++n->reverseIterator(), false);
  }

  return createFusionGroup(n, aliasDb);
}

void createFusionGroups(Block* block, AliasDb* aliasDb) {
  bool any_changed = true;
  while (any_changed) {
    any_changed = false;
    for (auto it = block->nodes().rbegin(); it != block->nodes().rend();) {
      bool changed;
      std::tie(it, changed) = scanNode(*it, aliasDb);
      any_changed |= changed;
    }
  }

  for (Node* n : block->nodes()) {
    for (Block* b : n->blocks()) {
      createFusionGroups(b, aliasDb);
    }
  }

  // Try to merge adjacent fusion groups together. Because we have only merged
  // by looking at graph inputs, without this we would not attempt to merge
  // adjacent fusion groups that don't have a depdency on each other

  std::vector<Node*> initial_fusion_groups;
  for (Node* n : block->nodes()) {
    if (n->kind() == prim::StaticSubgraph) {
      initial_fusion_groups.push_back(n);
    }
  }

  Node* prev_fusion_group =
      initial_fusion_groups.size() ? initial_fusion_groups[0] : nullptr;

  for (size_t i = 1; i < initial_fusion_groups.size(); ++i) {
    // Try merging the just created fusion group into the previous one.
    // If it did not work, then put the previous fusion group into
    // fusion_groups vector - we will not touch it anymore in this loop.
    // If merging suceeded, save the merged group as the "previous" fusion
    // group so that we can try to merge the next one into it.

    Node* fusion_group = initial_fusion_groups[i];
    debugDumpFusionGroup(
        "Trying to merge into the previous fusion group: ", prev_fusion_group);
    if (auto merged_fusion_group =
            tryMerge(prev_fusion_group, fusion_group, aliasDb)) {
      prev_fusion_group = *merged_fusion_group;
      debugDumpFusionGroup(
          "Successfully merged into the previous fusion group: ",
          prev_fusion_group);
    } else {
      GRAPH_DEBUG("Cannot merge into the previous fusion group");
      prev_fusion_group = fusion_group;
    }
  }
}

} // namespace jit
} // namespace torch
