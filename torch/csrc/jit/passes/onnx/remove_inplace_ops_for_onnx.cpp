#include <torch/csrc/jit/passes/onnx/remove_inplace_ops_for_onnx.h>
#include <torch/csrc/jit/passes/remove_inplace_ops.h>
#include <torch/csrc/jit/passes/remove_mutation.h>
#include <limits>

namespace torch {
namespace jit {

namespace {

Value* CreateSizeOfDim(Value* input, int64_t dim, Node* insertBefore) {
  auto graph = input->owningGraph();
  WithInsertPoint guard(insertBefore);
  auto size = graph->insert(aten::size, {input, dim});
  return size;
}

Value* ConvertSelectToIndex(Value* index, Node* insertBefore) {
  // Create index tensor based on index input of aten::select node.
  auto graph = insertBefore->owningGraph();
  WithInsertPoint guard(insertBefore);
  auto idx_tensor = graph->createNumToTensor(index);
  graph->insertNode(idx_tensor);
  return graph->insert(aten::unsqueeze, {idx_tensor->output(), 0});
}

Value* ConvertSliceToIndex(Node* slice, Value* size, Node* insertBefore) {
  // Create index tensor based on aten::slice node.
  const int64_t int_max = std::numeric_limits<int>::max();
  auto graph = slice->owningGraph();
  WithInsertPoint guard(insertBefore);
  TORCH_INTERNAL_ASSERT((slice->inputs()).size() == 5);
  auto start = slice->inputs()[2];
  auto end = slice->inputs()[3];
  auto step = slice->inputs()[4];
  auto index =
      graph->insert(aten::arange, {size}, {NamedValue("dtype", c10::kLong)});
  auto sliced_index =
      graph->insert(aten::slice, {index, {0}, start, end, step});
  return sliced_index;
}

Value* CreateCompleteIndexTensor(Value* size, Node* insertBefore) {
  // Create index tensor of size.
  // The result is torch.tensor([0, 1, 2, ..., size - 1])
  auto graph = size->owningGraph();
  WithInsertPoint guard(insertBefore);
  auto index =
      graph->insert(aten::arange, {size}, {NamedValue("dtype", c10::kLong)});
  return index;
}

bool IsSameSource(const Node* n, const Node* m) {
  const auto& source_n = n->sourceRange().source();
  const auto& source_m = m->sourceRange().source();
  return (
      (source_n->text() == source_m->text()) &&
      (source_n->starting_line_no() == source_m->starting_line_no()));
}

// Trace back all the slice & select nodes associated with the index_put node.
// E.g. The IR for x[1:3, 0] = update
//    ...
//    %8 : Float(2, 4) = aten::slice(%0, %4, %5, %6, %7)
//    ...
//    %11 : Float(2) = aten::select(%8, %9, %10)
//    ...
//    %13 : Tensor?[] = prim::ListConstruct()
//    ...
//    %16 : Float(2) = aten::index_put(%11, %13, %14, %15)
//
// We collect %11 and %8, to construct the index tensors.
// The vector slice_and_select_node contains all the associated slice and
// select node, in the reversed order.
std::vector<Node*> FetchSliceAndSelect(const Node* index_put_node) {
  std::vector<Node*> slice_and_select_node;
  auto src_node = index_put_node->input(0)->node();
  while (src_node) {
    if ((src_node->kind() == aten::slice || src_node->kind() == aten::select) &&
        IsSameSource(src_node, index_put_node)) {
      slice_and_select_node.emplace_back(src_node);
      src_node = src_node->input(0)->node();
    } else {
      src_node = nullptr;
    }
  }
  return slice_and_select_node;
}

struct ConvertedIndex {
  ConvertedIndex(Value* index, c10::Symbol orig_node_kind)
      : index(index), orig_node_kind(orig_node_kind) {}

  Value* index = nullptr;
  c10::Symbol orig_node_kind;
};

std::unordered_map<int64_t, ConvertedIndex> MergeSliceAndSelectToIndices(
    Graph* graph,
    Node* index_put_node,
    const std::vector<Node*>& slice_and_select_nodes,
    Value* orig_data) {
  std::unordered_map<int64_t, ConvertedIndex> dim_index_map;

  // Loop over fetched slice and select nodes and convert them to index tensors.
  // keep track of which dimension the current slice/select node is applying to.
  int64_t cur_dim = 0;
  int64_t dim_offset = 0;
  const auto orig_tensor_indices = index_put_node->input(1)->node()->inputs();
  for (auto it = slice_and_select_nodes.rbegin();
       it != slice_and_select_nodes.rend();
       ++it) {
    auto node = *it;
    // select does not keep dims,
    // this creates offset for latter slice and select nodes.
    auto dim = node->get(attr::dim)->toInt();
    if (dim < 0) {
      auto input_type = orig_data->type()->expect<TensorType>();
      if (input_type->dim().has_value()) {
        auto rank = input_type->dim().value();
        // Rank of original tensor to index on.
        // Minus the offset created by select operators.
        dim = dim + rank - dim_offset;
      } else {
        std::cerr
            << "Error: ONNX Remove Inplace Ops - Cannot export ellipsis indexing for input "
            << "of unknown rank.";
      }
    }
    dim = dim + dim_offset;

    while (cur_dim < dim) {
      // Handle skipped dims, these are created from ..., or tensor indices
      // E.g.: x[torch.tensor([1, 0]), ..., 0] = update, where x has rank 3.
      // Both torch.tensor([1, 0]) and ... are skipped, we only observe
      // aten::select node with dim == 2. Tensor indices will be handled later.
      // Ellipsis(...) are treated as a complete slice over the axes, thus we
      // create index tensors here accordingly.
      if (cur_dim - dim_offset >= orig_tensor_indices.size() ||
          index_put_node->input(1)
              ->node()
              ->input(cur_dim - dim_offset)
              ->node()
              ->mustBeNone()) {
        auto size = CreateSizeOfDim(orig_data, cur_dim, index_put_node);
        WithInsertPoint guard(index_put_node);
        auto index_tensor = graph->insert(
            aten::arange, {size}, {NamedValue("dtype", c10::kLong)});
        dim_index_map.emplace(
            std::piecewise_construct,
            std::forward_as_tuple(cur_dim),
            std::forward_as_tuple(index_tensor, aten::slice));
      } else if (cur_dim - dim_offset < orig_tensor_indices.size()) {
        dim_index_map.emplace(
            std::piecewise_construct,
            std::forward_as_tuple(cur_dim),
            std::forward_as_tuple(
                orig_tensor_indices[cur_dim - dim_offset], aten::index));
      }
      cur_dim++;
    }

    AT_ASSERT(cur_dim == dim);
    if (node->kind() == aten::slice) {
      auto size = CreateSizeOfDim(orig_data, dim, index_put_node);
      auto index_tensor = ConvertSliceToIndex(node, size, index_put_node);
      dim_index_map.emplace(
          std::piecewise_construct,
          std::forward_as_tuple(dim),
          std::forward_as_tuple(index_tensor, aten::slice));
    } else if (node->kind() == aten::select) {
      auto index_tensor = ConvertSelectToIndex(node->input(2), index_put_node);
      dim_index_map.emplace(
          std::piecewise_construct,
          std::forward_as_tuple(dim),
          std::forward_as_tuple(index_tensor, aten::select));
      dim_offset++;
    } else {
      AT_ERROR(
          "Unexpected node kind ",
          node->kind().toDisplayString(),
          " Expected aten::slice or aten::select.");
    }

    cur_dim++;
  }

  while (cur_dim - dim_offset < orig_tensor_indices.size()) {
    dim_index_map.emplace(
        std::piecewise_construct,
        std::forward_as_tuple(cur_dim),
        std::forward_as_tuple(
            orig_tensor_indices[cur_dim - dim_offset], aten::index));
    cur_dim++;
  }

  // Each dimension should have its associated index tensor.
  AT_ASSERT(dim_index_map.size() == cur_dim);
  return dim_index_map;
}

// Convert slice/select operators to tensor indices.
// Reshape the tensor indices according to their axis.
// E.g.                 x[1:3, 0, ind1, ind2] = y
//  slice index shape:   [2,   1, 1 ]
//  select index shape:  [     1, 1 ]
//  ind1 shape:          [        _ ]
//  ind2 shape:          [        _ ]
// where _ is the original size of ind1 and ind2.
// ind1 and ind2 are both 1-d tensors since currently we only supports 1-d
// tensor indices.
std::vector<Value*> ReshapeToAdvancedIndexingFormat(
    Graph* graph,
    Node* index_put_node,
    std::unordered_map<int64_t, ConvertedIndex>& dim_index_map) {
  std::vector<Value*> indices;

  size_t min_index_dim = dim_index_map.size();
  size_t max_index_dim = 0;
  size_t tensor_ind_count = 0;
  for (size_t i = 0; i < dim_index_map.size(); ++i) {
    auto index_i = dim_index_map.find(i);
    AT_ASSERT(index_i != dim_index_map.end());
    if (index_i->second.orig_node_kind == aten::index) {
      if (i < min_index_dim)
        min_index_dim = i;
      if (i > max_index_dim)
        max_index_dim = i;
      tensor_ind_count++;
    }
  }

  if (((max_index_dim - min_index_dim + 1) != tensor_ind_count) &&
      tensor_ind_count != 0) {
    AT_ERROR(
        "Only consecutive 1-d tensor indices are supported in exporting aten::index_put to ONNX.");
  }

  size_t tensor_ind_offset = tensor_ind_count == 0 ? 0 : tensor_ind_count - 1;
  WithInsertPoint guard(index_put_node);
  for (size_t i = 0; i < dim_index_map.size(); ++i) {
    size_t ind_size = 0;
    auto index_i = dim_index_map.find(i);
    AT_ASSERT(index_i != dim_index_map.end());
    Value* index = index_i->second.index;
    switch (index_i->second.orig_node_kind) {
      case aten::select:
      case aten::slice: {
        if (i < min_index_dim) {
          ind_size = dim_index_map.size() - tensor_ind_offset - i;
        } else {
          ind_size = dim_index_map.size() - i;
        }
        break;
      }

      case aten::index: {
        ind_size = dim_index_map.size() - tensor_ind_offset - min_index_dim;
        break;
      }
      default:
        AT_ERROR("Unexpected node kind ", index_i->second.orig_node_kind);
    }

    std::vector<int64_t> view_shape(ind_size, 1);
    view_shape[0] = -1;
    auto unsqueezed_index = graph->insert(aten::view, {index, view_shape});
    indices.emplace_back(unsqueezed_index);
  }

  return indices;
}

// Trace back all the slice & select nodes associated with the index_put node,
// and convert them to associated indices.
// E.g. The IR for x[1:3, 0] = update
//    ...
//    %8 : Float(2, 4) = aten::slice(%0, %4, %5, %6, %7)
//    ...
//    %11 : Float(2) = aten::select(%8, %9, %10)
//    ...
//    %13 : Tensor?[] = prim::ListConstruct()
//    ...
//    %16 : Float(2) = aten::index_put(%11, %13, %14, %15)
// The aten::index_put node alone does not contain any indices (%13 : Tensor?[]
// = prim::ListConstruct()).
//    ...
//    # Below constructs index from slice node.
//    %23 : Long() = aten::size(%0, %4)
//    %28 : Tensor = aten::arange(%23, %24, %25, %26, %27)
//    %33 : Tensor = aten::slice(%28, %4, %5, %6, %7)
//    %39 : int[] = prim::Constant[value=[-1, 1]]()
//    %40 : Tensor = aten::view(%33, %39)
//    ...
//    # Below constructs index from select node.
//    %36 : int = prim::Constant[value=0]()
//    %37 : Tensor = aten::unsqueeze(%10, %36)
//    %42 : int[] = prim::Constant[value=[-1]]()
//    %43 : Tensor = aten::view(%37, %42)
//    ...
//    # Adding the above two indices to index_put
//    %44 : Tensor?[] = prim::ListConstruct(%40, %43)
//    %45 : Float(2, 5) = aten::index_put(%0, %44, %14, %15)
void SquashSliceAndSelect(Node* index_put_node) {
  auto graph = index_put_node->owningGraph();

  // Find slice and select operators that are associated with this index
  // operator. E.g. x[1:3, 0] = y will generate one slice operator(1:3) and one
  // select operator(0).
  std::vector<Node*> slice_and_select_nodes =
      FetchSliceAndSelect(index_put_node);

  Node* last_node = slice_and_select_nodes.size() > 0
      ? slice_and_select_nodes.back()
      : index_put_node;
  Value* orig_data = last_node->input(0);

  // Convert fetched slice/select operators into tensor indices.
  std::unordered_map<int64_t, ConvertedIndex> dim_index_map =
      MergeSliceAndSelectToIndices(
          graph, index_put_node, slice_and_select_nodes, orig_data);
  std::vector<Value*> indices =
      ReshapeToAdvancedIndexingFormat(graph, index_put_node, dim_index_map);

  // Create new aten::index_put operator.
  WithInsertPoint guard(index_put_node);
  const auto list_indices =
      graph->insertNode(graph->createList(OptionalType::ofTensor(), indices))
          ->output();
  auto new_index_put = graph->insert(
      aten::index_put,
      {orig_data,
       list_indices,
       index_put_node->input(2),
       index_put_node->input(3)});
  new_index_put->copyMetadata(index_put_node->output());
  index_put_node->output()->replaceAllUsesWith(new_index_put);

  orig_data->replaceAllUsesAfterNodeWith(new_index_put->node(), new_index_put);
}

void PrepareCopyForONNX(Block* block) {
  auto it = block->nodes().begin();
  while (it != block->nodes().end()) {
    auto node = *it;
    ++it;
    for (auto block : node->blocks()) {
      PrepareCopyForONNX(block);
    }

    if (node->kind() == aten::copy_) {
      // aten::copy_ can be viewed as a special case of index_put, where the
      // tensor indices input is empty.
      // Remove aten::copy_, and replace it with index_put.
      // 1. create an empty listConstruct node as indices input for index_put.
      // 2. create index_put node.

      // Tracing aten::copy_ broadcasts the rhs values.
      // 3. Apply broadcasting for scripting.
      WithInsertPoint guard(node);
      auto graph = node->owningGraph();
      auto dummy_list =
          graph->insertNode(graph->createList(OptionalType::ofTensor(), {}))
              ->output();

      auto expanded_value =
          graph->insert(aten::expand_as, {node->input(1), node->input(0)});
      expanded_value->node()->setSourceRange(node->sourceRange());
      expanded_value->copyMetadata(node->input(1));

      auto index_put = graph->insert(
          aten::index_put,
          {node->input(0), dummy_list, expanded_value, node->input(2)});
      index_put->node()->setSourceRange(node->sourceRange());
      index_put->copyMetadata(node->output());
      node->output()->replaceAllUsesWith(index_put);
    }
  }
}

void PrepareIndexPutForONNX(Block* block) {
  auto it = block->nodes().begin();
  while (it != block->nodes().end()) {
    auto node = *it;
    ++it;
    for (auto block : node->blocks()) {
      PrepareIndexPutForONNX(block);
    }

    if (node->kind() == aten::index_put || node->kind() == aten::index_put_) {
      SquashSliceAndSelect(node);
    }
  }
}

// aten::pop is inplace. The tensor list input is updated.
// This pass creates an aten::__getitem__ op to return the original output from
// aten::pop. Then it makes the original aten::pop operator return the updated
// tensor list, and replaces all later uses of that tensor list with this new
// output.
static void PrepareListPopForONNX(Block* b) {
  for (auto it = b->nodes().begin(), end = b->nodes().end(); it != end; ++it) {
    for (auto* child_block : it->blocks()) {
      PrepareListPopForONNX(child_block);
    }

    if (it->kind() == aten::pop) {
      //   %ten : Tensor = aten::pop(%seq, %pos)
      // Convert to
      //   %ten : Tensor = aten::__getitem__(%seq, %pos)
      //   %new_seq : Tensor[] = aten::pop(%seq, %pos)
      // And replace all uses of %seq afterwards with %new_seq
      Node* getitem_node =
          b->owningGraph()->create(aten::__getitem__, {it->inputs()});
      getitem_node->output()->copyMetadata(it->output());
      getitem_node->insertBefore(*it);
      it->output()->replaceAllUsesWith(getitem_node->output());

      it->output()->copyMetadata(it->inputs()[0]);
      it->inputs()[0]->replaceAllUsesAfterNodeWith(*it, it->output());
    }
  }
}

static void PrepareListAppendAndInsertForONNX(Block* b) {
  for (auto it = b->nodes().begin(), end = b->nodes().end(); it != end; ++it) {
    for (auto* child_block : it->blocks()) {
      PrepareListPopForONNX(child_block);
    }

    if (it->kind() == aten::insert || it->kind() == aten::append) {
      if (it->outputs().size() == 0) {
        it->addOutput();
        it->output()->copyMetadata(it->inputs()[0]);
      }
      it->inputs()[0]->replaceAllUsesAfterNodeWith(*it, it->output());
    }
  }
}

// Remove Mutation pass does not handle mutation on block inputs.
// To fix this, insert a clone node following the graph input:
// Example for graph input node %0:
// Before:
// graph(%0 : Tensor):
//   %5 : Tensor = aten::zero_(%0)
//   ...
// After:
// graph(%0 : Tensor):
//   %2 : None = prim::Constant()
//   %3 : Tensor = aten::clone(%0, %2)
//   %5 : Tensor = aten::zero_(%3)
//   ...

static void PrepareForRemoveMutations(MutationRemover& mr, Block* b) {
  for (auto it = b->nodes().begin(), end = b->nodes().end(); it != end; ++it) {
    for (auto* child_block : it->blocks()) {
      PrepareForRemoveMutations(mr, child_block);
    }
  }

  for (auto input : b->inputs()) {
    for (auto use : input->uses()) {
      Node* node = use.user;
      if (!mr.inplaceOpVariant(node)) {
        continue;
      }

      auto it = std::find(node->inputs().begin(), node->inputs().end(), input);

      if (it != node->inputs().end()) {
        int index = std::distance(node->inputs().begin(), it);

        std::cerr
            << "Warning: ONNX Preprocess - Removing mutation on block inputs. "
            << "This changes graph semantics." << std::endl;

        if (input->type()->kind() == TypeKind::ListType) {
          // Create an aten::list to clone the list in graph inputs
          auto newNode = node->owningGraph()->create(aten::list, 1);
          newNode->output()->copyMetadata(input);
          newNode->addInput(input);
          newNode->insertBefore(node);
          node->replaceInput(index, newNode->output());
          input->replaceAllUsesAfterNodeWith(node, newNode->output());
        } else {
          // Create an aten::clone to clone the tensor in graph inputs
          auto newNode = node->owningGraph()->create(aten::clone, 1);
          newNode->output()->copyMetadata(input);
          newNode->addInput(input);

          auto* noneNode = node->owningGraph()->create(prim::Constant);
          noneNode->output()->setType(NoneType::get());
          newNode->addInput(noneNode->output());

          newNode->insertBefore(node);
          noneNode->insertBefore(newNode);
          node->replaceInput(index, newNode->output());
          input->replaceAllUsesAfterNodeWith(node, newNode->output());
        }
      }
    }
  }
}

} // namespace

void PrepareInplaceOpsForONNX(const std::shared_ptr<Graph>& graph) {
  PrepareCopyForONNX(graph->block());
  PrepareIndexPutForONNX(graph->block());
  PrepareListPopForONNX(graph->block());
  PrepareListAppendAndInsertForONNX(graph->block());
}

void RemoveInplaceOpsForONNX(const std::shared_ptr<Graph>& graph) {
  MutationRemover mr(graph);
  PrepareForRemoveMutations(mr, graph->block());
  RemoveTensorMutation(graph);
  RemoveListMutation(graph);
}

} // namespace jit
} // namespace torch
