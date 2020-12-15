#include <torch/csrc/jit/passes/specialize_autogradzero.h>
#include <c10/util/Exception.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/clear_undefinedness.h>
#include <torch/csrc/jit/runtime/graph_executor.h>

namespace torch {
namespace jit {

struct AutogradZeroSpecializer {
  enum class State { Nonzero, Zero, Unknown };

  AutogradZeroSpecializer(std::shared_ptr<Graph> graph)
      : graph_(std::move(graph)) {}

  void run() {
    if (!isBackwardGraph()) {
      return;
    }
    if (getProfilingMode()) {
      if (auto versioning_if = guardSpecializations()) {
        specializeAutogradOps(versioning_if->blocks()[0]);
        GRAPH_DUMP("After versioning graph", graph_);
      }
    } else {
      setStatesOnGraphInputs();
      specializeAutogradOps(graph_->block());
    }
    GRAPH_DUMP("After specializeAutogradOps graph", graph_);
  }

 private:
  bool isBackwardGraph() {
    return std::any_of(
        graph_->nodes().begin(), graph_->nodes().end(), [](Node* n) {
          switch (n->kind()) {
            case prim::AutogradAnyNonZero:
            case prim::AutogradAdd:
            case aten::_grad_sum_to_size:
              return true;
            default:
              return false;
          }
        });
  }

  void replaceBlockInputsWithGraphInputs(Block* b) {
    TORCH_INTERNAL_ASSERT(graph_->inputs().size() == b->inputs().size());
    size_t num_inputs = graph_->inputs().size();
    for (size_t i = 0; i < num_inputs; ++i) {
      b->inputs().at(i)->replaceAllUsesWith(graph_->inputs().at(i));
    }
    for (size_t i = 0; i < num_inputs; ++i) {
      b->eraseInput(num_inputs - (1 + i));
    }
  }

  void setStatesOnGraphInputs() {
    for (Value* input : graph_->inputs()) {
      const auto& tp = input->type();
      if (auto tt = tp->cast<TensorType>()) {
        if (tt->undefined()) {
          if (*tt->undefined()) {
            state_[input] = State::Zero;
          } else {
            state_[input] = State::Nonzero;
          }
        } else {
          state_[input] = State::Unknown;
        }
      } else if (
          tp->isSubtypeOf(TensorType::get()) ||
          tp->isSubtypeOf(ListType::ofTensors())) {
        state_[input] = State::Nonzero;
      } else {
        state_[input] = State::Unknown;
      }
    }
  }

  static Node* getUse(Value* inp, Symbol kind) {
    for (auto use : inp->uses()) {
      if (use.user->kind() == kind) {
        return use.user;
      }
    }

    return nullptr;
  }

  void removeProfiledOptionalUses(Value* v) {
    std::vector<Node*> profiled_opt_uses;
    for (const Use& use : v->uses()) {
      if (use.user->kind() == prim::profile_optional) {
        profiled_opt_uses.push_back(use.user);
      }
    }
    for (Node* n : profiled_opt_uses) {
      n->output()->replaceAllUsesWith(v);
    }
  }

  Node* guardSpecializations() {
    auto versioning_if = graph_->create(prim::If, {}, graph_->outputs().size());
    auto value_map = [](Value* v) { return v; };
    auto true_block = versioning_if->addBlock();
    auto false_block = versioning_if->addBlock();

    // we will optimize true_block
    true_block->cloneFrom(graph_->block(), value_map);
    replaceBlockInputsWithGraphInputs(true_block);
    false_block->cloneFrom(graph_->block(), value_map);
    replaceBlockInputsWithGraphInputs(false_block);
    replaceBlockWithFallbackGraph(false_block, graph_->inputs());

    WithInsertPoint wip{graph_->block()->param_node()->next()};
    Value* none_val = graph_->insertConstant(IValue());
    std::vector<Value*> checks;
    std::vector<Value*> zero_values;
    std::vector<Value*> nonzero_values;

    for (auto inp : graph_->inputs()) {
      if (auto profile_optional_node = getUse(inp, prim::profile_optional)) {
        if (profile_optional_node->i(attr::num_present) == 0 &&
            profile_optional_node->i(attr::num_none) != 0) {
          auto check = graph_->insert(aten::__is__, {inp, none_val})->node();
          checks.push_back(check->output());
          profiled_none_.insert(inp);
        }
        removeProfiledOptionalUses(inp);
        continue;
      }

      if (inp->uses().size() == 0 || !inp->type()->cast<TensorType>()) {
        continue;
      }

      // TODO: check multiple uses ?
      auto pout = getUse(inp, prim::profile);
      if (!pout) {
        continue;
      }

      auto pttp = pout->ty(attr::profiled_type)->expect<TensorType>();
      if (!pttp->undefined().has_value()) {
        continue;
      }

      state_[inp] = *pttp->undefined() ? State::Zero : State::Nonzero;

      if (*pttp->undefined()) {
        zero_values.push_back(inp);
      } else {
        nonzero_values.push_back(inp);
      }
    }

    // unable to specialize any of the inputs
    if (nonzero_values.size() == 0 && zero_values.size() == 0) {
      GRAPH_DUMP("Unable to add any specialization guards", graph_);
      versioning_if->destroy();
      // the checks we inserted will be cleaned up
      // by any subsequent DCE pass
      return nullptr;
    }

    Node* nonzero_check = graph_->insert(prim::AutogradAllNonZero, {})->node();
    for (Value* v : nonzero_values) {
      nonzero_check->addInput(v);
    }
    checks.push_back(nonzero_check->output());

    Node* zero_check = graph_->insert(prim::AutogradAllZero, {})->node();
    for (Value* v : zero_values) {
      zero_check->addInput(v);
    }
    checks.push_back(zero_check->output());

    Value* bool_list =
        graph_->insertNode(graph_->createList(BoolType::get(), checks))
            ->output();
    Value* conjunction = graph_->insert(aten::all, {bool_list});

    versioning_if->addInput(conjunction);
    graph_->insertNode(versioning_if);

    auto ret = graph_->return_node();
    for (size_t i = 0; i < ret->inputs().size(); i++) {
      auto ogo = ret->input(i);
      auto ngo = versioning_if->output(i);
      ngo->copyMetadata(ogo);
      ret->replaceInput(i, ngo);
    }

    // We've created:
    // succesful_checks = Guards(...)
    // if (succesful_checks)
    // -> optimized graph
    // else:
    // -> fallback graph
    // original graph
    //
    // Remove the dead original graph
    for (auto it = graph_->block()->nodes().reverse().begin();
         *it != versioning_if;) {
      Node* n = *it;
      it++;
      n->destroy();
    }

    GRAPH_DUMP("After guardSpecializations", graph_);
    return versioning_if;
  }

  void specializeAutogradOps(Block* block) {
    for (auto it = block->nodes().begin(); it != block->nodes().end(); ++it) {
      auto n = *it;
      switch (n->kind()) {
        case prim::AutogradAdd: {
          auto a = n->input(0);
          auto b = n->input(1);
          // if one is Autograd zero, we can just drop the add
          if (state_[a] == State::Zero) {
            // Zero + b == b
            n->output()->replaceAllUsesWith(b);
            it.destroyCurrent();
          } else if (state_[b] == State::Zero) {
            // a + Zero == a
            n->output()->replaceAllUsesWith(a);
            it.destroyCurrent();
          } else if (
              state_[a] == State::Nonzero && state_[b] == State::Nonzero) {
            // when both are Nonzero, we can use a normal, optimizable add
            // instruction
            WithInsertPoint guard(n);
            auto* cOne = graph_->insertConstant(1);
            auto* add_node = graph_->insertNode(graph_->create(aten::add, 1));
            add_node->addInput(a);
            add_node->addInput(b);
            add_node->addInput(cOne);
            auto* add_output = add_node->output();
            add_output->setType(n->output()->type());
            state_[add_output] = State::Nonzero;
            n->output()->replaceAllUsesWith(add_output);
            it.destroyCurrent();
          } else {
            // otherwise we have conditionally-Nonzero things, and we need
            // to actually run an AutogradAdd which will guard for Zeros
            // so we leave the op as is
            state_[n->output()] = State::Unknown;
          }
        } break;
        case prim::AutogradZero: {
          state_[n->output()] = State::Zero;
        } break;
        case prim::profile: {
          // this a profile node on a tensor use
          // if we decided to specialize this graph
          // its input may have undefinedness info
          // otherwise it should be Unknown
          if (n->inputs().size() > 0) {
            state_[n->output()] = !state_.count(n->input())
                ? State::Unknown
                : state_[n->output()] = state_[n->input()];
          }
          break;
        }
        case prim::BailOut: {
          if (auto ptt = n->output()->type()->expect<TensorType>()) {
            state_[n->output()] = ptt->undefined()
                ? *ptt->undefined() ? State::Zero : State::Nonzero
                : State::Unknown;
          }
        } break;
        // Lowered GradOf block
        case prim::If: {
          auto if_input = n->input(0)->node();
          if (if_input->kind() == prim::AutogradAnyNonZero) {
            auto all_zeros = std::all_of(
                if_input->inputs().begin(),
                if_input->inputs().end(),
                [&](Value* v) { return state_[v] == State::Zero; });

            auto all_nonzeros = std::all_of(
                if_input->inputs().begin(),
                if_input->inputs().end(),
                [&](Value* v) { return state_[v] == State::Nonzero; });
            // Property 1: if all the gradInputs to the GradOf are Zero
            // then the gradOutputs are also zero and will be represented as
            // AutogradZero nodes
            if (all_zeros) {
              auto zero =
                  graph_->createAutogradZero()->insertAfter(n)->output();
              state_[zero] = State::Zero;
              for (auto o : n->outputs()) {
                o->replaceAllUsesWith(zero);
              }
              it.destroyCurrent();
              break;
            }

            specializeGradSumToSize(n->blocks().at(0));
            if (all_nonzeros) {
              auto body = n->blocks().at(0);
              // hoist the nodes in the GradOf body to be before the linear
              // block
              for (auto it = body->nodes().begin();
                   it != body->nodes().end();) {
                auto block_node = *it++;
                block_node->moveBefore(n);
              }

              for (size_t i = 0; i < n->outputs().size(); ++i) {
                n->outputs().at(i)->replaceAllUsesWith(body->outputs().at(i));
                state_[body->outputs().at(i)] = State::Nonzero;
              }
              it.destroyCurrent();
              break;
            }
          }

          for (auto o : n->outputs()) {
            state_[o] = State::Unknown;
          }
          break;
        }
        default:
          for (auto o : n->outputs()) {
            state_[o] = State::Unknown;
          }
          break;
      }
    }
  }

  void specializeGradSumToSize(Block* b) {
    for (auto it = b->nodes().begin(); it != b->nodes().end(); ++it) {
      Node* n = *it;
      if (n->kind() == aten::_grad_sum_to_size) {
        if (n->input(1)->mustBeNone() || profiled_none_.count(n->input(1))) {
          n->output()->replaceAllUsesWith(n->input(0));
          it.destroyCurrent();
        }
      }
    }
  }

  std::shared_ptr<Graph> graph_;
  std::unordered_set<Value*> profiled_none_;
  std::unordered_map<Value*, State> state_;
};

// propagate autograd zero information through a gradient graph and
// remove grad_of blocks if present.
// Note: this is a very limited pass. It only propagates autograd zeros for
// operations generated by the symbolic autodiff code and cleans up
// AutogradAdds when possible. Outputs of other nodes are conservatively
// marked Unknown and not optimized.
void specializeAutogradZero(std::shared_ptr<Graph> g) {
  AutogradZeroSpecializer azs(std::move(g));
  azs.run();
}

} // namespace jit
} // namespace torch
