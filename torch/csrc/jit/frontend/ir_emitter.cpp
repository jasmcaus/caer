#include <torch/csrc/jit/frontend/ir_emitter.h>

#include <c10/util/Exception.h>
#include <c10/util/StringUtil.h>
#include <torch/csrc/jit/api/function_impl.h>
#include <torch/csrc/jit/frontend/canonicalize_modified_loop.h>
#include <torch/csrc/jit/frontend/convert_to_ssa.h>
#include <torch/csrc/jit/frontend/parser.h>
#include <torch/csrc/jit/frontend/schema_matching.h>
#include <torch/csrc/jit/frontend/script_type_parser.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/annotate_warns.h>
#include <torch/csrc/jit/passes/canonicalize.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/inline_forked_closures.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/passes/lift_closures.h>
#include <torch/csrc/jit/passes/lower_tuples.h>
#include <torch/csrc/jit/passes/normalize_ops.h>
#include <torch/csrc/jit/runtime/interpreter.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <torch/csrc/jit/runtime/slice_indices_adjust.h>
#include <torch/csrc/jit/testing/hooks_for_testing.h>

#include <torch/csrc/jit/ir/constants.h>

#include <c10/util/Optional.h>

#include <atomic>
#include <climits>
#include <set>
#include <stack>

namespace torch {
namespace jit {

using FunctionTable = std::unordered_map<std::string, Function&>;
using ValueTable = std::unordered_map<std::string, SugaredValuePtr>;
using TypeTable = std::unordered_map<std::string, TypePtr>;
using AttributeMap = std::unordered_map<std::string, Const>;
using ListAttributeMap = std::unordered_map<std::string, std::vector<Const>>;

struct Refinement {
  Refinement(std::string identifier, TypePtr type)
      : identifier_(std::move(identifier)), type_(std::move(type)) {}
  const std::string& identifier() const {
    return identifier_;
  }
  TypePtr type() const {
    return type_;
  }

 private:
  std::string identifier_;
  TypePtr type_;
};

struct RefinementSet {
  // When a comparison like x is None is made, we associate type refinements
  // with its true value and its false value. If a boolean that has refinements
  // associated with it is used in a conditional of an if statement, the true
  // and false refinements are inserted into the corresponding blocks
  using Refinements = std::vector<Refinement>;

  RefinementSet(Refinements true_refinements, Refinements false_refinements)
      : true_refinements_(std::move(true_refinements)),
        false_refinements_(std::move(false_refinements)) {}
  RefinementSet(Refinement single) : RefinementSet({std::move(single)}, {}) {}
  RefinementSet(Refinement single_true, Refinement single_false)
      : RefinementSet(
            Refinements({std::move(single_true)}),
            Refinements({std::move(single_false)})) {}
  RefinementSet() = default; // empty
  RefinementSet And(const RefinementSet& rhs) const {
    // if the result of an AND is true, both a & b had to be true,
    // so we take the union of a.true_refinements and b.true_refinements.
    // if the result is false, either a or b could have been false,
    // so we take their intersection.
    return RefinementSet(
        unionSet(true_refinements_, rhs.true_refinements_),
        intersectSet(false_refinements_, rhs.false_refinements_));
  }
  RefinementSet Or(const RefinementSet& rhs) const {
    // if the result of an OR is true, either a & b could have been true,
    // so we take the intersection of a.true_refinements & b.true_refinements.
    // if the result is false, both a and b had to be false,
    // so we take their union.
    return RefinementSet(
        intersectSet(true_refinements_, rhs.true_refinements_),
        unionSet(false_refinements_, rhs.false_refinements_));
  }

  RefinementSet Not() const {
    return RefinementSet(false_refinements_, true_refinements_);
  }
  const std::vector<Refinement> activeRefinements() const {
    return true_refinements_;
  }

 private:
  static bool sameVar(const Refinement& a, const Refinement& b) {
    return a.identifier() == b.identifier();
  }
  static Refinements unionSet(const Refinements& a, const Refinements& b) {
    Refinements result = a;
    for (const Refinement& r : b) {
      auto it =
          std::find_if(result.begin(), result.end(), [&](const Refinement& e) {
            return e.identifier() == r.identifier();
          });
      if (it == result.end()) {
        result.push_back(r);
      } else if (*it->type() != *r.type()) {
        // we only keep refinements when they exactly match one
        // refinement type, for instance, we do not attempt to refine:
        // isinstance(x, float) and isinstance(x, int)
        result.erase(it);
      }
    }
    return result;
  }
  static Refinements intersectSet(const Refinements& a, const Refinements& b) {
    Refinements result;
    for (const Refinement& r : a) {
      auto it = std::find_if(b.begin(), b.end(), [&](const Refinement& e) {
        return e.identifier() == r.identifier();
      });
      if (it != b.end() && r.type() == it->type()) {
        result.push_back(r);
      }
    }
    return result;
  }

  Refinements true_refinements_;
  Refinements false_refinements_;
};

struct CondValue {
  CondValue(
      Value* value,
      RefinementSet refinements,
      c10::optional<bool> static_if)
      : value_(value),
        refinements_(std::move(refinements)),
        static_if_(static_if) {}
  CondValue(
      Graph& g,
      const SourceRange& loc,
      bool static_value,
      RefinementSet refinements)
      : value_(g.insertConstant(static_value, loc)),
        refinements_(std::move(refinements)),
        static_if_(static_value) {}
  Value* value() const {
    return value_;
  }
  const RefinementSet& refinements() const {
    return refinements_;
  }
  c10::optional<bool> staticIf() const {
    return static_if_;
  }

 private:
  Value* value_;
  RefinementSet refinements_;
  c10::optional<bool>
      static_if_; // certain expression cause us to emit a static if statement
                  // this value is present if this is the case.
                  // this is not equivalent to value_ being a constant
                  // it is possible for value_ to be constant but for
                  // the expression that produced it to not trigger the
                  // static if behavior. e.g. use of a variable assigned
                  // to a constant
};

enum NoneStatus { ALWAYS, MAYBE, NEVER };
NoneStatus canBeNone(Value* v) {
  if (v->node()->mustBeNone()) {
    return ALWAYS;
  }
  if (v->type()->kind() == OptionalType::Kind) {
    return MAYBE;
  }
  return NEVER;
}

static Value* asSimple(const SugaredValuePtr& value) {
  if (SimpleValue* sv = dynamic_cast<SimpleValue*>(value.get())) {
    return sv->getValue();
  }
  return nullptr;
}

static std::shared_ptr<MagicMethod> makeMagic(
    const std::string& name,
    SugaredValuePtr base) {
  return std::make_shared<MagicMethod>(name, base);
}

// Auxiliary data structure for desugaring variable binding into our always
// explicitly scoped language as we descend down nested control structures in
// the frontend (which themselves don't introduce scopes)
//
// The Environment keeps track of two tables, one for values which are not first
// class and a type table for values which are. When a first class value
// is set in the environment, we emit a prim::Store which sets the
// name of the variable to appropriate type, and when a first-class value is
// referenced we emit a prim::Load that generates a value of the appropriate
// type.
//
// a = 1
// print(a)
// becomes:
// = prim::Store[name="a"](%a.1)
// %a : int = prim::Load[name="a"]()
// prim::Print(%a)

struct Environment {
  Environment(
      Function& method,
      ResolverPtr resolver,
      Block* b,
      std::shared_ptr<Environment> next = nullptr)
      : method(method),
        resolver(std::move(resolver)),
        b(b),
        next(std::move(next)) {}

  Function& method;
  ResolverPtr resolver;
  std::unordered_map<std::string, std::function<std::string()>> error_messages;
  Block* b;

  std::shared_ptr<Environment> next;

  // set type error in the lowest environment. if the variable is used after an
  // error has been set, then we will use the more informative error message
  void setVariableTypeError(
      const std::string& name,
      std::function<std::string()> msg) {
    auto runner = this;
    while (runner->next) {
      runner = runner->next.get();
    }
    runner->error_messages[name] = std::move(msg);
  }

  // see if type error has been set for a variable
  c10::optional<std::string> findVariableTypeError(const std::string& name) {
    auto runner = this;
    while (runner->next) {
      runner = runner->next.get();
    }
    auto msg = runner->error_messages.find(name);
    if (msg != runner->error_messages.end()) {
      return msg->second();
    } else {
      return c10::nullopt;
    }
  }

  SugaredValuePtr insertLoad(const std::string& name, const TypePtr& type) {
    auto g = b->owningGraph();
    auto load = g->insertNode(g->createLoad(name, type));
    if (meaningfulName(name)) {
      load->output()->setDebugName(name);
    }
    return std::make_shared<SimpleValue>(load->output());
  }

  // note: type is not always the same as v->type(), e.g.
  // type: Optional[Tensor]
  // v->type(): Tensor
  void insertStore(
      const std::string& name,
      const SourceRange& loc,
      Value* v,
      TypePtr type) {
    auto g = b->owningGraph();
    g->insertNode(g->createStore(name, v))->setSourceRange(loc);
    type_table[name] = std::move(type);
  }

  SugaredValuePtr findInThisFrame(const std::string& name) {
    auto it = value_table.find(name);
    if (it != value_table.end()) {
      return it->second;
    }
    auto it2 = type_table.find(name);
    if (it2 != type_table.end()) {
      return insertLoad(name, it2->second);
    }
    return nullptr;
  }

  SugaredValuePtr findInParentFrame(const std::string& name) {
    return next ? next->findInAnyFrame(name) : nullptr;
  }

  void setType(const std::string& name, TypePtr type) {
    type_table[name] = std::move(type);
  }

  SugaredValuePtr findInAnyFrame(const std::string& name) {
    for (auto runner = this; runner; runner = runner->next.get()) {
      if (auto r = runner->findInThisFrame(name)) {
        return r;
      }
    }
    return nullptr;
  }

  Block* block() {
    return b;
  }

  void setVar(const SourceRange& loc, const std::string& name, Value* value) {
    setSugaredVar(
        loc,
        name,
        std::make_shared<SimpleValue>(value),
        /*annotated_type=*/nullptr);
  }

  void setSugaredVar(
      const SourceRange& loc,
      const std::string& name,
      SugaredValuePtr value,
      TypePtr annotated_type) {
    Value* as_simple_value = asSimple(value);
    if (as_simple_value && !as_simple_value->hasDebugName() &&
        meaningfulName(name) &&
        // note: if the value wasn't defined in this block, we might be giving a
        // name only used inside this block to a value outside of this. this is
        // not normally helpful for debugging and causes import/export jitter.
        as_simple_value->node()->owningBlock() == block()) {
      as_simple_value->setDebugName(name);
    }
    // prevent re-assignment involving any sugared values
    // any reassignment like:
    // a = ...
    // while ...
    //   a = ..
    // requires 'a' to be first-class in the graph since its value depends on
    // control flow
    if (auto parent = findInParentFrame(name)) {
      if (annotated_type) {
        throw ErrorReport(loc)
            << "Attempting to declare and annotate the type of variable '"
            << name << "' but it is already defined in an outer block";
      }
      if (!as_simple_value) {
        throw ErrorReport(loc)
            << "Cannot re-assign '" << name << "' to a value of type "
            << value->kind() << " because " << name
            << " is not a first-class value.  Only reassignments to first-class values are allowed";
      }
      Value* simple_parent = asSimple(parent);
      if (!simple_parent) {
        throw ErrorReport(loc)
            << "Cannot re-assign '" << name << "' because it has type "
            << value->kind() << " and " << name
            << " is not a first-class value.  Only reassignments to first-class values are allowed";
      }

      auto parent_type = unshapedType(simple_parent->type());
      as_simple_value = tryConvertToType(
          loc,
          *b->owningGraph(),
          parent_type,
          as_simple_value,
          /*allow_conversions=*/true);
      std::stringstream why_not;
      if (!as_simple_value->type()->isSubtypeOfExt(parent_type, &why_not)) {
        auto error = ErrorReport(loc);
        error << "Variable '" << name << "' previously has type "
              << simple_parent->type()->repr_str()
              << " but is now being assigned to a value of type "
              << as_simple_value->type()->repr_str();

        // Special-cased error msg if we're trying to assign to a tensor list.
        if (simple_parent->type()->kind() == TypeKind::ListType &&
            as_simple_value->type()->kind() == TypeKind::ListType) {
          error << "\nEmpty lists default to List[Tensor]. Add a variable "
                   "annotation to the assignment to create an empty list "
                   "of another type (torch.jit.annotate(List[T, []]) where T "
                   "is the type of elements in the list for Python 2)";
        }
        error << "\n" << why_not.str();
        throw error;
      }
    }
    if (as_simple_value) {
      if (!annotated_type) {
        annotated_type = as_simple_value->type();
      }
      if (!as_simple_value->type()->isSubtypeOf(annotated_type)) {
        throw ErrorReport(loc)
            << "Variable '" << name << "' is annotated with type "
            << annotated_type->repr_str()
            << " but is being assigned to a value of type "
            << as_simple_value->type()->repr_str();
      }
      insertStore(name, loc, as_simple_value, annotated_type);
    } else {
      value_table[name] = std::move(value);
    }
  }

  SugaredValuePtr getSugaredVar(const Ident& ident, bool required = true) {
    return getSugaredVar(ident.name(), ident.range());
  }
  Value* getVar(const Ident& ident) {
    return getSugaredVar(ident)->asValue(ident.range(), method);
  }

  void throwVarNotFoundError(
      const std::string& ident,
      const SourceRange& range) {
    // check if this value was not emitted in an if statement because of a
    // type mismatch. if it was, then we print a more informative error msg
    if (auto msg = findVariableTypeError(ident)) {
      throw ErrorReport(range) << *msg << "and was used here";
    }
    throw ErrorReport(range) << "undefined value " << ident;
  }

  SugaredValuePtr getSugaredVar(
      const std::string& ident,
      const SourceRange& range,
      bool required = true) {
    auto retval = findInAnyFrame(ident);

    if (!retval) {
      static std::unordered_map<std::string, SugaredValuePtr> globals = {
          {"print", std::make_shared<PrintValue>()},
          {"tuple", SpecialFormValue::create(prim::TupleConstruct)},
          {"float",
           makeMagic(
               "__float__",
               std::make_shared<CastValue>(FloatType::get(), aten::Float))},
          {"int",
           makeMagic(
               "__int__",
               std::make_shared<CastValue>(IntType::get(), aten::Int))},
          {"bool",
           makeMagic(
               "__bool__",
               std::make_shared<CastValue>(BoolType::get(), aten::Bool))},
          {"str",
           makeMagic(
               "__str__",
               std::make_shared<CastValue>(StringType::get(), aten::str))},
          {"getattr", SpecialFormValue::create(prim::GetAttr)},
          {"hasattr", SpecialFormValue::create(prim::HasAttr)},
          {"isinstance", SpecialFormValue::create(prim::isinstance)},
          // todo(zach): remove when we can correctly export torch.full via ONNX
          // or we have implicit conversion that can convert numbers to tensors
          {"_to_tensor",
           std::make_shared<CastValue>(TensorType::get(), prim::NumToTensor)},
          {"len",
           makeMagic(
               "__len__",
               std::make_shared<BuiltinFunction>(aten::len, at::nullopt))},
          {"hex",
           makeMagic(
               "__hex__",
               std::make_shared<BuiltinFunction>(aten::hex, at::nullopt))},
          {"oct",
           makeMagic(
               "__oct__",
               std::make_shared<BuiltinFunction>(aten::oct, at::nullopt))},
          {"round",
           makeMagic(
               "__round__",
               std::make_shared<BuiltinFunction>(aten::round, at::nullopt))},
          {"hash", std::make_shared<BuiltinFunction>(aten::hash, at::nullopt)},
          {"id", std::make_shared<BuiltinFunction>(prim::id, at::nullopt)},
          {"min", std::make_shared<BuiltinFunction>(prim::min, at::nullopt)},
          {"max", std::make_shared<BuiltinFunction>(prim::max, at::nullopt)},
          {"abs", std::make_shared<BuiltinFunction>(prim::abs, at::nullopt)},
          {"all", std::make_shared<BuiltinFunction>(aten::all, at::nullopt)},
          {"divmod",
           std::make_shared<BuiltinFunction>(aten::divmod, at::nullopt)},
          {"list", SpecialFormValue::create(prim::list)},
          {"ord", std::make_shared<BuiltinFunction>(aten::ord, at::nullopt)},
          {"chr", std::make_shared<BuiltinFunction>(aten::chr, at::nullopt)},
          {"bin", std::make_shared<BuiltinFunction>(aten::bin, at::nullopt)},
          {"range", SpecialFormValue::create(prim::range)},
          {"zip", SpecialFormValue::create(prim::zip)},
          {"enumerate", SpecialFormValue::create(prim::enumerate)},
          {"rangelist",
           std::make_shared<BuiltinFunction>(prim::rangelist, at::nullopt)},
          {"sorted",
           std::make_shared<BuiltinFunction>(aten::sorted, at::nullopt)},
          // Only AssertionError is bound so that we can use it from emitAssert,
          // all other exceptions should be resolved at the Python level
          {"AssertionError",
           std::make_shared<ExceptionValue>("AssertionError")},
      };
      auto it = globals.find(ident);
      if (it != globals.end()) {
        retval = it->second;
      }
    }

    if (!retval) {
      if (auto type = resolver->resolveType(ident, range)) {
        if (auto tuple_type = type->cast<TupleType>()) {
          retval = std::make_shared<NamedTupleConstructor>(tuple_type);
        }
      }
    }

    if (!retval) {
      retval = resolver->resolveValue(ident, method, range);
    }

    if (!retval) {
      if (auto type = resolver->resolveType(ident, range)) {
        if (auto class_type = type->cast<ClassType>()) {
          retval = std::make_shared<ClassValue>(class_type);
        }
      }
    }

    if (!retval && required) {
      throwVarNotFoundError(ident, range);
    }
    return retval;
  }

  Value* getVar(const std::string& ident, const SourceRange& range) {
    return getSugaredVar(ident, range)->asValue(range, method);
  }

  void removeVar(const Ident& ident, bool check_if_removed = false) {
    bool removed = false;

    for (auto runner = this; runner; runner = runner->next.get()) {
      auto a = runner->value_table.erase(ident.name());
      auto b = runner->type_table.erase(ident.name());
      removed = a || b;
    }

    if (check_if_removed && !removed) {
      throwVarNotFoundError(ident.name(), ident.range());
    }
  }

  std::vector<std::string> definedVariables() {
    std::vector<std::string> result;
    for (auto& kv : type_table) {
      result.push_back(kv.first);
    }
    return result;
  }

 private:
  TypeTable type_table;
  ValueTable value_table;
};

template <class T>
static Value* materializeConstant(
    T val,
    Graph& graph,
    const SourceRange& r,
    std::unordered_map<T, Value*>& map) {
  auto existing_constant = map.find(val);
  if (existing_constant != map.end()) {
    return existing_constant->second;
  }

  WithInsertPoint guard(graph.block()->nodes().front());
  auto new_constant = graph.insertConstant(val, r);
  map[val] = new_constant;

  return new_constant;
}

inline bool isSupportedListElementType(const TypePtr& type) {
  return type->isSubtypeOf(TensorType::get()) ||
      type->isSubtypeOf(NumberType::get());
}

// Information for each def being emitted.
// Defs can be nested to support closures so we need a stack of this information
// Currently records information about the functions return type.
struct DefContext {
  TypePtr declared_return_type_; // nullptr if not annotated
  TypePtr merged_return_type_; // nullptr if a Return has not been seen yet
};

enum class LoopStatus { NOT_IN_LOOP, IN_LOOP, IN_UNROLLED_LOOP };

struct WithLoopStatus {
  WithLoopStatus(LoopStatus* prev, LoopStatus new_status) {
    prev_value_ = *prev;
    prev_ptr_ = prev;
    *prev = new_status;
  }
  ~WithLoopStatus() {
    *prev_ptr_ = prev_value_;
  }

 private:
  LoopStatus* prev_ptr_;
  LoopStatus prev_value_;
};

struct to_ir {
  to_ir(
      const Def& def,
      ResolverPtr resolver_,
      const Self* self,
      Function& method) // method being constructed
      : method(method),
        graph(method.graph()),
        resolver(std::move(resolver_)),
        typeParser_(resolver),
        environment_stack(nullptr) {
    AT_ASSERT(resolver);
    pushFrame(graph->block(), /*starts_def=*/true);

    // Type annotations exclude explicitly typing the "self" parameter, so in
    // the case that this is a method with self we expect one fewer parameter
    // annotation than the number of parameters this Def takes.
    if (self && def.decl().params().size() == 0) {
      throw ErrorReport(def.decl().params().range())
          << "methods must have a self argument";
    }
    method.setSchema(emitDef(def, self, graph->block()));

    // NB ORDERING: SSA conversion has to occur before
    // lifting of closures and forks, this way closures are converted
    // to SSA while part of their original graph, and closures are ready to
    // be inlined into forked closures
    ConvertToSSA(graph);
    // convert loops with an iter and body condition specified to
    // python-recognize while loops. we do this so they can be exported,
    // and run the pass early to avoid jitter. Like conversion to SSA,
    // it only needs to run once.
    CanonicalizeModifiedLoops(graph);

    // Convert Ops to a Normalized Form
    NormalizeOps(graph);

    runCleanupPasses(graph);
  }

 private:
  Function& method;
  std::shared_ptr<Graph> graph;
  ResolverPtr resolver;
  std::unordered_map<int64_t, Value*> integral_constants;
  std::unordered_map<double, Value*> fp_constants;
  std::unordered_set<Block*> exit_blocks;
  ScriptTypeParser typeParser_;
  LoopStatus loop_status_ = LoopStatus::NOT_IN_LOOP;

  // Singly-linked list of environments. This top element contains a member
  // `next` that points to the most immediate enclosing scope's value.
  std::shared_ptr<Environment> environment_stack;
  std::vector<DefContext> def_stack_;
  size_t temp_name_count_ = 0;
  std::string createTempName(const std::string& prefix) {
    return prefix + c10::to_string(temp_name_count_++);
  }

  void pushFrame(Block* b, bool starts_def = false) {
    if (starts_def) {
      def_stack_.emplace_back();
    }
    environment_stack =
        std::make_shared<Environment>(method, resolver, b, environment_stack);
  }
  std::shared_ptr<Environment> popFrame(bool ends_def = false) {
    auto old_frame = environment_stack;
    environment_stack = environment_stack->next;
    if (ends_def) {
      def_stack_.pop_back();
    }
    return old_frame;
  }

  // If the graph might not return, add an implicit None return at the end
  void handleMaybeNoReturn(const Def& def, Block* block) {
    auto decl_ret = def_stack_.back().declared_return_type_;
    if (exit_blocks.count(block) == 0) {
      auto decl_ret = def_stack_.back().declared_return_type_;
      if (decl_ret && decl_ret != NoneType::get()) {
        throw ErrorReport(def.range())
            << "Function was not annotated as having type None, but does not "
            << "return along all paths";
      }
      WithInsertPoint b(*block->nodes().end());
      emitReturn(Return::create(
          def.range(), Expr(Compound::create(TK_NONE, def.range(), {}))));
    } else {
      // if we haven't seen any return statements, but the graph block exits
      // (the function always throws) then we accept the declared return type if
      // it exists or set it to none
      if (def_stack_.back().merged_return_type_ == nullptr) {
        def_stack_.back().merged_return_type_ =
            decl_ret != nullptr ? decl_ret : NoneType::get();
      }
    }
  }

  FunctionSchema emitDef(const Def& def, const Self* self, Block* block) {
    auto schema = typeParser_.parseSchemaFromDef(def, bool(self));
    // TODO need guards on init returning none
    if (schema.returns().size() == 1) {
      def_stack_.back().declared_return_type_ = schema.returns().at(0).type();
    }
    std::vector<Argument> arguments =
        emitFormalArguments(def, self, schema, block);

    // body
    auto stmts_list = def.statements();
    emitStatements(stmts_list.begin(), stmts_list.end());
    handleMaybeNoReturn(def, block);
    std::vector<Argument> returns = {emitOutput(def.range(), schema, block)};
    return {def.name().name(), "", std::move(arguments), std::move(returns)};
  }

  // see [setstate type]
  static TypePtr getTypeForSetStateArg(const Def& def, const Self* self) {
    TORCH_CHECK(self, "Expected __setstate__ to have a `self` argument");
    auto getstate = self->getClassType()->findMethod("__getstate__");
    if (!getstate) {
      throw ErrorReport(def.range())
          << "`__setstate__` defined but not `__getstate__`. "
          << "You must have both defined on a ScriptModule "
          << "to customize serialization.\n"
          << "Did you forget to use `@torch.jit.export`?";
    }
    getstate->ensure_defined();
    return self->getClassType()
        ->getMethod("__getstate__")
        .getSchema()
        .returns()
        .at(0)
        .type();
  }

  // see [setstate type]
  static bool shouldDeriveSetStateType(
      const Def& def,
      const FunctionSchema& schema) {
    const bool noTypeAnnotations = std::all_of(
        schema.arguments().begin(),
        schema.arguments().end(),
        [](const Argument& arg) { return arg.is_inferred_type(); });

    bool shouldInfer = def.name().name() == "__setstate__" && noTypeAnnotations;
    if (!shouldInfer) {
      return false;
    }

    // Do some additional basic validation that the __setstate__ func is
    // well-formed
    TORCH_INTERNAL_ASSERT(def.name().name() == "__setstate__");
    const auto numDeclParams = def.decl().params().size();
    if (numDeclParams != 2) {
      throw ErrorReport(def.range())
          << "Expected 2 arguments for `__setstate__`, got: " << numDeclParams;
    }
    return true;
  }

  std::vector<Argument> emitFormalArguments(
      const Def& def,
      const Self* self,
      const FunctionSchema& schema,
      Block* block) {
    std::vector<Argument> arguments; // for schema
    // inputs
    auto it = def.decl().params().begin();
    auto end = def.decl().params().end();
    auto expected_annotation_size = def.decl().params().size();
    if (self) {
      expected_annotation_size--;
    }
    if (schema.arguments().size() != expected_annotation_size) {
      throw ErrorReport(def.decl().params().range())
          << "Number of type annotations for"
          << " function parameters (" << schema.arguments().size() << ")"
          << " does not match the number of parameters on the function ("
          << expected_annotation_size << ")!";
    }

    if (self) {
      AT_ASSERT(it != end);
      const auto& name = (*it).ident().name();
      Value* new_input = block->addInput()->setDebugName(name);
      environment_stack->setSugaredVar(
          (*it).ident().range(),
          name,
          self->makeSugared(new_input),
          /*annotated_type=*/nullptr);
      arguments.emplace_back(name, new_input->type());
      ++it;
    }

    // [setstate type]
    // __setstate__ is special, because if the user leaves it un-annotated we
    // will derive the type for `state` from the output type of __getstate__.
    // This is necessary so that we can allow submodules to appear in `state`.
    bool shouldDeriveType = shouldDeriveSetStateType(def, schema);
    size_t arg_annotation_idx = 0;
    for (; it != end; ++it) {
      auto& name = (*it).ident().name();
      // Add the input to the graph
      Value* new_input = block->addInput();
      if (meaningfulName(name)) {
        new_input->setDebugName(name);
      }
      // Record the type for the schema and set the Type on the Value*
      auto arg = schema.arguments().at(arg_annotation_idx++);
      if (shouldDeriveType) {
        TORCH_INTERNAL_ASSERT(schema.arguments().size() == 1);
        const auto& inferredStateType = getTypeForSetStateArg(def, self);
        arg = arg.cloneWithType(inferredStateType);
      }

      arguments.push_back(arg);
      new_input->setType(arguments.back().type());

      // NB: set type of new_input before setVar call so the Store is
      // typed appropriately
      environment_stack->setVar((*it).ident().range(), name, new_input);
    }
    return arguments;
  }

  Argument emitOutput(
      const SourceRange& range,
      const FunctionSchema& schema,
      Block* block) {
    // handleMaybeNoReturn ensures that merged_return_type_ is always set
    auto ret_type = def_stack_.back().merged_return_type_;
    TORCH_INTERNAL_ASSERT(ret_type);

    // in the ConvertToSSA pass, prim::ReturnStmts are lowered so that the
    // correct return value is set. Until then, we have a correctly-typed
    // placeholder return value. This is needed so that closures & graphs
    // are correctly typed.
    auto placeholder_return =
        graph->insertNode(graph->createUninitialized(ret_type))->output();
    block->registerOutput(placeholder_return);
    return Argument("", def_stack_.back().merged_return_type_);
  }

  void emitStatements(const List<Stmt>& statements) {
    return emitStatements(statements.begin(), statements.end());
  }

  // XXX: Right now closures are not generically implemented and are only used
  // as an intermediate form for special tasks, like defining gradients or
  // forked functions.
  //
  // There are several unfinished aspects that make them unusable generally
  // 1. We do not have a type, ivalue, operator to represent prim::Closure, so
  // closure_node has type None
  // 2. There is no export logic for it yet, so it cannot be
  // exported/python_printed
  // 3. There is nothing preventing the assignment of already existing variables
  // inside the closures
  //    the changes to those variables will just get forgotten.
  // 4. There is no parsing support in frontend.py, this is intentional since it
  //    prevents people from accidentally using this feature.
  //
  // This function leaves in the graph something like:
  //
  //   %2 : None = prim::Closure()
  //     block0():
  //       %1 : Tensor = prim::DoSomething(%0)
  //       -> (%1)
  //
  // A separate pass is required to erase this closure and replace it with
  // something actually executable (see liftClosure and inlineForkedClosure).
  std::shared_ptr<ClosureValue> emitClosure(
      const std::function<void(Block*)>& emit_body) {
    Node* closure_node = graph->insertNode(graph->create(prim::Closure, 1));
    // it is not a real thing yet, so just say the type is None
    closure_node->output()->setType(NoneType::get());
    Block* block = closure_node->addBlock();
    WithLoopStatus loop_guard(&loop_status_, LoopStatus::NOT_IN_LOOP);
    {
      WithInsertPoint guard(block);
      pushFrame(block, /*starts_def=*/true);
      emit_body(block);
      popFrame(/*ends_def=*/true);
    }
    return std::make_shared<ClosureValue>(closure_node->output());
  }

  void emitClosure(const Def& def) {
    // invoked once the closure block is set as the environment
    auto emit_body = [&](Block* closure_block) {
      emitDef(
          def,
          nullptr,
          closure_block); // ignore schema return, we just wont use it for now
                          // since we never create a Method for the closure
    };
    auto closure_value = emitClosure(emit_body);
    environment_stack->setSugaredVar(
        def.name().range(),
        def.name().name(),
        closure_value,
        /*annotated_type=*/nullptr);
  }

  void checkBreakContinue(
      const SourceRange& loc,
      const std::string& stmt_name) {
    if (loop_status_ == LoopStatus::NOT_IN_LOOP) {
      throw ErrorReport(loc) << "SyntaxError: '" << stmt_name << "'"
                             << " outside loop";
    } else if (loop_status_ == LoopStatus::IN_UNROLLED_LOOP) {
      throw ErrorReport(loc)
          << "Because we emit iteration over modulelists or tuples as "
             "unrolled loops, we do not support break or continue inside the body of these loops";
    }
  }

  void emitBreak(const Break& stmt) {
    checkBreakContinue(stmt.range(), "break");
    auto break_node =
        graph->create(prim::BreakStmt, {}, 0)->setSourceRange(stmt.range());
    graph->insertNode(break_node);
  }

  void emitContinue(const Continue& stmt) {
    checkBreakContinue(stmt.range(), "continue");
    auto continue_node =
        graph->create(prim::ContinueStmt, {}, 0)->setSourceRange(stmt.range());
    graph->insertNode(continue_node);
  }

  void emitDelete(const Delete& stmt) {
    for (const auto& target : stmt.targets()) {
      if (target.kind() == TK_SUBSCRIPT) {
        Subscript subscript(target);
        const List<Expr>& subscript_exprs = subscript.subscript_exprs();
        if (subscript_exprs[0].kind() == TK_SLICE_EXPR) {
          throw ErrorReport(target.range())
              << "del statements only support deletion at a single index, "
                 "slicing is not supported"
                 " (see https://github.com/pytorch/pytorch/issues/31430)";
        }
        const SugaredValuePtr sv = emitSugaredExpr(subscript.value(), 1);
        const SourceRange& val_range = subscript.value().range();
        Value* idx = emitExpr(subscript_exprs[0]);
        Value* val = sv->asValue(val_range, method);

        // If val is a class instance, this is a method call to a type-specific
        // implementation of del defined in a __delitem__ method.
        if (auto cls = val->type()->cast<ClassType>()) {
          if (!cls->findMethod("__delitem__")) {
            throw ErrorReport(target.range())
                << "Class does not define __delitem__";
          }

          // Use MethodValue to call the method to handle recursion.
          MethodValue(val, "__delitem__")
              .call(stmt.range(), method, {idx}, {}, 0);
        } else {
          auto node = graph->create(aten::Delete, {val, idx}, 0)
                          ->setSourceRange(target.range());
          graph->insertNode(node);
        }
      } else if (target.kind() == TK_VAR) {
        Var var(target);
        environment_stack->removeVar(var.name(), /*check_if_removed=*/true);
      } else {
        throw ErrorReport(target.range())
            << "del statements are only supported for deleting"
               " list and dict items and variables";
      }
    }
  }

  void emitReturn(const Return& stmt) {
    Value* result = emitExpr(stmt.expr());
    TypePtr result_type = def_stack_.back().declared_return_type_;
    // result type is annotated, every return must convert to that type
    if (result_type) {
      // this guard skips implicit conversion from None -> Tensor for the return
      // type. otherwise forgetting a return a function returning a tensor will
      // cause a None to be converted to a tensor.
      if (!(result_type->isSubtypeOf(TensorType::get()) &&
            result->type()->isSubtypeOf(NoneType::get()))) {
        result = tryConvertToType(
            stmt.range(),
            *graph,
            result_type,
            result,
            /*allow_conversions=*/true);
      }

      if (!result->type()->isSubtypeOf(result_type)) {
        throw ErrorReport(stmt.range())
            << "Return value was annotated as having type "
            << result_type->repr_str() << " but is actually of type "
            << result->type()->repr_str();
      }
    } else {
      result_type = def_stack_.back().merged_return_type_;
      if (!result_type) {
        result_type = result->type();
      }
      auto merged_result_type = unifyTypes(result_type, result->type());
      if (!merged_result_type) {
        throw ErrorReport(stmt.range())
            << "Previous return statement returned a value of type "
            << result_type->repr_str()
            << " but this return statement returns a value of type "
            << result->type()->repr_str();
      }
      result_type = merged_result_type.value();
    }
    AT_ASSERT(result_type);

    def_stack_.back().merged_return_type_ = result_type;

    // If the annotated return type is Any and the result type is not Any,
    // cast the result to Any to facilitate type unification between return
    // statements on different code paths (e.g. different branches of an if,
    // body and containing scope of a loop).
    if (result_type == AnyType::get() && result->type() != AnyType::get()) {
      result = graph->insertUncheckedCast(result, result_type);
    }

    graph->insertNode(graph->create(prim::ReturnStmt, {result}, 0));
    exit_blocks.insert(environment_stack->block());
  }

  void emitStatements(
      List<Stmt>::const_iterator begin,
      List<Stmt>::const_iterator end) {
    for (; begin != end; ++begin) {
      auto stmt = *begin;
      ErrorReport::CallStack::update_pending_range(stmt.range());
      switch (stmt.kind()) {
        case TK_IF:
          emitIf(If(stmt));
          break;
        case TK_WHILE:
          emitWhile(While(stmt));
          break;
        case TK_FOR:
          emitFor(For(stmt));
          break;
        case TK_ASSIGN:
          emitAssignment(Assign(stmt));
          break;
        case TK_AUG_ASSIGN:
          emitAugAssignment(AugAssign(stmt));
          break;
        case TK_EXPR_STMT: {
          auto expr = ExprStmt(stmt).expr();
          emitSugaredExpr(expr, 0);
        } break;
        case TK_RAISE:
          emitRaise(Raise(stmt));
          break;
        case TK_ASSERT:
          emitAssert(Assert(stmt));
          break;
        case TK_RETURN: {
          emitReturn(Return(stmt));
        } break;
        case TK_CONTINUE: {
          emitContinue(Continue(stmt));
        } break;
        case TK_BREAK: {
          emitBreak(Break(stmt));
        } break;
        case TK_PASS:
          // Emit nothing for pass
          break;
        case TK_DEF:
          emitClosure(Def(stmt));
          break;
        case TK_DELETE:
          emitDelete(Delete(stmt));
          break;
        case TK_WITH:
          emitWith(With(stmt));
          break;
        default:
          throw ErrorReport(stmt)
              << "Unrecognized statement kind " << kindToString(stmt.kind());
      }
      // Found an exit statement in this block. The remaining statements aren't
      // reachable so we don't emit them.
      if (exit_blocks.count(environment_stack->block()))
        return;
    }
  }

  RefinementSet findIsNoneRefinements(
      const Expr& lhs,
      Value* lhs_value,
      const Expr& rhs,
      Value* rhs_value,
      int tok) {
    if (rhs.kind() != TK_NONE && lhs.kind() == TK_NONE) {
      // make 'None is var' into 'var is None'
      return findIsNoneRefinements(rhs, rhs_value, lhs, lhs_value, tok);
    }
    if (rhs.kind() != TK_NONE || lhs.kind() != TK_VAR) {
      return {};
    }
    // statement must be var {is, is not} None
    auto name = Var(lhs).name().name();
    // XXX - while it should in theory be possible to specialize
    // the `x is None` to know x has type NoneType, we have previously not
    // done this. Unfortunately, doing this will make the type None
    // propagate further in all loaded models. The handling of
    // unwrap_optional will fail in these cases since export did
    // not expect that the input would be none and an unannotated None.
    // cannot be passed to unwrapoptional To enable this,
    // we need to (1) implement a real casting operator
    // annotated(T, X) that stays in the graph and does the cast
    // and (2) only enable this OPTIONAL_NONE when loading newer
    // graphs because it is incompatible with older graphs.
    // Refinement none(name, RefinementKind::OPTIONAL_NONE);
    if (auto optional_type = lhs_value->type()->cast<OptionalType>()) {
      Refinement present(name, optional_type->getElementType());
      if (tok == TK_IS) {
        return RefinementSet({}, {present});
      } else { // TK_ISNOT
        return RefinementSet({present}, {});
      }
    }
    return RefinementSet();
  }

  CondValue emitCondExpr(const Expr& expr) {
    switch (expr.kind()) {
      case TK_AND:
      case TK_OR: {
        auto binop = BinOp(expr);
        return emitShortCircuitLogical(
            binop.range(), binop.lhs(), binop.rhs(), expr.kind() == TK_OR);
      }
      case TK_NOT: {
        CondValue v = emitCondExpr(Expr(expr.tree()->trees()[0]));
        Value* result = emitBuiltinCall(
            expr.range(), *graph, aten::__not__, {v.value()}, {});
        c10::optional<bool> static_if;
        if (v.staticIf()) {
          static_if = !*v.staticIf();
        }
        return CondValue(result, v.refinements().Not(), static_if);
      } break;
      case TK_IS:
      case TK_ISNOT: {
        // meta programming on AST for is/is not cases and emit branches base on
        auto cond_op = BinOp(expr);
        Value* lhs_val = emitExpr(cond_op.lhs());
        Value* rhs_val = emitExpr(cond_op.rhs());

        auto lhs_none = canBeNone(lhs_val);
        auto rhs_none = canBeNone(rhs_val);

        // Dispatch logic (A: ALWAYS, N: NEVER, M: MAYBE):
        //
        // AA, -> statically IS always holds, IS_NOT never holds
        // AN , NA-> statically IS_NOT always holds, IS never holds
        // MA, MM, MN, NM, NN, AM -> cannot prove anything statically
        bool its_is = expr.kind() == TK_IS;
        if (lhs_none == ALWAYS && rhs_none == ALWAYS) {
          return CondValue(*graph, expr.range(), its_is, {});
        } else if (
            (lhs_none == ALWAYS && rhs_none == NEVER) ||
            (lhs_none == NEVER && rhs_none == ALWAYS)) {
          // lhs_val/rhs_val with A/M: only emit never_none_branch
          return CondValue(*graph, expr.range(), !its_is, {});
        } else {
          auto kind = getNodeKind(expr.kind(), expr.get()->trees().size());
          Value* cond_value = emitBuiltinCall(
              expr.get()->range(),
              *method.graph(),
              kind,
              {lhs_val, rhs_val},
              {});
          auto refinements = RefinementSet(findIsNoneRefinements(
              cond_op.lhs(), lhs_val, cond_op.rhs(), rhs_val, expr.kind()));
          return CondValue(cond_value, refinements, c10::nullopt);
        }
      } break;
      default: {
        if (expr.kind() == TK_APPLY) {
          auto apply = Apply(expr);
          auto callee = Apply(expr).callee();
          if (callee.kind() == TK_VAR) {
            if (Var(callee).name().name() == "isinstance") {
              checkApplyNumInputs(apply, 2);
              return emitIsInstance(apply.inputs()[0], apply.inputs()[1]);
            }
            if (Var(callee).name().name() == "hasattr") {
              checkApplyNumInputs(apply, 2);
              return emitHasAttr(apply.inputs()[0], apply.inputs()[1]);
            }
          }
          auto sv = emitSugaredExpr(apply.callee(), 1);
          auto loc = apply.callee().range();
          if (auto special_form = dynamic_cast<SpecialFormValue*>(sv.get())) {
            if (special_form->form() == prim::isinstance) {
              checkApplyNumInputs(apply, 2);
              return emitIsInstance(apply.inputs()[0], apply.inputs()[1]);
            }
          }
        }
        auto expr_out = emitToBool(expr.range(), emitExpr(expr));
        c10::optional<bool> static_if = c10::nullopt;
        if (expr_out->node()->kind() == aten::is_scripting) {
          static_if = true;
        }
        // MetaCompile on boolean literals and constants
        if (auto maybe_ivalue = toIValue(expr_out)) {
          static_if = maybe_ivalue->toBool();
        }
        return CondValue(expr_out, RefinementSet({}), static_if);
      } break;
    }
  }

  std::shared_ptr<Environment> emitSingleIfBranch(
      Block* b,
      const List<Stmt>& branch,
      const RefinementSet& refinements) {
    pushFrame(b);
    WithInsertPoint guard(b);
    insertRefinements(branch.range(), refinements);
    emitStatements(branch);
    return popFrame();
  }

  Node* create(Symbol kind, const SourceRange& loc, size_t n_outputs) {
    return graph->create(kind, n_outputs)->setSourceRange(loc);
  }

  Value* emitTernaryIf(const TernaryIf& expr) {
    CondValue cond_value = emitCondExpr(expr.cond());
    auto true_expr = [&] { return emitExpr(expr.true_expr()); };
    auto false_expr = [&] { return emitExpr(expr.false_expr()); };
    return emitIfExpr(expr.range(), cond_value, true_expr, false_expr);
  }

  Value* emitListComprehension(const ListComp& lc, const TypePtr& type_hint) {
    const auto loc = lc.range();
    const auto targets_list = List<Expr>::create(lc.range(), {lc.target()});
    const auto itrs = List<Expr>::create(lc.range(), {lc.iter()});

    // If there is no type hint, and this is emitted over an iterable that is
    // unrolled and of length 0, then we emit a List of tensors
    Value* list_value = graph->insertNode(graph->create(prim::ListConstruct, 1))
                            ->output()
                            ->setType(ListType::ofTensors());
    bool type_set = false;
    if (type_hint) {
      if (!type_hint->cast<ListType>()) {
        throw ErrorReport(loc)
            << "Expected list type annotation for list comprehension"
               ", found "
            << type_hint->repr_str();
      }
      list_value->setType(type_hint);
      type_set = true;
    }

    // comprehension introduces it's own scope. no variable assigned
    // leaks into the rest of the graph
    Node* n =
        graph->insertNode(create(prim::ListComprehensionScope, lc.range(), 0));
    auto* comprehension_block = n->addBlock();
    pushFrame(comprehension_block);
    WithInsertPoint guard(comprehension_block);
    auto emit_body = [&]() {
      auto comprehension_out = emitExpr(lc.elt());
      if (!type_set) {
        list_value->setType(ListType::create(comprehension_out->type()));
        type_set = true;
      }
      NamedValue self = NamedValue(loc, "self", list_value);
      NamedValue input = NamedValue(loc, "", comprehension_out);
      emitBuiltinCall(loc, *graph, aten::append, {input}, {}, self);
    };
    emitFor(targets_list, itrs, loc, emit_body);
    popFrame();
    return list_value;
  }

  // Insert subtyping refinements
  void insertRefinements(const SourceRange& loc, const RefinementSet& ref) {
    for (const Refinement& r : ref.activeRefinements()) {
      Value* v = environment_stack->getVar(r.identifier(), loc);
      Value* new_v = graph->insertUncheckedCast(v, r.type());
      environment_stack->setVar(loc, r.identifier(), new_v);
    }
  }

  CondValue emitShortCircuitLogical(
      const SourceRange& loc,
      const Expr& first_expr,
      const Expr& second_expr,
      bool is_or) {
    CondValue lhs = emitCondExpr(first_expr);
    // if the continue expr in the short circuit is not evaluated,
    // than the const expression is False if the short circuit
    // is an `and` and True if the short circuit is an `or`.
    // `False and expr` -> False, `True or expr` -> True
    //
    // inserting it as a constant makes optimization easier

    // if it's an OR the first expr is emitted in the true branch
    // and the second expr in the false branch, if it's an AND the opposite
    auto get_const_expr = [&] { return graph->insertConstant(is_or, loc); };

    c10::optional<CondValue> rhs;
    auto get_continue_expr = [&] {
      rhs = emitCondExpr(second_expr);
      return rhs->value();
    };

    // if this is an OR, eval second expression if first expr is False
    // If this is an AND, eval second expression if first expr is True
    Value* new_result;
    c10::optional<RefinementSet> refinements;
    c10::optional<bool> static_if;
    if (is_or) {
      new_result = emitIfExpr(loc, lhs, get_const_expr, get_continue_expr);
      refinements = lhs.refinements().Or(rhs->refinements());
      if ((lhs.staticIf() && *lhs.staticIf()) ||
          (rhs->staticIf() && *rhs->staticIf())) {
        static_if = true;
      } else if (lhs.staticIf() && rhs->staticIf()) {
        static_if = *lhs.staticIf() || *rhs->staticIf();
      }
    } else {
      new_result = emitIfExpr(loc, lhs, get_continue_expr, get_const_expr);
      refinements = lhs.refinements().And(rhs->refinements());
      if (((lhs.staticIf() && !*lhs.staticIf()) ||
           (rhs->staticIf() && !*rhs->staticIf()))) {
        static_if = false;
      } else if (lhs.staticIf() && rhs->staticIf()) {
        static_if = *lhs.staticIf() && *rhs->staticIf();
      }
    }
    return CondValue(new_result, std::move(*refinements), static_if);
  }

  Value* emitIfExpr(
      const SourceRange& range,
      const CondValue& cond_value,
      const std::function<Value*()>& true_expr,
      const std::function<Value*()>& false_expr) {
    Node* n = graph->insertNode(create(prim::If, range, 0));
    n->addInput(cond_value.value());
    auto* true_block = n->addBlock();
    auto* false_block = n->addBlock();

    auto emit_if_expr = [this, &range](
                            Block* b,
                            const RefinementSet& refinements,
                            const std::function<Value*()>& expr_value) {
      pushFrame(b);
      WithInsertPoint guard(b);
      insertRefinements(range, refinements);
      Value* out_val = expr_value();
      b->registerOutput(out_val);
      popFrame();
    };

    emit_if_expr(true_block, cond_value.refinements(), true_expr);
    emit_if_expr(false_block, cond_value.refinements().Not(), false_expr);

    auto true_type = true_block->outputs().at(0)->type();
    auto false_type = false_block->outputs().at(0)->type();
    auto unified = unifyTypes(true_type, false_type);
    if (!unified) {
      throw ErrorReport(range)
          << "if-expression's true branch has type " << true_type->repr_str()
          << " but false branch has type " << false_type->repr_str();
    }

    // Add op outputs
    auto expr_value = n->addOutput()->setType(*unified); // Resulting value

    return expr_value;
  }
  Value* emitToBool(const SourceRange& loc, Value* v) {
    Value* out;
    try {
      auto bool_cast = environment_stack->getSugaredVar("bool", loc);
      out = asSimple(bool_cast->call(loc, method, {v}, {}, 0));
    } catch (...) {
      throw ErrorReport(loc) << "Could not cast value of type "
                             << v->type()->repr_str() << " to bool";
    }
    // cast value not response for checking output type
    if (!out->type()->isSubtypeOf(BoolType::get())) {
      throw ErrorReport(loc)
          << "expected a bool expression for condition but found "
          << out->type()->repr_str();
    }
    return out;
  }

  void emitIfElseBlocks(
      const SourceRange& loc,
      const CondValue& cond_value,
      const List<Stmt>& trueBranch,
      const List<Stmt>& falseBranch) {
    // this is a static if statement: that is, it contains a subset
    // of operators where we are willing to specialize the if statement
    // to be only the true or false branch when the condition is statically
    // known. This is used to meta-program modules, for instance, when a
    // submodule is absent, an is None check can be used to ensure the
    // accesses to the None check, which would error, are not compiled.
    if (cond_value.staticIf()) {
      if (*cond_value.staticIf()) {
        insertRefinements(loc, cond_value.refinements());
        emitStatements(trueBranch);
      } else {
        insertRefinements(loc, cond_value.refinements().Not());
        emitStatements(falseBranch);
      }
      return;
    }

    Node* n = graph->insertNode(create(prim::If, loc, 0));
    n->addInput(cond_value.value());
    auto* true_block = n->addBlock();
    auto* false_block = n->addBlock();

    // Emit both blocks once to get the union of all mutated values
    auto save_true =
        emitSingleIfBranch(true_block, trueBranch, cond_value.refinements());
    auto save_false = emitSingleIfBranch(
        false_block, falseBranch, cond_value.refinements().Not());

    bool true_exits = exit_blocks.count(true_block);
    bool false_exits = exit_blocks.count(false_block);
    if (true_exits && false_exits) {
      exit_blocks.insert(n->owningBlock());
    }

    // In python, every variable assigned in an if statement escapes
    // the scope of the if statement (all variables are scoped to the function).
    // Script is a subset of python: we consider variables to be in scope
    // as long as there is a definition of the variable along all paths
    // through the if statement
    // ----
    // if ...:
    //   a =
    // else:
    //   ...
    // ... = a  # error, a is not defined along all paths
    // ----
    // if ...:
    //   a =
    // else:
    //   a =
    // ... = a # OK, a is defined along all paths
    // ----
    // a = ...
    // if ...:
    //   a =
    // ... = a # OK, a is defined along all paths
    // if ...:
    //   a =
    // else:
    //   return
    // ... = a # OK, a is always defined

    // ordered set, because we want deterministic graph output
    std::set<std::string> mutated_variables;

    // When we access either the true or false environment,
    // we need to set the insertion point so the prim::Load is inserted
    // into the right block.
    // if var is only defined in one branch save error in case it's used later
    for (auto& v : save_true->definedVariables()) {
      {
        WithInsertPoint insert(false_block);
        if (save_false->findInAnyFrame(v) || false_exits) {
          mutated_variables.insert(v);
        } else {
          ErrorReport error(loc);
          environment_stack->setVariableTypeError(v, [=]() -> std::string {
            error << v << " is not defined in the false branch";
            return error.what();
          });
        }
      }
    }
    for (auto& v : save_false->definedVariables()) {
      {
        WithInsertPoint insert(true_block);
        if (save_true->findInAnyFrame(v) || true_exits) {
          mutated_variables.insert(v);
        } else {
          ErrorReport error(loc);
          environment_stack->setVariableTypeError(v, [=]() -> std::string {
            error << v << " is not defined in the true branch";
            return error.what();
          });
        }
      }
    }

    // Register outputs in each block
    for (const auto& x : mutated_variables) {
      Value* tv;
      Value* fv;

      {
        WithInsertPoint insert(true_block);
        if (!true_exits) {
          tv = save_true->getVar(x, loc);
        }
      }
      {
        WithInsertPoint insert(false_block);
        if (!false_exits) {
          fv = save_false->getVar(x, loc);
        }
      }

      // if both branches exit don't emit any variables
      // if one branch exits then we allow the all variables in the other branch
      // to escape scope since they are well-defined
      if (true_exits && false_exits) {
        continue;
      } else if (true_exits) {
        tv = graph->createUninitialized(fv->type())
                 ->insertBefore(true_block->return_node())
                 ->output();
        graph->createStore(x, tv)->insertBefore(true_block->return_node());
      } else if (false_exits) {
        fv = graph->createUninitialized(tv->type())
                 ->insertBefore(false_block->return_node())
                 ->output();
        graph->createStore(x, fv)->insertBefore(false_block->return_node());
      }

      auto unified = unifyTypes(tv->type(), fv->type());

      // attempt to unify the types. we allow variables to be set to different
      // types in each branch as long as that variable is not already in scope,
      // or if that variable does not get used later. here, we save the error
      // so that the error message will be more informative in the case that is
      // used later. When a is accessed in (a + 1), the error will get printed
      // if cond:
      //    a = 1
      // else:
      //    a = tensor
      // b = a + 1
      //
      if (!unified) {
        ErrorReport error(loc);
        error << "Type mismatch: " << x << " is set to type "
              << tv->type()->repr_str() << " in the true branch"
              << " and type " << fv->type()->repr_str()
              << " in the false branch";
        if (save_true->findInParentFrame(x) ||
            save_false->findInParentFrame(x)) {
          throw error;
        } else {
          environment_stack->setVariableTypeError(
              x, [=]() -> std::string { return error.what(); });
          continue;
        }
      }
      environment_stack->setType(x, *unified);
    }
  }

  CondValue emitHasAttr(const Expr& objExpr, const Expr& attrExpr) {
    auto obj = emitSugaredExpr(objExpr, 1);
    if (attrExpr.kind() != TK_STRINGLITERAL) {
      throw ErrorReport(attrExpr)
          << "hasattr's second argument must be a string literal";
    }
    const std::string& name = StringLiteral(attrExpr).text();
    const bool hasAttr = obj->hasAttr(objExpr.range(), method, name);
    return CondValue(*graph, objExpr.range(), hasAttr, {});
  }

  CondValue emitIsInstance(const Expr& obj, const Expr& classinfo) {
    // turn (float, (int, tuple)) into a flat list of types and type kind
    // category checks: tuple_check = true, types = {float, int}
    struct GatheredTypes {
      GatheredTypes(ScriptTypeParser parser) : typeParser_(std::move(parser)) {}
      void gather(const Expr& classinfo) {
        if (classinfo.kind() == TK_TUPLE_LITERAL) {
          for (Expr e : TupleLiteral(classinfo).inputs()) {
            gather(e);
          }
          return;
        }
        TypePtr type = typeParser_.parseTypeFromExpr(classinfo);
        types.emplace_back(type);
      }
      bool staticallyTrue(const TypePtr& actual_type) {
        // is this isinstance check statically true?
        for (const TypePtr& typ : types) {
          if (actual_type->isSubtypeOf(typ)) {
            return true;
          }
        }
        return false;
      }
      bool maybeOfKind(TypeKind kind, const TypePtr& actual_type) {
        if (actual_type->kind() == AnyType::Kind) {
          return true;
        }
        if (auto op = actual_type->cast<OptionalType>()) {
          return op->getElementType()->kind() == kind;
        }
        return false;
      }
      bool staticallyFalse(const TypePtr& actual_type) {
        for (const TypePtr& typ : types) {
          if (typ->isSubtypeOf(actual_type)) {
            return false;
          }
          if ((typ->isSubtypeOf(AnyListType::get()) &&
               maybeOfKind(ListType::Kind, actual_type)) ||
              (typ->isSubtypeOf(AnyTupleType::get()) &&
               maybeOfKind(TupleType::Kind, actual_type))) {
            return false;
          }
        }
        return true;
      }
      ScriptTypeParser typeParser_;
      std::vector<TypePtr> types;
    };
    GatheredTypes gathered(typeParser_);
    gathered.gather(classinfo);
    auto val = emitExpr(obj);
    RefinementSet refinement;
    if (gathered.types.size() == 1 &&
        gathered.types.at(0)->isSubtypeOf(val->type()) &&
        obj.kind() == TK_VAR) {
      std::string ident = Var(obj).name().name();
      Refinement isinstance(std::move(ident), gathered.types.at(0));
      refinement = RefinementSet({isinstance}, {});
    }

    if (gathered.staticallyTrue(val->type())) {
      return CondValue(*graph, obj.range(), true, std::move(refinement));
    }
    if (gathered.staticallyFalse(val->type())) {
      return CondValue(*graph, obj.range(), false, std::move(refinement));
    }
    // check maybe true/false at runtime, need an actual op
    Value* result =
        graph->insertNode(graph->createIsInstance(val, gathered.types))
            ->output();
    return CondValue(result, std::move(refinement), c10::nullopt);
  }

  void emitIf(const If& stmt) {
    Expr cond = stmt.cond();
    CondValue cond_value = emitCondExpr(cond);
    emitIfElseBlocks(
        stmt.range(), cond_value, stmt.trueBranch(), stmt.falseBranch());
  }

  // *********************** Loop Operators ************************************
  // Emits a loop operator with the form:
  // Loop(max_trip_count)
  // block0(loop_counter) {
  //   <body>
  // }
  // block1 {
  //   <loop condition>
  //   -> (condition)
  // }
  // For loops will have an empty loop condition block with condition set to
  // true. In the convert to ssa pass, the loop condition will correctly
  // inlined. and inputs and outputs added so that the loop conforms to the
  // semantics specified at
  // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Loop
  void emitLoopCommon(
      const SourceRange& range,
      const std::function<void()>& emit_body,
      const SugaredValuePtr& iter_val,
      c10::optional<List<Expr>> targets,
      c10::optional<Expr> cond) {
    Value* max_trip_count_val = nullptr;
    if (iter_val != nullptr) {
      max_trip_count_val = iter_val->len(range, method);
    } else {
      max_trip_count_val = materializeConstant(
          std::numeric_limits<int64_t>::max(),
          *graph,
          range,
          integral_constants);
    }

    Node* n = graph->insertNode(create(prim::Loop, range, 0));
    auto* body_block = n->addBlock();
    {
      Block* condition_block = n->addBlock();
      pushFrame(condition_block);
      Value* out;
      if (cond) {
        WithInsertPoint insert(condition_block);
        out = emitToBool(cond.value().range(), emitExpr(cond.value()));
      } else {
        WithInsertPoint insert(n);
        out = graph->insertConstant(true, range);
      }
      condition_block->registerOutput(out);
      popFrame();
    }
    n->addInput(max_trip_count_val);

    WithLoopStatus loop_guard(&loop_status_, LoopStatus::IN_LOOP);
    Value* trip_count =
        body_block->addInput()->setType(IntType::get()); // Iteration num
    {
      pushFrame(body_block);
      WithInsertPoint guard(body_block);

      // if the FOR iters and targets are present, emit FOR target assignments
      if (iter_val != nullptr && targets) {
        Value* cur_elem = iter_val->getitem(range, method, trip_count)
                              ->asValue(range, method);
        SugaredValuePtr sv = std::make_shared<SimpleValue>(cur_elem);
        List<Expr> target_exprs = targets.value();
        validateAssignLhsExpr(target_exprs, range);

        // if target exprs are more than 1, it means iteration unpacking on LHS
        // we create Tuple literal to wrap those target exprs for assignments
        if (target_exprs.size() > 1) {
          Expr tl = TupleLiteral::create(range, target_exprs);
          target_exprs = List<Expr>::create(range, {tl});
        }
        emitExprsAssign(target_exprs, {sv}, range, /*n_binders=*/1);
      }
      emit_body();
      popFrame();
    }
  }

  void emitUnrolledLoop(
      const SourceRange& loc,
      const std::function<void()>& emit_body,
      const SugaredValuePtr& iterable,
      const List<Expr>& targets) {
    auto static_len = iterable->staticLen();
    TORCH_INTERNAL_ASSERT(
        static_len, "Unrolled loop iter should have static length");
    int64_t len = *static_len;
    WithLoopStatus loop_guard(&loop_status_, LoopStatus::IN_UNROLLED_LOOP);
    // In order to support ModuleLists which return different types,
    // as with an nn.Sequential which has a module that returns a Dict and then
    // a module which returns a Tensor,
    // we do not push a new environment frame because if we did all intermediary
    // values would have to subtype the input type.
    for (int64_t i = 0; i < len; ++i) {
      auto index =
          materializeConstant(i, *method.graph(), loc, integral_constants);
      auto sugared_value = iterable->getitem(loc, method, index);
      emitExprsAssign(
          targets, {sugared_value}, targets.range(), /*n_binders=*/1);
      emit_body();
    }
  }

  void emitFor(
      const List<Expr>& targets,
      const List<Expr>& itrs,
      const SourceRange& loc,
      const std::function<void()>& emit_body) {
    if (itrs.size() != 1) {
      throw ErrorReport(loc) << "List of iterables is not supported currently";
    }

    // Emit loop information for builtinFunction values like range(), zip(),
    // enumerate() or SimpleValue like List, Tensor, Dict, etc.
    SugaredValuePtr sv = emitSugaredExpr(itrs[0], 1);
    SugaredValuePtr iterable = sv->iter(loc, method);

    // We unroll the loop for iterables that contain ModuleLists so that we can
    // compile Heterogenous module lists.
    if (!iterable->shouldEmitUnrolled()) {
      emitLoopCommon(loc, emit_body, iterable, targets, {});
    } else {
      emitUnrolledLoop(loc, emit_body, iterable, targets);
    }
  }

  void emitFor(const For& stmt) {
    auto emit_body = [&]() { emitStatements(stmt.body()); };
    emitFor(stmt.targets(), stmt.itrs(), stmt.range(), emit_body);
  }

  void emitWhile(const While& stmt) {
    auto cond = stmt.cond();
    auto emit_body = [&]() { emitStatements(stmt.body()); };
    emitLoopCommon(stmt.range(), emit_body, nullptr, {}, cond);
  }

  void emitWith(const With& stmt) {
    auto targets = stmt.targets();
    // Keep a stack of entered objects so they can be exited
    // in the right order.
    std::stack<Value*> entered;

    for (const auto& target : targets) {
      Expr e = target.target();

      auto* rhs = emitExpr(e);
      auto* n = graph->insertNode(graph->create(prim::Enter, {rhs}));
      entered.push(rhs);
      auto rhsClass = rhs->type()->expect<ClassType>();

      if (!rhsClass) {
        throw ErrorReport(e.range())
            << "With item expression does not return a class type";
      }

      auto* enterMethod = rhsClass->findMethod("__enter__");
      auto* exitMethod = rhsClass->findMethod("__exit__");

      if (!enterMethod || !exitMethod) {
        throw ErrorReport(e.range())
            << "Object returned by with item expression does not define __enter__ and __exit__ methods";
      }

      // Check the schema of __enter__.
      auto& enterSchema = enterMethod->getSchema();
      if (enterSchema.arguments().size() != 1) {
        throw ErrorReport(e.range())
            << "__enter__ must have only one argument and one return value";
      }

      // Check the schema of __exit__.
      auto& exitSchema = exitMethod->getSchema();
      if (exitSchema.arguments().size() != 4) {
        throw ErrorReport(e.range())
            << "__exit__ must have four arguments and no return value";
      } else {
        if (exitSchema.returns().at(0).type() != NoneType::get()) {
          throw ErrorReport(e.range()) << "__exit__ must have no return value";
        }

        for (unsigned i = 1; i < 4; ++i) {
          if (exitSchema.arguments().at(i).type() != AnyType::get()) {
            throw ErrorReport(e.range())
                << "argument " << i
                << " of __exit__ must have Any type; TorchScript does not currently support passing exception type, value, or traceback to the __exit__ function.";
          }
        }
      }

      // Set the output of the enter node to be the return type of __enter__.
      n->output(0)->setType(enterSchema.returns().at(0).type());

      // Set i = e.__enter__() so that references to i in the body of the with
      // will resolve correctly.
      if (target.var().present()) {
        Var i = target.var().get();
        environment_stack->setVar(i.range(), i.name().name(), n->output(0));
      }
    }

    emitStatements(stmt.body());

    // Insert all the corresponding prim::Exit nodes.
    while (!entered.empty()) {
      auto* input = entered.top();
      entered.pop();
      auto* n = graph->create(prim::Exit);
      graph->insertNode(n);
      n->addInput(input);
    }
  }

  // Currently we do not support assigning exceptions to variables,
  // a = Exception("hi")
  // raise a
  //
  // We ignore the expression following raise
  void emitRaise(const Raise& raise) {
    auto sv = emitSugaredExpr(raise.expr(), 1);
    Value* error_message = nullptr;

    if (auto exception_instance =
            std::dynamic_pointer_cast<ExceptionMessageValue>(sv)) {
      // The typical case, an instance of the exception class was thrown:
      //    raise RuntimeError("error")
      error_message = exception_instance->getValue();
    } else if (
        auto exception_class = std::dynamic_pointer_cast<ExceptionValue>(sv)) {
      // A bare exception was thrown so add an empty message. e.g.
      //    raise RuntimeError
      error_message = insertConstant(*graph, "", raise.range());
    } else {
      // The raise was not followed by an exception (i.e. it was something like
      // `raise "error"` instead of `raise RuntimeError("error")`)
      throw ErrorReport(raise.range())
          << "exceptions must derive from BaseException";
    }

    if (!error_message->type()->isSubtypeOf(StringType::get())) {
      error_message = graph->insert(aten::str, {error_message});
    }

    graph->insert(prim::RaiseException, {error_message}, {}, raise.range());
    exit_blocks.insert(environment_stack->block());
  }

  // emit assserions as an if branch so that assertions will reuse the
  void emitAssert(const Assert& stmt) {
    CondValue cond_value = emitCondExpr(stmt.test());
    List<Stmt> true_branch = List<Stmt>::create(stmt.range(), {});
    // Create an `AssertionError("the_message")` call
    auto message = (stmt.msg().present())
        ? stmt.msg().get()
        : StringLiteral::create(stmt.range(), "");
    auto callee = Var::create(
        stmt.range(), Ident::create(stmt.range(), "AssertionError"));
    auto apply = Apply::create(
        stmt.range(),
        callee,
        List<Expr>::create(stmt.range(), {message}),
        List<Attribute>::create(stmt.range(), {}));

    List<Stmt> false_branch =
        List<Stmt>::create(stmt.range(), {Raise::create(stmt.range(), apply)});
    emitIfElseBlocks(stmt.range(), cond_value, true_branch, false_branch);
  }

  // Validate that the `lhs` Expr's in an assignment statement are valid. That
  // is:
  //
  // 1) All lhs Expr's are either Var, Tuple or Starred nodes
  // 2) There is at most one Starred node in the lhs Expr
  // 3) A Starred node can only appear when there is another non-Starred lhs
  //    Expr. Concretely this means that `*abc = func()` is illegal. Unpacking
  //    all outputs into a tuple is covered by `abc = func()`.
  bool validateAssignLhsExpr(const List<Expr>& lhs, const SourceRange& r) {
    size_t num_normal_assign = 0;
    size_t num_starred = 0;
    for (const auto& assignee : lhs) {
      if (assignee.kind() == TK_VAR || assignee.kind() == TK_SUBSCRIPT ||
          assignee.kind() == TK_TUPLE_LITERAL) {
        num_normal_assign++;
      } else if (assignee.kind() == TK_STARRED) {
        num_starred++;
      } else {
        throw ErrorReport(assignee) << "lhs of assignment must be a variable, "
                                    << "subscript, or starred expression";
      }
    }

    if (num_starred > 1) {
      throw ErrorReport(r)
          << "Only one starred expression is allowed on the lhs";
    }

    if (num_starred > 0 && num_normal_assign == 0) {
      throw ErrorReport(r) << "A Starred expression may only appear on the "
                           << "lhs within the presence of another non-starred"
                           << " expression";
    }

    return num_starred;
  }

  // Get the appropriate builtin op for this augmented assignment
  // If the RHS is a tensor, return the corresponding ATen in-place op
  // If it's a list of scalars, then return the corresponding list augment op
  Symbol getAugOp(const AugAssign& stmt, const TypePtr& type) {
    bool use_inplace_op = type->isSubtypeOf(TensorType::get()) ||
        type->kind() == TypeKind::ListType;
    switch (stmt.aug_op()) {
      case '+':
        return use_inplace_op ? aten::add_ : aten::add;
      case '-':
        return use_inplace_op ? aten::sub_ : aten::sub;
      case '/':
        return use_inplace_op ? aten::div_ : aten::div;
      case '*':
        return use_inplace_op ? aten::mul_ : aten::mul;
      case '%':
        return use_inplace_op ? aten::fmod_ : aten::fmod;
      default:
        throw ErrorReport(stmt)
            << "Unknown augmented assignment: " << kindToString(stmt.aug_op());
    }
  }

  // Get a pair of <in place magic method name, out of place magic method name>
  // since the out of place method is called if the in place method is not
  // present
  std::pair<std::string, std::string> getAugMagicMethod(const AugAssign& stmt) {
    switch (stmt.aug_op()) {
      case '+':
        return std::make_pair(std::string("__iadd__"), std::string("__add__"));
      case '-':
        return std::make_pair(std::string("__isub__"), std::string("__sub__"));
      case '/':
        return std::make_pair(
            std::string("__itruediv__"), std::string("__truediv__"));
      case '*':
        return std::make_pair(std::string("__imul__"), std::string("__mul__"));
      case '%':
        return std::make_pair(std::string("__imod__"), std::string("__mod__"));
      default:
        throw ErrorReport(stmt)
            << "Unknown augmented assignment: " << kindToString(stmt.aug_op());
    }
  }

  // Emit nodes for augmented assignments like `+=`
  void emitAugAssignment(const AugAssign& stmt) {
    switch (stmt.lhs().kind()) {
      case TK_VAR: {
        emitAugAssignmentToVar(stmt);
      } break;
      case '.': {
        emitAugAssignmentToSelectVar(stmt);
      } break;
      case TK_SUBSCRIPT: {
        emitAugAssignmentToSubscript(stmt);
      } break;
      default:
        throw ErrorReport(stmt.lhs())
            << "unexpected expression on "
            << "left-hand side of augmented assignment";
    }
  }

  // This will be called when there is a class param or module buffer
  // mutation which make the LHS of the expr be a select expression
  //
  // Example like:
  // class A(Module):
  //  def __init__():
  //    self.register_buffer("running_var", torch.zeros(1))
  //
  //  def forward():
  //    self.num_batches += 1
  void emitAugAssignmentToSelectVar(const AugAssign& stmt) {
    const auto lhs = Select(stmt.lhs());
    auto lhsSugaredVar = emitSugaredExpr(lhs.value(), 1);
    const auto lhsValue =
        lhsSugaredVar->attr(lhs.range(), method, lhs.selector().name())
            ->asValue(lhs.range(), method);
    auto result = emitAugAssignmentHelper(stmt, lhsValue);
    lhsSugaredVar->setAttr(stmt.range(), method, lhs.selector().name(), result);
  }

  void emitAugAssignmentToVar(const AugAssign& stmt) {
    const auto lhs = Var(stmt.lhs());
    auto lhsValue = emitExpr(lhs);
    auto result = emitAugAssignmentHelper(stmt, lhsValue);
    environment_stack->setVar(lhs.range(), lhs.name().name(), result);
  }

  Value* emitAugAssignmentHelper(const AugAssign& stmt, Value* lhs) {
    if (lhs->type()->kind() == TypeKind::ClassType) {
      // Call `__iadd__` so updates happen in place on class types
      // https://docs.python.org/3/reference/datamodel.html#object.__iadd__
      std::string in_place_method_name;
      std::string out_of_place_method_name;
      std::tie(in_place_method_name, out_of_place_method_name) =
          getAugMagicMethod(stmt);
      const auto rhs = emitExpr(stmt.rhs());

      // Determine whether to use __iadd__ or __add__ (use __add__ only if
      // __iadd__ is not present)
      auto type = lhs->type()->expect<ClassType>();
      std::string magic_method_name;
      if (type->findMethod(in_place_method_name)) {
        magic_method_name = in_place_method_name;
      } else if (type->findMethod(out_of_place_method_name)) {
        magic_method_name = out_of_place_method_name;
      } else {
        throw ErrorReport(stmt.range())
            << "Cannot emit inplace op on " << type->repr_str()
            << " since it does not define an " << in_place_method_name << " or "
            << out_of_place_method_name << " method";
      }

      // x += y is equivalent to x = x.__iadd__(y) or x = x.__add__(y) if
      // __iadd__ is not present
      return MethodValue(lhs, magic_method_name)
          .call(stmt.range(), method, {rhs}, {}, 0)
          ->asValue(stmt.range(), method);
    } else {
      const auto rhs = NamedValue(stmt.rhs().range(), emitExpr(stmt.rhs()))
                           .value(*method.graph());
      return emitBuiltinCall(
          stmt.range(),
          *method.graph(),
          getAugOp(stmt, lhs->type()),
          /*args=*/{lhs, rhs},
          /*kwargs=*/{},
          /*self=*/c10::nullopt);
    }
  }

  void emitAugAssignmentGeneric(
      const AugAssign& stmt,
      const Subscript& lhs,
      Value* sliceable) {
    // Get the idx to augment
    const auto subscriptExprs = lhs.subscript_exprs();
    const TypePtr type = sliceable->type();
    if (subscriptExprs.size() != 1) {
      throw ErrorReport(subscriptExprs)
          << "Sliced expression not yet supported for " << type->repr_str()
          << " augmented assignment. "
          << "File a bug if you want this";
    }

    TypePtr elemType = nullptr;
    if (const ListTypePtr listType = type->cast<ListType>()) {
      elemType = listType->getElementType();
    } else if (const DictTypePtr dictType = type->cast<DictType>()) {
      elemType = dictType->getKeyType();
    }

    if (elemType == nullptr) {
      throw ErrorReport(lhs)
          << type->repr_str() << " does not support augmented assignment.";
    }
    const auto idxValue = emitExpr(subscriptExprs[0]);
    const auto containerArg =
        NamedValue(lhs.value().range(), type->str(), sliceable);
    const auto idxArg = NamedValue(subscriptExprs.range(), "idx", idxValue);
    const auto valueArg =
        NamedValue(stmt.rhs().range(), "value", emitExpr(stmt.rhs()));

    const auto getItem = graph->insert(
        aten::__getitem__, {containerArg, idxArg}, {}, stmt.range());
    const auto augmentedItem = graph->insert(
        getAugOp(stmt, elemType), {getItem, valueArg}, {}, stmt.range());
    graph->insert(
        aten::_set_item,
        {containerArg, idxArg, augmentedItem},
        {},
        stmt.range());
  }

  void emitAugAssignmentToSubscript(const AugAssign& stmt) {
    // Process the base list value
    const auto lhs = Subscript(stmt.lhs());
    const auto sliceable = emitExpr(lhs.value());

    if (sliceable->type()->isSubtypeOf(TensorType::get())) {
      // If it's a tensor, just fully evaluate the subscript operation and emit
      // an in-place assignment
      std::vector<Value*> tensorIndices;
      Value* sliced;
      std::tie(sliced, tensorIndices) = emitIntAndSliceIndexing(
          lhs.range(), sliceable, lhs.subscript_exprs());

      const auto slicedArg = NamedValue(stmt.lhs().range(), "self", sliced);
      const auto rhs = NamedValue(stmt.rhs().range(), emitExpr(stmt.rhs()));
      if (tensorIndices.size() == 0) {
        // Common case: we only tried to index with int and slices. Emit the
        // correct augmented assignment op to the sliced value
        emitBuiltinCall(
            stmt.range(),
            *method.graph(),
            getAugOp(stmt, sliceable->type()),
            {rhs},
            {},
            slicedArg);
      } else {
        // Special case: we tried to do "advanced indexing". Lower this expr
        // into `index` and `index_put_` ops with tensordices of Tensor?[]
        const auto indices = graph
                                 ->insertNode(graph->createList(
                                     OptionalType::ofTensor(), tensorIndices))
                                 ->output();
        const auto indexed =
            graph->insert(aten::index, {slicedArg, indices}, {}, stmt.range());
        const auto augmented = emitBuiltinCall(
            stmt.range(),
            *method.graph(),
            getAugOp(stmt, sliceable->type()),
            {rhs},
            {},
            indexed);
        graph->insert(
            aten::index_put_,
            {slicedArg, indices, augmented},
            {},
            stmt.range());
      }
    } else {
      emitAugAssignmentGeneric(stmt, lhs, sliceable);
    }
  }

  NamedValue emitValueToTensor(
      const NamedValue& value,
      const NamedValue& matchTypeOf) {
    // Add implicit conversion of int/float/bool/number types to tensors
    // Used in emitSubscriptAssign to convert:
    //   `tensor(...)[x] = 99` to `tensor(...)[x] = tensor(99)`
    // Mirrors the `valueToTensor` behavior in python_variable_indexing.cpp
    const auto kind = value.type()->kind();
    if (kind == c10::TypeKind::NumberType || kind == c10::TypeKind::IntType ||
        kind == c10::TypeKind::BoolType || kind == c10::TypeKind::FloatType) {
      auto dtype = graph->insert(prim::dtype, {matchTypeOf}, {});
      auto device = graph->insert(prim::device, {matchTypeOf}, {});
      auto converted = graph->insert(
          aten::tensor,
          {value},
          {NamedValue("dtype", dtype), NamedValue("device", device)});
      return NamedValue(value.loc(), converted);
    }

    return value;
  }

  // Emit mutating assignments like `foo[0] = bar`
  void emitSubscriptAssign(
      const SourceRange& stmtRange,
      const Subscript& lhs,
      const Expr& rhs) {
    emitSubscriptAssign(stmtRange, lhs, NamedValue(rhs.range(), emitExpr(rhs)));
  }

  void emitSubscriptAssign(
      const SourceRange& stmtRange,
      const Subscript& lhs,
      const NamedValue& rhs) {
    // First check the base value.
    auto sliceable = emitExpr(lhs.value());

    // If it's a tensor, copy the RHS data into it
    if (sliceable->type()->isSubtypeOf(TensorType::get())) {
      std::vector<Value*> tensorIndices;
      Value* sliced;
      // Handle multi-dimensional slicing: first emit int/slice indexing
      // TODO: the Python equivalent code has special-cased copy_to
      // broadcasting to match NumPy semantics (see PR#4853). We can't
      // replicate that without knowing the size of the Tensor; so really that
      // code should be moved into the aten function
      std::tie(sliced, tensorIndices) = emitIntAndSliceIndexing(
          lhs.range(), sliceable, lhs.subscript_exprs());

      const auto slicedArg = NamedValue(lhs.range(), sliced);

      // rhs must be a tensor, implicitly convert int/float/bool
      const auto convertedRhs = emitValueToTensor(rhs, slicedArg);

      if (tensorIndices.size() == 0) {
        // Common case: we only tried to index with int and slices. Copy the
        // RHS into the resulting tensor.
        graph->insert(aten::copy_, {slicedArg, convertedRhs}, {}, stmtRange);
      } else {
        // Special case: we tried to do "advanced indexing" with a tensor.
        // Dispatch to `aten::index_put_` with tensorindices of Tensor?[]
        const auto indices = graph
                                 ->insertNode(graph->createList(
                                     OptionalType::ofTensor(), tensorIndices))
                                 ->output();

        graph->insert(
            aten::index_put_,
            {slicedArg, indices, convertedRhs},
            {},
            stmtRange);
      }
      // Otherwise, this is a list or a classtype.
      // Dispatch to aten::_set_item to both select and assign
    } else {
      const auto subscript = lhs.subscript_exprs();
      if (subscript.size() != 1 || subscript[0].kind() == TK_SLICE_EXPR) {
        throw ErrorReport(subscript)
            << "Sliced expression not yet supported for"
            << " subscripted assignment. "
            << "File a bug if you want this";
      }
      if (sliceable->type()->isSubtypeOf(AnyTupleType::get())) {
        throw ErrorReport(lhs) << sliceable->type()->repr_str()
                               << " does not support subscripted assignment";
      }

      std::vector<NamedValue> args;
      args.emplace_back(lhs.value().range(), "self", sliceable);
      args.emplace_back(
          lhs.subscript_exprs().range(), "idx", emitExpr(subscript[0]));
      args.push_back(rhs);
      makeMagic(
          "__setitem__",
          std::make_shared<BuiltinFunction>(aten::_set_item, at::nullopt))
          ->call(stmtRange, method, args, {}, 0);
    }
  }

  void emitTupleAssign(const TupleLiteral& tl, const Expr& rhs) {
    size_t n_binders = tl.inputs().size();
    bool starred_unpack = validateAssignLhsExpr(tl.inputs(), tl.range());
    if (starred_unpack)
      n_binders--;
    auto output = emitSugaredExpr(rhs, n_binders);
    emitTupleAssign(tl, output, rhs.range(), n_binders, starred_unpack);
  }

  void emitTupleAssign(
      const TupleLiteral& tl,
      const SugaredValuePtr& rhs_output,
      const SourceRange& rhs_loc,
      size_t n_binders,
      bool starred_unpack) {
    auto outputs = rhs_output->asTuple(
        rhs_loc,
        method,
        starred_unpack ? c10::nullopt : c10::optional<size_t>{n_binders});
    if (outputs.size() < n_binders) {
      throw ErrorReport(tl)
          << "need " << (starred_unpack ? "at least " : "") << n_binders
          << " values to unpack but found only " << outputs.size();
    }
    if (outputs.size() > n_binders && !starred_unpack) {
      throw ErrorReport(tl) << "too many values to unpack: need " << n_binders
                            << " but found " << outputs.size();
    }

    emitExprsAssign(tl.inputs(), outputs, rhs_loc, n_binders);
  }

  void emitExprsAssign(
      const List<Expr>& lhs_exprs,
      const at::ArrayRef<SugaredValuePtr> outputs,
      const SourceRange& rhs_loc,
      size_t n_binders) {
    int i = 0;
    for (auto assignee : lhs_exprs) {
      switch (assignee.kind()) {
        case TK_SUBSCRIPT:
          emitSubscriptAssign(
              rhs_loc,
              Subscript(assignee),
              NamedValue(rhs_loc, outputs.at(i)->asValue(rhs_loc, method)));
          i++;
          break;
        case TK_VAR:
          environment_stack->setSugaredVar(
              assignee.range(),
              Var(assignee).name().name(),
              outputs.at(i),
              /*annotated_type=*/nullptr);
          i++;
          break;
        case TK_STARRED: {
          auto var = Starred(assignee).expr();
          if (var.kind() != TK_VAR) {
            throw ErrorReport(var) << "Cannot pack a tuple into a non-variable";
          }
          size_t n_matched = outputs.size() - n_binders;
          ArrayRef<std::shared_ptr<SugaredValue>> outputs_ref = outputs;
          auto values = fmap(
              outputs_ref.slice(i, n_matched),
              [&](const std::shared_ptr<SugaredValue>& v) {
                return v->asValue(assignee.range(), method);
              });
          auto tup = graph->insertNode(graph->createTuple(values))->output();
          environment_stack->setVar(var.range(), Var(var).name().name(), tup);
          i += n_matched;
        } break;
        case TK_TUPLE_LITERAL: {
          // recursively emit tuple assignments on tuple literal input
          TupleLiteral sub_tl = TupleLiteral(assignee);
          size_t sub_n_binders = sub_tl.inputs().size();
          bool sub_starred_unpack =
              validateAssignLhsExpr(sub_tl.inputs(), sub_tl.range());
          if (sub_starred_unpack)
            sub_n_binders--;
          emitTupleAssign(
              sub_tl,
              outputs.at(i),
              rhs_loc,
              sub_n_binders,
              sub_starred_unpack);
          i++;
        } break;
        default:
          throw ErrorReport(assignee)
              << "unexpected expression on the left-hand side";
      }
    }
  }

  void emitAssignment(const Assign& stmt) {
    if (stmt.lhs_list().size() == 1) {
      return emitSingleAssignment(stmt);
    }
    // multiple assign & annotated type not supported in python
    TORCH_INTERNAL_ASSERT(stmt.lhs_list().size() > 1 && !stmt.type().present());
    // a = b = expr()
    // the semantics of multiple assignment is that expr() is emitted once, then
    // from left to right the assignments are made
    const auto tmp_name = createTempName("$tmp_assign_");
    environment_stack->setSugaredVar(
        stmt.rhs().range(),
        tmp_name,
        emitSugaredExpr(stmt.rhs().get(), 1),
        /*annotated_type=*/nullptr);
    auto ident = Var::create(
        stmt.rhs().range(), Ident::create(stmt.rhs().range(), tmp_name));
    for (auto expr : stmt.lhs_list()) {
      emitSingleAssignment(Assign::create(
          stmt.range(),
          List<Expr>::create(expr.range(), {expr}),
          Maybe<Expr>::create(stmt.rhs().range(), ident),
          Maybe<Expr>::create(stmt.range())));
    }
  }

  void emitSingleAssignment(const Assign& stmt) {
    if (!stmt.rhs().present()) {
      throw ErrorReport(stmt.range())
          << "For an assignment, expected an expression on the right-hand side";
    }
    const Expr& rhs = stmt.rhs().get();
    switch (stmt.lhs().kind()) {
      case TK_VAR: {
        auto v = Var(stmt.lhs());
        TypePtr type = nullptr;
        if (stmt.type().present()) {
          type = typeParser_.parseTypeFromExpr(stmt.type().get());
        }
        auto rhs_sugared_val = emitSugaredExpr(rhs, 1, type);
        // START BC HACK
        //
        // For old serialized quantized RNN modules, switch
        // quantized::linear_prepack to quantized::linear_prepack_legacy. We
        // changed linear_prepack to return a TorchBind class and not a
        // cpp_custom_type_hack tensor anymore, but the old serialized models
        // are tightly coupled with the type_hack version. If we still create a
        // Tensor here, then the quantized_lstm.legacy overload can kick in in
        // forward_impl(), and the module will still run correctly.
        if (method.qualname() ==
            "__torch__.torch.nn.quantized.dynamic.modules.rnn.PackedParameter.__setstate__") {
          if (auto sv =
                  std::dynamic_pointer_cast<SimpleValue>(rhs_sugared_val)) {
            Node* rhs_node = sv->getValue()->node();
            if (rhs_node->kind() ==
                Symbol::fromQualString("quantized::linear_prepack")) {
              std::vector<NamedValue> inputs;
              for (Value* i : rhs_node->inputs()) {
                inputs.emplace_back(i);
              }
              Value* new_val = rhs_node->owningGraph()->insert(
                  Symbol::fromQualString("quantized::linear_prepack_legacy"),
                  inputs,
                  {},
                  rhs_node->sourceRange());
              rhs_sugared_val = std::make_shared<SimpleValue>(new_val);
            }
          }
        }
        // END BC HACK
        environment_stack->setSugaredVar(
            v.range(),
            v.name().name(),
            std::move(rhs_sugared_val),
            /*annotated_type=*/type);
      } break;
      case TK_TUPLE_LITERAL:
        emitTupleAssign(TupleLiteral(stmt.lhs()), rhs);
        break;
      case '.':
        emitSelectAssign(stmt);
        break;
      case TK_SUBSCRIPT:
        emitSubscriptAssign(stmt.range(), Subscript(stmt.lhs()), rhs);
        break;
      default:
        throw ErrorReport(stmt.lhs())
            << "unexpected expression on left-hand side of assignment";
    }
  }

  void emitSelectAssign(const Assign& stmt) {
    if (!stmt.rhs().present()) {
      throw ErrorReport(stmt.range()) << "Expected RHS for assignment";
    }

    TypePtr type_hint = nullptr;
    if (stmt.type().present()) {
      type_hint = typeParser_.parseTypeFromExpr(stmt.type().get());
    }
    const auto lhs = Select(stmt.lhs());
    auto lhsObject = emitSugaredExpr(lhs.value(), 1);
    const auto rhsValue = emitSugaredExpr(stmt.rhs().get(), 1, type_hint)
                              ->asValue(stmt.rhs().range(), method);
    lhsObject->setAttr(stmt.range(), method, lhs.selector().name(), rhsValue);
  }

  NodeKind getNodeKind(int kind, int ninputs) {
    switch (kind) {
      case '+':
        return aten::add;
      case '-':
        return aten::sub;
      case TK_UNARY_MINUS:
        return aten::neg;
      case '*':
        return aten::mul;
      case TK_POW:
        return aten::pow;
      case '@':
        return aten::matmul;
      case TK_STARRED:
        return prim::Starred;
      case '/':
        return aten::div;
      case '%':
        return aten::remainder;
      case TK_NE:
        return aten::ne;
      case TK_EQ:
        return aten::eq;
      case '<':
        return aten::lt;
      case '>':
        return aten::gt;
      case TK_LE:
        return aten::le;
      case TK_GE:
        return aten::ge;
      case TK_AND:
        return aten::__and__;
      case TK_OR:
        return aten::__or__;
      case TK_IS:
        return aten::__is__;
      case TK_ISNOT:
        return aten::__isnot__;
      case TK_NOT:
        return aten::__not__;
      case TK_FLOOR_DIV:
        return aten::floordiv;
      case TK_LSHIFT:
        return aten::__lshift__;
      case TK_RSHIFT:
        return aten::__rshift__;
      case '&':
        return aten::__and__;
      case '|':
        return aten::__or__;
      case '^':
        return aten::__xor__;
      case TK_IN:
        return aten::__contains__;
      default:
        throw std::runtime_error("unknown kind " + c10::to_string(kind));
    }
  }

  std::string getOperatorOverload(int kind, int ninputs) {
    switch (kind) {
      case '+':
        return "__add__";
      case '-':
        return "__sub__";
      case TK_UNARY_MINUS:
        return "__neg__";
      case '~':
        return "__invert__";
      case '*':
        return "__mul__";
      case TK_POW:
        return "__pow__";
      case '/':
        return "__truediv__";
      case '%':
        return "__mod__";
      case TK_NE:
        return "__ne__";
      case TK_EQ:
        return "__eq__";
      case '<':
        return "__lt__";
      case '>':
        return "__gt__";
      case TK_LE:
        return "__le__";
      case TK_GE:
        return "__ge__";
      case '&':
        return "__and__";
      case '|':
        return "__or__";
      case '^':
        return "__xor__";
      case TK_IN:
        return "__contains__";
      case TK_LSHIFT:
        return "__lshift__";
      case TK_RSHIFT:
        return "__rshift__";
      default:
        throw std::runtime_error("unknown kind " + c10::to_string(kind));
    }
  }

  std::vector<NamedValue> getNamedValues(
      const TreeList& trees,
      bool maybe_unpack) {
    std::vector<NamedValue> values;
    for (const auto& tree : trees) {
      if (maybe_unpack && tree->kind() == TK_STARRED) {
        auto starred = Starred(tree);
        auto entries = emitSugaredExpr(starred.expr(), 1)
                           ->asTuple(starred.range(), method);
        for (const auto& entry : entries) {
          values.emplace_back(
              tree->range(), entry->asValue(starred.range(), method));
        }
      } else {
        values.emplace_back(tree->range(), emitExpr(Expr(tree)));
      }
    }
    return values;
  }
  std::vector<NamedValue> getNamedValues(
      const List<Expr>& trees,
      bool maybe_unpack) {
    return getNamedValues(trees.tree()->trees(), maybe_unpack);
  }

  std::vector<Value*> getValues(const TreeList& trees, bool maybe_unpack) {
    return toValues(*graph, getNamedValues(trees, maybe_unpack));
  }
  std::vector<Value*> getValues(const List<Expr>& trees, bool maybe_unpack) {
    return getValues(trees.tree()->trees(), maybe_unpack);
  }

  std::vector<NamedValue> emitAttributes(const List<Attribute>& attributes) {
    return fmap(attributes, [&](const Attribute& attr) {
      return NamedValue(
          attr.range(), attr.name().name(), emitExpr(attr.value()));
    });
  }

  void checkApplyNumInputs(Apply& apply, size_t expected_inputs) {
    const SourceRange& loc = apply.range();
    if (apply.inputs().size() != expected_inputs) {
      throw ErrorReport(loc)
          << Var(apply.callee()).name().name() << " expected exactly "
          << expected_inputs << " arguments but found "
          << apply.inputs().size();
    }
    if (apply.attributes().size() > 0) {
      throw ErrorReport(loc)
          << Var(apply.callee()).name().name() << " takes no keyword arguments";
    }
  }

  std::shared_ptr<SugaredValue> emitApplyExpr(
      Apply& apply,
      size_t n_binders,
      const TypePtr& type_hint = nullptr) {
    auto sv = emitSugaredExpr(apply.callee(), 1);
    auto loc = apply.callee().range();
    if (auto special_form = dynamic_cast<SpecialFormValue*>(sv.get())) {
      return emitApplySpecialForm(special_form->form(), apply, type_hint);
    }
    auto args = getNamedValues(apply.inputs(), true);
    auto kwargs = emitAttributes(apply.attributes());
    return sv->call(loc, method, args, kwargs, n_binders);
  }

  // this function handles expressions that look like apply statements
  // but have special evaluation rules for the arguments.
  // when adding a new case, only add a special form if it cannot be expressed
  // using the standard SugaredValue::call function, which enforces normal
  // evaluation order.
  std::shared_ptr<SugaredValue> emitApplySpecialForm(
      Symbol form,
      Apply& apply,
      const TypePtr& type_hint = nullptr) {
    switch (form) {
      case prim::fork: {
        auto& trees = apply.inputs().tree()->trees();
        if (trees.size() < 1) {
          throw ErrorReport(apply)
              << "Expected at least one argument to fork()";
        }
        auto forked = emitSugaredExpr(Expr(trees[0]), 1);
        TreeList sliced_trees(trees.begin() + 1, trees.end());
        auto args = getNamedValues(sliced_trees, true);
        auto kwargs = emitAttributes(apply.attributes());
        return emitForkExpr(apply.range(), forked, args, kwargs);
      }
      case prim::annotate: {
        checkApplyNumInputs(apply, 2);
        TypePtr type = typeParser_.parseTypeFromExpr(apply.inputs()[0]);
        Value* expr = tryConvertToType(
            apply.range(),
            *graph,
            type,
            emitExpr(apply.inputs()[1], type),
            /*allow_conversions=*/true);

        std::stringstream why_not;
        if (!expr->type()->isSubtypeOfExt(type, &why_not)) {
          throw ErrorReport(apply.inputs())
              << "expected an expression of type " << type->repr_str()
              << " but found " << expr->type()->repr_str() << "\n"
              << why_not.str();
        }

        // None is a subtype of Optional[T], but we want to remember what T is
        // after annotation so that variables assigned to this None will still
        // get the right type. To do this, we make a None constant that
        // has the type Optional[T]
        if (type->kind() == OptionalType::Kind &&
            expr->type()->isSubtypeOf(NoneType::get())) {
          Node* none = graph->createNone();
          none->output()->setType(type);
          graph->insertNode(none);
          expr = none->output();
        }

        return std::make_shared<SimpleValue>(expr);
      }
      case prim::rpc_async:
      case prim::rpc_sync:
      case prim::rpc_remote: {
        return emitRpcExpr(apply, form);
      }
      case prim::unchecked_cast: {
        checkApplyNumInputs(apply, 2);
        TypePtr type = typeParser_.parseTypeFromExpr(apply.inputs()[0]);
        Value* v = emitExpr(apply.inputs()[1]);
        // avoid generating nested unchecked_casts because they are already
        // inserted during serialization
        if (v->node()->kind() != prim::unchecked_cast || *v->type() != *type) {
          v = graph->insertUncheckedCast(v, type);
        }
        return std::make_shared<SimpleValue>(v);
      } break;
      case prim::GetAttr: {
        checkApplyNumInputs(apply, 2);
        auto obj = emitSugaredExpr(apply.inputs()[0], 1);
        auto selector = apply.inputs()[1];
        if (selector.kind() != TK_STRINGLITERAL) {
          throw ErrorReport(apply)
              << "getattr's second argument must be a string literal";
        }
        const std::string& name = StringLiteral(selector).text();
        return obj->attr(apply.range(), method, name);
      }
      case prim::Uninitialized: {
        checkApplyNumInputs(apply, 1);
        TypePtr type = typeParser_.parseTypeFromExpr(apply.inputs()[0]);
        auto out = graph->insertNode(graph->createUninitialized(type))
                       ->setSourceRange(apply.range());
        return std::make_shared<SimpleValue>(out->output());
      }
      case prim::TupleConstruct: {
        checkApplyNumInputs(apply, 1);
        auto arg = emitSugaredExpr(apply.inputs()[0], 1);
        auto inputs = arg->asTuple(apply.range(), method);
        auto inp_values = fmap(inputs, [&](const SugaredValuePtr& sv) {
          return sv->asValue(apply.range(), method);
        });
        return std::make_shared<SimpleValue>(
            graph->insertNode(graph->createTuple(inp_values))->output());
      }
      case prim::isinstance: {
        checkApplyNumInputs(apply, 2);
        auto result = emitIsInstance(apply.inputs()[0], apply.inputs()[1]);
        return std::make_shared<SimpleValue>(result.value());
      }
      case prim::tolist: {
        auto select = Select(apply.callee());
        auto value = select.value();
        auto operand = emitSugaredExpr(value, 1);

        if (!type_hint) {
          throw ErrorReport(apply)
              << "Expected type hint for result of tolist()";
        }

        return std::make_shared<SimpleValue>(graph->insertToList(
            operand->asValue(value.range(), method), type_hint));
      }
      case prim::HasAttr: {
        checkApplyNumInputs(apply, 2);
        const auto result = emitHasAttr(apply.inputs()[0], apply.inputs()[1]);
        return std::make_shared<SimpleValue>(result.value());
      } break;
      // This represents the "__new__" method on classes
      // because it takes a ClassValue as input.
      // So if we see:
      //   Foo.__new__(Foo)
      // Foo is a ClassValue, calling `attr("__new__")` will return a
      // CreateObject special form.
      case prim::CreateObject: {
        if (apply.inputs().size() != 1) {
          throw ErrorReport(apply) << "Only one argument to __new__ allowed";
        }
        auto arg = emitSugaredExpr(apply.inputs()[0], 1);
        auto class_arg = dynamic_cast<ClassValue*>(arg.get());
        if (!class_arg) {
          throw ErrorReport(apply)
              << "Expected class value as argument to __new__, got "
              << arg->kind() << " instead";
        }
        auto createNode =
            graph->insertNode(graph->createObject(class_arg->type_));
        return std::make_shared<SimpleValue>(createNode->output());
      }
      // We construct the iterable tree here using the IterableTree
      // SugaredValue, The tree consists of SimpleValue, RangeValue or
      // IterableTree: For SimpleValues(List, Dict, etc) or RangeValue. We will
      // make them as tree leaves since we could get the loop information from
      // len() and get_item(). For IterableTree like zip(), enumerate(), we can
      // model them as a combination of leaves, and we emit a IterableTree value
      // to record the tree information
      case prim::range: {
        std::vector<Value*> input_vals =
            getValues(apply.inputs(), /*maybe_unpack=*/true);
        return std::make_shared<RangeValue>(apply.range(), method, input_vals);
      }
      case prim::enumerate: {
        const SourceRange& loc = apply.range();
        auto inputs = apply.inputs();
        auto input_size = apply.inputs().size();
        // enumerate(x) can be rewrite as subtrees:
        // IterableTree(RangeValue(0, math.inf), SimpleValue(x))
        Value* start_index = nullptr;
        if (input_size == 0) {
          throw ErrorReport(loc)
              << "enumerate expected at least 1 arguments, got 0";
        }

        if (input_size == 2) {
          start_index = emitSugaredExpr(inputs[1], 1)->asValue(loc, method);
        }

        if (input_size > 2) {
          throw ErrorReport(loc)
              << "enumerate expected at most 2 arguments, got " << input_size;
        }
        std::vector<Value*> range_inputs;
        if (start_index != nullptr) {
          range_inputs.emplace_back(start_index);
        }
        Value* end = materializeConstant(
            std::numeric_limits<int64_t>::max(),
            *graph,
            loc,
            integral_constants);
        range_inputs.emplace_back(end);
        SugaredValuePtr expr_sv = emitSugaredExpr(inputs[0], 1);
        auto iterable_value = expr_sv->iter(loc, method);

        // range should have the same static length as the other iterable
        c10::optional<int64_t> iter_static_len = iterable_value->staticLen();
        SugaredValuePtr range_sv = std::make_shared<RangeValue>(
            loc, method, range_inputs, iter_static_len);

        auto tree = std::make_shared<IterableTree>();
        tree->addChild(loc, method, range_sv);
        tree->addChild(loc, method, iterable_value);
        return tree;
      }
      case prim::zip: {
        // zip(x, y) can be rewrite as subtrees:
        // IterableTree(IterableTree(x), IterableTree(y))
        auto inputs = apply.inputs();
        if (inputs.size() == 0) {
          throw ErrorReport(apply)
              << "zip expected at least 1 arguments, got 0";
        }
        auto iterable_tree = std::make_shared<IterableTree>();
        for (Expr expr : inputs) {
          auto iterable = emitSugaredExpr(expr, 1)->iter(apply.range(), method);
          iterable_tree->addChild(apply.range(), method, iterable);
        }
        return iterable_tree;
      }
      case prim::list: {
        if (apply.inputs().size() == 0) {
          TypePtr type = type_hint ? type_hint : ListType::ofTensors();
          if (!type->cast<ListType>()) {
            throw ErrorReport(apply.range())
                << "Expected list type annotation for list(), found "
                << type_hint->repr_str();
          }
          return std::make_shared<SimpleValue>(
              graph
                  ->insertNode(graph->createList(
                      type->expect<ListType>()->getElementType(), {}))
                  ->output());
        }
        // list(iter) desugars to [_elem for _elem in iter]
        checkApplyNumInputs(apply, 1);
        auto iter_input = emitSugaredExpr(apply.inputs()[0], 1);

        // aten::list builtin op is registered for List and Str input
        // dispatch to the builtin op to avoid perf slowdown on existing uses
        if (auto simple = asSimple(iter_input)) {
          if (simple->type()->cast<ListType>() ||
              simple->type()->cast<StringType>()) {
            return std::make_shared<SimpleValue>(emitBuiltinCall(
                apply.range(), *method.graph(), aten::list, {simple}, {}));
          }
        }
        const std::string& iter_name = createTempName("$_iter");
        environment_stack->setSugaredVar(
            apply.range(),
            iter_name,
            iter_input,
            /*annotated_type=*/nullptr);

        const std::string& elem_name = createTempName("$_elem");
        auto ident =
            Var::create(apply.range(), Ident::create(apply.range(), elem_name));
        auto iter =
            Var::create(apply.range(), Ident::create(apply.range(), iter_name));
        auto lc = ListComp::create(apply.range(), ident, ident, iter);
        return std::make_shared<SimpleValue>(
            emitListComprehension(lc, type_hint));
      }
      default:
        TORCH_INTERNAL_ASSERT(false, "unknown special form: ", form);
    }
  }

  Value* emitExpr(const Expr& tree, const TypePtr& type_hint = nullptr) {
    // Push the source range of a call in case compiling this function
    // triggers an error
    ErrorReport::CallStack::update_pending_range(tree.range());
    return emitSugaredExpr(tree, 1, type_hint)->asValue(tree.range(), method);
  }

  NodeKind reverseComparision(NodeKind kind) {
    if (kind == aten::lt) {
      return aten::gt;
    } else if (kind == aten::le) {
      return aten::ge;
    } else if (kind == aten::gt) {
      return aten::lt;
    } else if (kind == aten::ge) {
      return aten::le;
    }
    throw std::runtime_error(
        "reverseComparision: unsupported NodeKind. File a bug");
  }

  // any expression that can produce a SugaredValue is handled here
  // expressions that only return a single Value* are handled in emitSimpleExpr
  // type_hint is set if there is a type that this value is expected to be
  // e.g. a : List[int] = []
  // or a = torch.jit.annotate(List[int], [])
  // the caller is responsible for checking that the result matches type_hint
  // emitSugaredExpr is free to ignore it.
  std::shared_ptr<SugaredValue> emitSugaredExpr(
      const Expr& tree,
      size_t n_binders,
      const TypePtr& type_hint = nullptr) {
    switch (tree.kind()) {
      case TK_VAR:
        return environment_stack->getSugaredVar(Var(tree).name());
      case '.': {
        auto select = Select(tree);
        auto sv = emitSugaredExpr(select.value(), 1);
        return sv->attr(select.range(), method, select.selector().name());
      }
      case TK_APPLY: {
        auto apply = Apply(tree);
        return emitApplyExpr(apply, n_binders, type_hint);
      } break;
      case TK_SUBSCRIPT: {
        return emitSubscript(Subscript(tree), type_hint);
      } break;
      default:
        return std::make_shared<SimpleValue>(emitSimpleExpr(tree, type_hint));
    }
  }

  Value* emitUnaryOp(
      const TreeRef& tree,
      const std::string& magicMethod,
      const c10::Symbol& opSymbol) {
    const auto& inputs = tree->trees();
    auto named_values = getNamedValues(inputs, /*maybe_unpack=*/false);
    auto val =
        asSimple(makeMagic(
                     magicMethod,
                     std::make_shared<BuiltinFunction>(opSymbol, at::nullopt))
                     ->call(tree->range(), method, named_values, {}, 0));

    // if we emitted the unary op and not some other overloaded function,
    // then try to constantfold
    if (val->node()->kind() != opSymbol) {
      return val;
    }

    auto maybe_out_stack = runNodeIfInputsAreConstant(val->node());
    if (!maybe_out_stack) {
      return val;
    }
    TORCH_INTERNAL_ASSERT(maybe_out_stack->size() == 1);
    return graph->insertConstant(maybe_out_stack->at(0), tree->range());
  }

  /**
   * Emit a fork expression, of the form:
   *   torch.jit.fork(forked, *args, **kwargs)
   */
  std::shared_ptr<SugaredValue> emitForkExpr(
      SourceRange loc,
      const std::shared_ptr<SugaredValue>& forked,
      at::ArrayRef<NamedValue> args,
      at::ArrayRef<NamedValue> kwargs) {
    auto g = method.graph();
    Node* fork_node;
    TypePtr out_type;

    fork_node = g->insertNode(method.graph()->create(prim::forkClosure, 1))
                    ->setSourceRange(loc);

    // We create a fork by emitting a closure and setting the closure output
    // into the fork input. If a closure doesn't already exist, we create one.
    {
      WithInsertPoint insert(fork_node);
      if (ClosureValue* sv = dynamic_cast<ClosureValue*>(forked.get())) {
        Value* closure_output = sv->asValue(loc, method);
        Block* closure_block = closure_output->node()->blocks().at(0);
        TORCH_INTERNAL_ASSERT(closure_block->outputs().size() == 1);
        out_type = closure_block->outputs().at(0)->type();
        fork_node->addInput(closure_output);
      } else {
        auto emit_closure_body = [&](Block* closure_block) {
          auto fn_sugared_output = forked->call(loc, method, args, kwargs, 1);
          auto fn_simple_output = fn_sugared_output->asValue(loc, method);
          closure_block->registerOutput(fn_simple_output);
          out_type = fn_simple_output->type();
        };
        auto closure_value = emitClosure(emit_closure_body);
        fork_node->addInput(closure_value->asValue(loc, method));
      }
    }
    Value* node_output =
        fork_node->output()->setType(FutureType::create(out_type));
    return std::make_shared<SimpleValue>(node_output);
  }

  std::shared_ptr<SugaredValue> emitRpcExpr(const Apply& apply, Symbol rpc_op) {
    // TODO: This is a temporary apporoach to enable calling user fucntion
    // through RPC in TorchScript,
    // Ideally, function value in JIT IR is first-class citizen and
    // The RPC C++ entry API can take c10::Function directly.
    size_t rpcMinInputs = 2;
    size_t rpcMaxInputs = 5; // NOLINT
    std::string op_name = rpc_op.toUnqualString();
    if (apply.inputs().size() < rpcMinInputs ||
        apply.inputs().size() > rpcMaxInputs) {
      throw ErrorReport(apply)
          << "Possible forms of call to " << op_name << "(..) are\n"
          << op_name
          << "(dst_worker_name, user_callable, args, kwargs, timeout)\n"
          << op_name << "(dst_worker_name, user_callable, args, kwargs)\n"
          << op_name << "(dst_worker_name, user_callable, args)\n"
          << op_name << "(dst_worker_name, user_callable)\n"
          << "Now the number of arguments is " << apply.inputs().size();
    }
    if (apply.attributes().size() != 0) {
      throw ErrorReport(apply)
          << op_name << "(dst_worker_name, user_callable, args, kwargs)"
          << "does not support kwargs yet";
    }
    // TODO: Make rpc_op(..) support taking kwargs,
    // like rpc_async(to="worker1", func=my_func, args=(), kwargs={})

    auto& input_trees = apply.inputs().tree()->trees();
    Value* dst_worker_name_value = emitExpr(Expr(input_trees[0]));
    std::shared_ptr<SugaredValue> user_callable_sugared_value =
        emitSugaredExpr(Expr(input_trees[1]), 1);
    TORCH_CHECK(
        user_callable_sugared_value->kind() == "function",
        "user_callable should be a FunctionValue, it's now a ",
        user_callable_sugared_value->kind())
    // NB: This should be done using `std::dynamic_pointer_cast`
    // and assert `user_callable_function_value != nullptr`. But somehow on
    // macos std::dynamic_pointer_cast always returns
    // `user_callable_function_value` as a `nullptr`, even if
    // `user_callable_sugared_value->kind() == "function"`.
    std::shared_ptr<FunctionValue> user_callable_function_value =
        std::static_pointer_cast<FunctionValue>(user_callable_sugared_value);
    // If `kwargs` is an empty dict, users are allowed to not pass `kwargs`.
    // If `args` and `kwargs` are an empty tuple and an empty dict,
    // respectively, users are allowed to not pass `args` and `kwargs`.

    TreeList args_kwargs_timeout_trees(
        input_trees.begin() + 2, input_trees.end());

    // Get user callable.
    const auto& callablePtrs = user_callable_function_value->callees();
    TORCH_INTERNAL_ASSERT(
        callablePtrs.size() == 1,
        "User-provided callable size should be 1. Now it's",
        callablePtrs.size())
    Function* callablePtr = callablePtrs.at(0);

    const auto& functionSchema = callablePtr->getSchema();
    const SourceRange& loc = apply.range();
    auto graphPtr = method.graph();

    // Match FunctionSchema.
    std::vector<NamedValue> args;
    std::vector<NamedValue> kwargs;
    // Get args and kwargs as `NamedValue`s.
    // Similar to getNamedValues(..) and emitAttributes(..).
    if (args_kwargs_timeout_trees.size() >= 1) {
      // Unroll args from a Var that is known to be a Tuple.
      auto& args_tree = args_kwargs_timeout_trees[0];
      auto entry_sugared_values = emitSugaredExpr(Expr(args_tree), 1)
                                      ->asTuple(args_tree->range(), method);
      args.reserve(entry_sugared_values.size());
      for (const auto& entrie_sugared_value : entry_sugared_values) {
        args.emplace_back(
            args_tree->range(),
            entrie_sugared_value->asValue(args_tree->range(), method));
      }
      // NB: Can't do schema check on kwargs, given the RPC API is
      // rpc_op(to, user_callable, args, kwargs),
      // users can construct kwargs = {"first" + "_arg" : 1}.
      // Notice the key is determined at run time.
      // We can do it at compile time, unless one day the RPC API is
      // rpc_op(to, user_callable, arg_0, arg_1, kwarg_0="foo",
      // kwarg_1="bar")
    }
    matchSchema(functionSchema, loc, *graphPtr, args, kwargs);

    // Graph insert the QualifiedName as an constant input IR Value.
    const auto& qualname = callablePtr->qualname();
    IValue userCallableQualNameIValue(qualname.qualifiedName());
    Value* userCallableQualNameValue =
        graphPtr->insertConstant(userCallableQualNameIValue, loc);

    // Graph insert the corresponding RPC node to the graph.
    Node* rpc_node =
        graphPtr->insertNode(graphPtr->create(rpc_op, 1))->setSourceRange(loc);
    {
      WithInsertPoint insert(rpc_node);
      rpc_node->addInput(dst_worker_name_value);
      rpc_node->addInput(userCallableQualNameValue);

      for (const auto& tree : args_kwargs_timeout_trees) {
        rpc_node->addInput(emitExpr(Expr(tree)));
      }
    }
    Value* rpc_node_output = rpc_node->output();

    // Set output type from FunctionSchema and corresponding rpc_op.
    const std::vector<Argument>& returns = functionSchema.returns();
    TORCH_INTERNAL_ASSERT(returns.size() == 1);
    TypePtr output_type = nullptr;
    if (rpc_op == prim::rpc_async) {
      // rpc_async returns FutureType of the functionSchema's return type
      output_type = FutureType::create(returns[0].type());
    } else if (rpc_op == prim::rpc_sync) {
      // rpc_sync returns the functionSchema's return type
      output_type = returns[0].type();
    } else if (rpc_op == prim::rpc_remote) {
      // rpc_remote returns RRefType of the functionSchema's return type
      output_type = RRefType::create(returns[0].type());
    } else {
      throw ErrorReport(apply)
          << rpc_op.toDisplayString() << " is not supported in TorchScript!'";
    }
    rpc_node_output->setType(output_type);
    return std::make_shared<SimpleValue>(rpc_node_output);
  }

  Value* emitBinaryOp(const TreeRef& tree) {
    const auto& inputs = tree->trees();
    auto kind = getNodeKind(tree->kind(), inputs.size());
    auto overload = getOperatorOverload(tree->kind(), inputs.size());
    auto named_values = getNamedValues(inputs, /*maybe_unpack=*/false);
    if (tree->kind() == TK_IN) {
      // For `in` the arguments are in reverse order (the object being
      // checked is second)
      std::iter_swap(named_values.begin() + 0, named_values.begin() + 1);
    }
    return asSimple(
        makeMagic(
            overload, std::make_shared<BuiltinFunction>(kind, at::nullopt))
            ->call(tree->range(), method, named_values, {}, 0));
  }

  Value* emitSimpleExpr(
      const TreeRef& tree,
      const TypePtr& type_hint = nullptr) {
    switch (tree->kind()) {
      case TK_FLOOR_DIV:
      case '@': {
        const auto& inputs = tree->trees();
        auto kind = getNodeKind(tree->kind(), inputs.size());
        auto named_values = getNamedValues(inputs, /*maybe_unpack=*/false);
        return emitBuiltinCall(
            tree->range(), *method.graph(), kind, named_values, {});
      }
      case '%': {
        auto lhs = emitSugaredExpr(Expr(tree->tree(0)), 0)
                       ->asValue(tree->tree(0)->range(), method);
        auto const& lhs_type = lhs->type();
        if (lhs_type == StringType::get()) {
          auto values = getValues(tree->trees(), /*maybe_unpack=*/false);
          auto node = graph->create(aten::percentFormat, values, 1)
                          ->setSourceRange(tree->range());
          Value* output = graph->insertNode(node)->output();
          output->setType(StringType::get());
          return output;
        } else {
          return emitBinaryOp(tree);
        }
      }
      case TK_IN:
      case TK_POW:
      case TK_NE:
      case TK_EQ:
      case '<':
      case '>':
      case TK_LE:
      case TK_GE:
      case '*':
      case '/':
      case '+':
      case '-':
      case '&':
      case '|':
      case '^':
      case TK_LSHIFT:
      case TK_RSHIFT:
        return emitBinaryOp(tree);
      case TK_IS:
      case TK_ISNOT:
      case TK_AND:
      case TK_OR:
      case TK_NOT: {
        return emitCondExpr(Expr(tree)).value();
      }
      case TK_UNARY_MINUS: {
        return emitUnaryOp(tree, "__neg__", aten::neg);
      }
      case '~': {
        return emitUnaryOp(tree, "__invert__", aten::bitwise_not);
      }
      case TK_STARRED: {
        throw ErrorReport(tree)
            << "Unexpected starred expansion. File a bug report";
      }
      case TK_CONST: {
        return emitConst(Const(tree));
      } break;
      case TK_TRUE: {
        return graph->insertConstant(true, tree->range());
      } break;
      case TK_FALSE: {
        return graph->insertConstant(false, tree->range());
      } break;
      case TK_NONE: {
        return graph->insertConstant(IValue(), tree->range());
      } break;
      case TK_IF_EXPR: {
        return emitTernaryIf(TernaryIf(tree));
      } break;
      case TK_STRINGLITERAL: {
        return emitStringLiteral(StringLiteral(tree));
      } break;
      case TK_LIST_LITERAL: {
        auto ll = ListLiteral(tree);
        auto values = getValues(ll.inputs(), /*maybe_unpack=*/true);

        // determine the element type of the list
        // if we have a type hint of List[T], use T
        // if the list is non-empty use type_of(list[0])
        // otherwise assume it is List[Tensor]
        TypePtr elem_type = TensorType::get();
        if (type_hint) {
          if (type_hint->kind() == TypeKind::ListType) {
            elem_type = type_hint->expect<ListType>()->getElementType();
          } else {
            // If the type hint was not a List[T] throw an error
            throw ErrorReport(tree)
                << "Expected a List type hint but instead got "
                << type_hint->repr_str();
          }
        } else if (!values.empty()) {
          std::stringstream ss;
          auto types = fmap(values, [](const Value* v) { return v->type(); });
          auto maybe_elem_type = unifyTypeList(types, ss);
          if (!maybe_elem_type) {
            throw ErrorReport(tree) << "Lists must contain only a single type\n"
                                    << ss.str();
          }
          elem_type = maybe_elem_type.value();
        }

        for (auto v : values) {
          std::stringstream ss;
          if (!v->type()->isSubtypeOfExt(elem_type, &ss)) {
            throw ErrorReport(tree)
                << "Lists must contain only a single type, expected: "
                << elem_type->repr_str() << " but found "
                << v->type()->repr_str() << " instead.\n"
                << ss.str();
          }
        }
        Value* result =
            graph->insertNode(graph->createList(elem_type, values))->output();
        return result;
      } break;
      case TK_TUPLE_LITERAL: {
        auto ll = TupleLiteral(tree);
        auto values = getValues(ll.inputs(), /*maybe_unpack=*/true);
        return graph->insertNode(graph->createTuple(values))->output();
      } break;
      case TK_DICT_LITERAL: {
        auto dl = DictLiteral(tree);
        auto key_trees = dl.key_inputs().tree()->trees();
        auto value_trees = dl.value_inputs().tree()->trees();
        AT_ASSERT(key_trees.size() == value_trees.size());
        std::vector<Value*> keys, values;

        for (size_t i = 0; i < key_trees.size(); ++i) {
          keys.push_back(emitExpr(Expr(key_trees[i])));
          values.push_back(emitExpr(Expr(value_trees[i])));
        }

        TypePtr key_type = nullptr;
        TypePtr value_type = nullptr;

        if (type_hint && type_hint->kind() == TypeKind::DictType) {
          auto dict_type = type_hint->expect<DictType>();
          key_type = dict_type->getKeyType();
          value_type = dict_type->getValueType();
        } else if (keys.empty()) {
          key_type = StringType::get();
          value_type = TensorType::get();
        } else {
          key_type = keys.at(0)->type();
          value_type = values.at(0)->type();
        }
        AT_ASSERT(key_type != nullptr && value_type != nullptr);

        auto checkTypeOfValues = [](const TypePtr& type,
                                    const char* what,
                                    const std::vector<Value*>& values,
                                    TreeList trees) {
          for (size_t i = 0, N = values.size(); i < N; ++i) {
            std::stringstream ss;
            if (!values[i]->type()->isSubtypeOfExt(type, &ss)) {
              throw ErrorReport(trees[i])
                  << "Dict " << what
                  << " must contain only a single type, expected: "
                  << type->repr_str() << " but found "
                  << values[i]->type()->repr_str() << " instead.\n"
                  << ss.str();
            }
          }
        };
        checkTypeOfValues(key_type, "keys", keys, key_trees);
        checkTypeOfValues(value_type, "values", values, value_trees);

        return graph
            ->insertNode(graph->createDict(key_type, value_type, keys, values))
            ->output();
      } break;
      case TK_LIST_COMP: {
        auto lc = ListComp(tree);
        return emitListComprehension(lc, type_hint);
      } break;
      default:
        throw ErrorReport(tree) << "Cannot emit expr for: " << tree;
    }
  }

  Value* emitConst(const Const& c) {
    if (c.isFloatingPoint())
      return materializeConstant(
          c.asFloatingPoint(), *graph, c.range(), fp_constants);
    else
      return materializeConstant(
          c.asIntegral(), *graph, c.range(), integral_constants);
  }

  Value* emitStringLiteral(const StringLiteral& c) {
    return insertConstant(*graph, c.text(), c.range());
  }

  // Desugars select indexing: tensor[i] -> tensor.select(dim, i)
  Value* emitSelect(
      const SourceRange& loc,
      Value* input,
      Value* dim,
      Value* index) {
    return emitBuiltinCall(loc, *graph, aten::select, {input, dim, index}, {});
  }

  Value* emitSliceOp(
      const SourceRange& loc,
      Value* sliceable,
      Value* dim,
      Value* start,
      Value* end,
      Value* step) {
    std::vector<NamedValue> args;
    args.reserve(4);
    args.emplace_back(loc, "self", sliceable);

    // XXX: If list slicing becomes more complicated or stops using
    // aten::slice, we should separate it from this function.
    if (dim) {
      AT_ASSERT(sliceable->type()->isSubtypeOf(TensorType::get()));

      args.emplace_back(dim);
    } else {
      AT_ASSERT(!sliceable->type()->isSubtypeOf(TensorType::get()));
    }
    // TODO for now let's deal with TupleType first. Ideally all list, tensor,
    // string, and tuple slicing should be same (tugsbayasgalan)
    if (sliceable->type()->cast<TupleType>()) {
      std::vector<at::optional<NamedValue>> tuple_args;
      // since we are only dealing with tuple slicing for now, we try to keep
      // tuple args seperate for now
      tuple_args.reserve(3);

      start ? tuple_args.emplace_back(start)
            : tuple_args.emplace_back(c10::nullopt);
      end ? tuple_args.emplace_back(end)
          : tuple_args.emplace_back(c10::nullopt);
      step ? tuple_args.emplace_back(step)
           : tuple_args.emplace_back(c10::nullopt);

      return emitTupleSlice(loc, args[0], tuple_args);
    }

    // TODO this needs to be cleaned for list slicing
    // Default value for start is 0.
    if (!start) {
      start = graph->insertConstant(0, loc);
    }
    args.emplace_back(loc, "start", start);

    if (end) {
      args.emplace_back(loc, "end", end);
    }

    if (!step) {
      step = graph->insertConstant(1, loc);
    }
    NamedValue step_nv = NamedValue(loc, "step", step);
    return emitBuiltinCall(loc, *graph, aten::slice, args, {step_nv});
  }

  // Desugars slice indexing: tensor[begin:end] -> tensor.slice(dim, begin, end,
  // 1)
  Value* emitSlice(
      const SourceRange& loc,
      Value* input,
      Value* dim, // Only used for tensor slicing
      const SliceExpr& slice) {
    Value* start = nullptr;
    Value* end = nullptr;
    Value* step = nullptr;
    if (slice.start().present()) {
      start = emitExpr(Expr(slice.start().get()));
    }
    if (slice.end().present()) {
      end = emitExpr(Expr(slice.end().get()));
    }
    if (slice.step().present()) {
      step = emitExpr(Expr(slice.step().get()));
    }
    return emitSliceOp(loc, input, dim, start, end, step);
  }

  Value* emitUnsqueeze(const SourceRange& loc, Value* input, Value* dim_val) {
    return emitBuiltinCall(loc, *graph, aten::unsqueeze, {input, dim_val}, {});
  }

  Value* emitIndex(
      const SourceRange& loc,
      Value* input,
      at::ArrayRef<Value*> indices) {
    // NB: the index of aten::index should be a type of List[Optional[Tensor]],
    // this is to support the case like t[:, :, 1] where : here indicates a
    // None/undefined tensor(optional tensor)
    auto* index =
        graph->insertNode(graph->createList(OptionalType::ofTensor(), indices))
            ->output();
    return emitBuiltinCall(loc, *graph, aten::index, {input, index}, {});
  }

  // Emits multidimensional slicing with int and slice indices.
  // Returns:
  // - Value*: the input after it has been indexed by int and slice indices.
  // - vector<Value*>: A list of tensor Value* indices that have not been
  // applied yet.
  //   Should be NULL at indices where sliceable (post-slicing) isn't indexed by
  //   a tensor.
  std::pair<Value*, std::vector<Value*>> emitIntAndSliceIndexing(
      const SourceRange& loc,
      Value* sliceable,
      const List<Expr>& subscript_exprs) {
    // Overall, to handle indexing (other than Tensors), we need to handle a
    // couple different things. For example, for x[1:3, None, 4], each of these
    // different index types (slice, None, and integer) result in different
    // number of dimensions. Slicing doesn't change the number of dimensions,
    // None adds a dimension, and integer removes a dimension. As these indexing
    // operations are applied left to right, the actual index that it's being
    // applied to depends on the previous operations. Ellipses indexing throws
    // another wrinkle. Ellipses selects any remaining unspecified dimensions.
    // Thus, for indexes following an ellipses, the actual index an indexing
    // operation is being applied to depends on the operations to the right.
    // Thus, we do two passes, one from left to right up until the ellipses, and
    // one from right to left.

    std::vector<Value*> tensor_indices;

    auto insert_value_for_dim = [&](int64_t dim) {
      return graph->insertConstant(dim, loc);
    };
    std::vector<int64_t> dims(subscript_exprs.size());
    std::vector<c10::optional<Value*>> exprs(
        subscript_exprs.size(), c10::nullopt);

    auto handle_indexing = [&](const Expr& subscript_expr,
                               int expr_idx,
                               int64_t dim,
                               bool is_reverse = false) {
      dims[expr_idx] = dim;

      // Slice expression case, does not represent a single index.
      if (subscript_expr.kind() == TK_SLICE_EXPR) {
        if (is_reverse) {
          return dim - 1;
        } else {
          return dim + 1;
        }
      }

      // Slice object case, does not represent a single index.
      auto subscript_sv = emitSugaredExpr(subscript_expr, 1);
      if (dynamic_cast<SliceValue*>(subscript_sv.get())) {
        if (is_reverse) {
          return dim - 1;
        } else {
          return dim + 1;
        }
      }

      TypePtr type_hint;
      if (subscript_expr.kind() == TK_NONE) {
        type_hint = NoneType::get();
      }
      auto index = emitExpr(subscript_expr, type_hint);

      // Accept list as subscript but convert it to a Tensor
      // since it's equivalent to indexing with Tensor.
      // The list can be a list literal or list variable.
      // Advanced indexing using list:
      // @torch.jit.script
      // def f(x):
      //   return x[[0, 1, 5]]  # or
      //   return x[[0, 1], [0, 1]]  # or
      //   return x[[[0, 1], [0, 1]], [[0, 1], [0, 1]]]  # or
      //   ls = [0, 1]
      //   return x[ls]
      // Statements above are equivalent to advanced indexing using Tensor:
      // @torch.jit.script
      // def f(x):
      //   return x[torch.tensor([0, 1, 5])]  # or
      //   return x[torch.tensor([0, 1]), torch.tensor([0, 1])]  # or
      //   return x[torch.tensor([[0, 1], [0, 1]]),
      //            torch.tensor([[0, 1], [0, 1]])]  # or
      //   ls = [0, 1]
      //   return x[torch.tensor(ls)]
      if (index->type()->kind() == c10::TypeKind::ListType) {
        // Always create index tensor as LongTensor.
        // This is to match Pytorch eager frontend behavior which accepts
        // indexing with float list.
        index = graph->insert(
            aten::tensor, {index}, {NamedValue("dtype", c10::kLong)});
      }

      exprs[expr_idx] = index;
      if (index->type()->isSubtypeOf(NoneType::get())) {
        if (is_reverse) {
          return dim;
        } else {
          return dim + 1;
        }
      } else if (index->type() == IntType::get()) {
        if (is_reverse) {
          return dim - 1;
        } else {
          return dim;
        }
      } else if (index->type()->isSubtypeOf(OptionalType::ofTensor())) {
        if (is_reverse) {
          throw ErrorReport(loc)
              << "Ellipses followed by tensor indexing is currently not supported";
        } else {
          return dim + 1;
        }
      } else {
        throw ErrorReport(loc)
            << "Unsupported operation: indexing tensor with unsupported index type '"
            << index->type()->repr_str()
            << "'. Only ints, slices, lists and tensors are supported";
      }
    };

    size_t idx = 0;
    int64_t dim = 0;
    for (; idx < subscript_exprs.size(); idx++) {
      auto subscript_expr = subscript_exprs[idx];
      if (subscript_expr.kind() == TK_DOTS) {
        break;
      }
      dim = handle_indexing(subscript_expr, idx, dim, /*is_reverse=*/false);
    }
    int64_t rdim = -1;
    for (size_t rev_idx = subscript_exprs.size() - 1; rev_idx > idx;
         rev_idx--) {
      auto subscript_expr = subscript_exprs[rev_idx];
      if (subscript_expr.kind() == TK_DOTS) {
        throw ErrorReport(loc)
            << "An index can only have a single ellipsis ('...')";
      }
      rdim =
          handle_indexing(subscript_expr, rev_idx, rdim, /*is_reverse=*/true);
    }
    for (size_t i = 0; i < exprs.size(); i++) {
      if (!exprs[i].has_value()) {
        if (subscript_exprs[i].kind() == TK_SLICE_EXPR) {
          sliceable = emitSlice(
              loc,
              sliceable,
              insert_value_for_dim(dims[i]),
              SliceExpr(subscript_exprs[i]));
          continue;
        }

        if (subscript_exprs[i].kind() == TK_DOTS) {
          continue;
        }

        auto subscript_sv = emitSugaredExpr(subscript_exprs[i], 1);
        if (const auto slice_value =
                dynamic_cast<SliceValue*>(subscript_sv.get())) {
          sliceable = emitSliceOp(
              loc,
              sliceable,
              insert_value_for_dim(dims[i]),
              slice_value->start(),
              slice_value->stop(),
              slice_value->step());
        }

        continue;
      }
      auto expr = exprs[i].value();
      if (expr->type()->isSubtypeOf(NoneType::get())) {
        sliceable =
            emitUnsqueeze(loc, sliceable, insert_value_for_dim(dims[i]));
      } else if (expr->type() == IntType::get()) {
        sliceable =
            emitSelect(loc, sliceable, insert_value_for_dim(dims[i]), expr);
      } else if (expr->type()->isSubtypeOf(OptionalType::ofTensor())) {
        tensor_indices.resize(dims[i] + 1);
        tensor_indices[dims[i]] = expr;
      } else {
        TORCH_INTERNAL_ASSERT(
            false, "Trying to process index type that we don't support.");
      }
    }
    // at::index takes in a List[Optional[Tensor]] where some dims can be None.
    // create None node with optional tensor output type and pass to at::index.
    for (auto& index : tensor_indices) {
      if (index == nullptr) {
        index = graph->insertNode(graph->createNone())->output();
      }
    }
    return std::make_pair(sliceable, tensor_indices);
  }

  // Desugars multidim slicing into slice/select/index/unsqueeze calls.
  //
  // XXX: Errors in user code are not elegantly reported.
  // Let's say someone were to do the following:
  //   @torch.jit.script
  //   def fn(x):
  //       return x[0, 1]
  //   fn(torch.randn(5))
  // Because we desugar this into two aten::select ops, the error message
  // complains about aten::select failing rather than there "not being
  // enough dimensions to index".
  //
  // The strategy is to slice and select the tensor for int and slices first
  // in one pass and then apply at::index on the result of the
  // slicing/selecting. Call the tensor after we've applied slice / select the
  // `sliced`. tensor_indices should have the same size as sliced.dim():
  // - tensor_indices[i] = NULL if we should not index `sliced` at dim i
  // - tensor_indices[i] = t if we should index `sliced` at dim i with tensor t.
  Value* emitMultidimSlicing(
      const SourceRange& loc,
      Value* sliceable,
      const List<Expr>& subscript_exprs) {
    if (!sliceable->type()->isSubtypeOf(TensorType::get())) {
      throw ErrorReport(loc)
          << "Unsupported operation: attempted to use multidimensional "
          << "indexing on a non-tensor type";
    }

    std::vector<Value*> tensor_indices;
    std::tie(sliceable, tensor_indices) =
        emitIntAndSliceIndexing(loc, sliceable, subscript_exprs);

    if (tensor_indices.empty()) {
      // XXX: Might need to at::alias this when we support mutability
      return sliceable;
    }

    return emitIndex(loc, sliceable, tensor_indices);
  }

  // Desugars slice syntactic sugar tensor[begin:end] -> tensor.slice(begin,
  // end).
  Value* emitBasicSlice(
      const SourceRange& loc,
      Value* sliceable,
      const List<Expr>& subscript_exprs) {
    AT_ASSERT(subscript_exprs.size() == 1);
    AT_ASSERT(subscript_exprs[0].kind() == TK_SLICE_EXPR);
    auto slice_exp = SliceExpr(subscript_exprs[0]);
    Value* maybe_dim = nullptr;
    if (sliceable->type()->isSubtypeOf(TensorType::get())) {
      // If the sliceable object is a tensor, specify a default dimension
      maybe_dim = graph->insertConstant(0, loc);
    }
    return emitSlice(loc, sliceable, maybe_dim, slice_exp);
  }

  int64_t getAdjTupleIndex(
      const SourceRange& loc,
      const TupleTypePtr& tuple_type,
      int64_t input_index,
      bool allow_out_of_bounds) {
    // set index to be positive to simplify logic in runtime
    int64_t adj_index = input_index;
    int64_t tuple_len = tuple_type->elements().size();
    if (input_index < 0) {
      adj_index = tuple_len + input_index;
    }
    if (!allow_out_of_bounds && (adj_index >= tuple_len || adj_index < 0)) {
      throw ErrorReport(loc) << "Tuple index out of range. Tuple is length "
                             << tuple_len << " and index is " << input_index;
    }
    return adj_index;
  }

  // When a list is marked const in a module, it gets converted to a tuple.
  // The result is indexing into a Tuple which contains only one type
  // is quite common. since indexing will likely be done in a for loop,
  // we do not want to invoke the overhead of converting the tuple to a list
  // each iter.
  Value* emitTupleIndex(
      const SourceRange& loc,
      Value* tuple_val,
      Value* idx_val) {
    auto tuple_typ = tuple_val->type()->cast<TupleType>();
    auto elems = tuple_typ->elements();
    TypePtr output_type;
    if (idx_val->type() != IntType::get()) {
      throw ErrorReport(loc) << "tuple index must be an integer";
    }
    auto idx = toIValue(idx_val);
    if (!idx) {
      if (elems.size() == 0 ||
          !convertibleToList(tuple_typ, ListType::create(elems[0]))) {
        throw ErrorReport(loc)
            << "Cannot index into a " << tuple_typ->repr_str()
            << " with a non-integer literal because we cannot resolve the output type";
      }
      output_type = elems[0];
    } else {
      auto adj_index = getAdjTupleIndex(
          loc, tuple_typ, idx->toInt(), /*allow_out_of_bounds*/ false);
      output_type = elems[adj_index];
    }
    return graph
        ->insertNode(graph->createTupleIndex(tuple_val, idx_val, output_type))
        ->output();
  }

  int64_t getSliceInd(Value* idx_val, const SourceRange& loc) {
    auto ivalue = toIValue(idx_val);
    if (ivalue && ivalue->isInt()) {
      return ivalue->to<int64_t>();
    } else {
      throw ErrorReport(loc) << "tuple slice indices must be integer constants";
    }
  }

  Value* emitTupleSlice(
      const SourceRange& loc,
      const NamedValue& tuple_val,
      const std::vector<at::optional<NamedValue>>& tuple_args) {
    auto tuple_type = tuple_val.value(*graph)->type()->expect<TupleType>();
    int64_t tuple_len = tuple_type->elements().size();
    auto beg_val = tuple_args[0];
    auto end_val = tuple_args[1];
    auto step = tuple_args[2];

    int64_t step_size = 1;
    if (step) {
      auto val = toIValue(step->value(*graph));
      TORCH_CHECK(val->isInt(), "Step size should always be an integer");
      step_size = val->to<int64_t>();
    }

    int64_t beg = std::numeric_limits<int64_t>::max();
    if (beg_val) {
      beg = getAdjTupleIndex(
          loc, tuple_type, getSliceInd(beg_val->value(*graph), loc), true);
    }

    int64_t end = std::numeric_limits<int64_t>::max();
    if (end_val) {
      end = getAdjTupleIndex(
          loc, tuple_type, getSliceInd(end_val->value(*graph), loc), true);
    }

    int64_t num_values = slice_indices_adjust(tuple_len, &beg, &end, step_size);

    return graph
        ->insertNode(graph->createTupleSlice(
            tuple_val.value(*graph), beg, step_size, num_values))
        ->output();
  }

  std::shared_ptr<SugaredValue> emitSubscript(
      const Subscript& subscript,
      TypePtr type_hint = nullptr) {
    const SugaredValuePtr sv = emitSugaredExpr(subscript.value(), 1);
    const List<Expr>& subscript_exprs = subscript.subscript_exprs();
    const SourceRange& range = subscript.range();
    const SourceRange& val_range = subscript.value().range();
    if (subscript_exprs.size() != 1) {
      return std::make_shared<SimpleValue>(emitMultidimSlicing(
          range, sv->asValue(val_range, method), subscript_exprs));
    }
    if (subscript_exprs[0].kind() == TK_SLICE_EXPR) {
      // TODO @wconstab refactor using Symbol instead of string compare
      if (sv->kind() == "module") {
        // Slicing isn't currently implemented for Sequential/ModuleList,
        // but is implemented for Tuples, so a quick workaround is to
        // convert to a tuple of Modules for slicing support.
        auto s_tuple_val =
            sv->asTupleValue(val_range, method)->asValue(val_range, method);
        const SliceExpr& slice = SliceExpr(subscript_exprs[0]);
        std::vector<at::optional<NamedValue>> tuple_args;
        tuple_args.reserve(3);
        auto begin =
            NamedValue(val_range, "begin", emitExpr(Expr(slice.startOr(0))));
        tuple_args.emplace_back(begin);
        if (slice.end().present()) {
          auto end =
              NamedValue(val_range, "end", emitExpr(Expr(slice.end().get())));
          tuple_args.emplace_back(end);

        } else {
          tuple_args.emplace_back(c10::nullopt);
        }
        // pushing step_size to match the tuple_args
        tuple_args.emplace_back(c10::nullopt);

        auto tupleSliceValue =
            emitTupleSlice(val_range, s_tuple_val, tuple_args);
        return std::make_shared<SimpleValue>(tupleSliceValue);
      } else {
        return std::make_shared<SimpleValue>(emitBasicSlice(
            range, sv->asValue(val_range, method), subscript_exprs));
      }
    } else {
      AT_ASSERT(subscript_exprs.size() == 1);
      Value* sliceable = sv->asValue(val_range, method);

      // In case of subscript expression being a Python Slice object.
      auto subscript_sv = emitSugaredExpr(subscript_exprs[0], 1);
      if (const auto slice_value =
              dynamic_cast<SliceValue*>(subscript_sv.get())) {
        Value* dim = nullptr;
        // aten::slice.tensor needs an additional `dim` input.
        if (sliceable->type()->isSubtypeOf(TensorType::get())) {
          dim = method.graph()->insertConstant(0, val_range);
        }

        Value* sliced = emitSliceOp(
            val_range,
            sliceable,
            dim,
            slice_value->start(),
            slice_value->stop(),
            slice_value->step());
        return std::make_shared<SimpleValue>(sliced);
      }

      // subscript is not a slice object, then it must be convertible to
      // a normal value.
      // Desugars gather syntactic sugar foo[i]
      Value* idx = subscript_sv->asValue(val_range, method);
      if (sliceable->type()->cast<TupleType>()) {
        return std::make_shared<SimpleValue>(
            emitTupleIndex(range, sv->asValue(val_range, method), idx));
      } else if (sliceable->type()->isSubtypeOf(TensorType::get())) {
        return std::make_shared<SimpleValue>(
            emitMultidimSlicing(range, sliceable, subscript_exprs));
      } else {
        return sv->getitem(range, method, idx, std::move(type_hint));
      }
    }
  }
};

struct FunctionResolver : public Resolver {
  explicit FunctionResolver(
      Resolver* otherResolver,
      const std::unordered_map<std::string, Function*>& functionTable)
      : otherResolver_(otherResolver), functionTable_(functionTable) {}

  std::shared_ptr<SugaredValue> resolveValue(
      const std::string& name,
      Function& m,
      const SourceRange& loc) override {
    auto it = functionTable_.find(name);
    if (it != functionTable_.end()) {
      return std::make_shared<FunctionValue>(it->second);
    }
    return otherResolver_->resolveValue(name, m, loc);
  }

  TypePtr resolveType(const std::string& name, const SourceRange& loc)
      override {
    return otherResolver_->resolveType(name, loc);
  }

 private:
  Resolver* otherResolver_;
  const std::unordered_map<std::string, Function*>& functionTable_;
};

CompilationUnit::CompilationUnit(const std::string& source)
    : CompilationUnit() {
  // calles the define with native resolver to generate the graph for functions
  define(c10::nullopt, source, nativeResolver(), nullptr);
}

// This pair represents a pair of functions (getter and setter) obtained from
// compiling a Property.
struct CompilationUnit::PropertyPair
    : public std::pair<std::unique_ptr<Function>, std::unique_ptr<Function>> {
  PropertyPair(
      std::unique_ptr<Function> getter,
      std::unique_ptr<Function> setter) {
    TORCH_INTERNAL_ASSERT(getter, "Property pair must have defined getter")
    this->first = std::move(getter);
    this->second = std::move(setter);
  }

  std::unique_ptr<Function>& getGetter() {
    return this->first;
  }

  std::unique_ptr<Function>& getSetter() {
    return this->second;
  }
};

CompilationUnit::PropertyPair CompilationUnit::define_property(
    const c10::optional<c10::QualifiedName>& prefix,
    const Property& prop,
    const ResolverPtr& resolver,
    const Self* self,
    const std::unordered_map<std::string, Function*>& function_table,
    bool shouldMangle) const {
  // self must be defined because properties are features of classes and
  // modules.
  TORCH_INTERNAL_ASSERT(self);

  // Compile the getter function.
  std::unique_ptr<Function> getter_fn = define(
      prefix, prop.getter(), resolver, self, function_table, shouldMangle);

  // Compile the setter function if it exists.
  std::unique_ptr<Function> setter_fn = nullptr;
  if (prop.setter().present()) {
    setter_fn = define(
        prefix,
        prop.setter().get(),
        resolver,
        self,
        function_table,
        shouldMangle);
  }

  // Add the property to the class type definition.
  self->getClassType()->addProperty(
      prop.name().name(), getter_fn.get(), setter_fn.get());

  return PropertyPair(std::move(getter_fn), std::move(setter_fn));
}

std::unique_ptr<Function> CompilationUnit::define(
    const c10::optional<QualifiedName>& prefix,
    const Def& def,
    const ResolverPtr& resolver,
    const Self* self,
    const std::unordered_map<std::string, Function*>& function_table,
    bool shouldMangle) const {
  TORCH_INTERNAL_ASSERT(resolver);
  auto _resolver = resolver;
  if (!self) {
    // if self is defined, then these are methods and do not go into the
    // global namespace otherwise, they get defined together so we add them to
    // the function table so the methods can see each other
    _resolver =
        std::make_shared<FunctionResolver>(resolver.get(), function_table);
  }
  auto creator = [def, _resolver, self](Function& method) {
    // Store the function name so that it can be referenced if there is an error
    // while compiling this function
    std::string call_name = method.qualname().name();
    if (self) {
      auto atoms = method.qualname().atoms();
      // There should be at least a ClassName.method_name
      TORCH_INTERNAL_ASSERT(atoms.size() >= 2);
      call_name = atoms.at(atoms.size() - 2) + "." + atoms.at(atoms.size() - 1);
    }
    ErrorReport::CallStack call(call_name, def.range());
    to_ir(def, _resolver, self, method);
  };
  auto name = prefix ? QualifiedName(*prefix, def.name().name())
                     : QualifiedName(def.name().name());
  if (shouldMangle) {
    // If `shouldMangle` is set, we should generate a unique name for this
    // function if there is already an existing one.
    if (auto fn = find_function(name)) {
      name = mangle(name);
    }
  }
  auto fn = torch::make_unique<GraphFunction>(
      std::move(name), std::make_shared<Graph>(), creator);
  if (self) {
    // Register this as a method on `self`'s type
    self->getClassType()->addMethod(fn.get());
  }
  return fn;
}

std::vector<Function*> CompilationUnit::define(
    const c10::optional<c10::QualifiedName>& prefix,
    const std::vector<Property>& properties,
    const std::vector<ResolverPtr>& propResolvers,
    const std::vector<Def>& definitions,
    const std::vector<ResolverPtr>& defResolvers,
    const Self* self,
    bool shouldMangle) {
  TORCH_INTERNAL_ASSERT(definitions.size() == defResolvers.size());
  TORCH_INTERNAL_ASSERT(properties.size() == propResolvers.size());
  std::vector<Function*> functions;
  std::unordered_map<std::string, Function*> function_table;

  // Records fn in function_table, functions and with register_function.
  // This is done several times below, so this lambda helps avoid repeating
  // code.
  auto record_function = [&](std::unique_ptr<Function> fn) {
    function_table[fn->name()] = fn.get();
    functions.emplace_back(fn.get());
    this->register_function(std::move(fn));
  };

  for (size_t i = 0; i < properties.size(); i++) {
    PropertyPair property_fns = define_property(
        prefix,
        properties[i],
        propResolvers[i],
        self,
        function_table,
        shouldMangle);

    auto& getter_fn = property_fns.getGetter();
    auto& setter_fn = property_fns.getSetter();

    record_function(std::move(getter_fn));

    if (setter_fn) {
      record_function(std::move(setter_fn));
    }
  }

  for (size_t i = 0; i < definitions.size(); i++) {
    auto fn = define(
        prefix,
        definitions[i],
        defResolvers[i],
        self,
        function_table,
        shouldMangle);

    record_function(std::move(fn));
  }

  // We need to compile `__init__` first, since it can determine what attributes
  // are available to other methods. So reorder the definitions accordingly.
  for (auto& kv : function_table) {
    if (kv.first == "__init__") {
      kv.second->ensure_defined();
    }
  }

  for (Function* function : functions) {
    function->ensure_defined();
  }

  return functions;
}

std::vector<Function*> CompilationUnit::define(
    const c10::optional<QualifiedName>& prefix,
    const std::string& source,
    const ResolverPtr& resolver,
    const Self* self) {
  Parser p(std::make_shared<Source>(source, "<string>", 1));
  std::vector<Def> definitions;
  std::vector<ResolverPtr> resolvers;
  while (p.lexer().cur().kind != TK_EOF) {
    auto def = Def(p.parseFunction(/*is_method=*/bool(self)));
    definitions.push_back(def);
    resolvers.push_back(resolver);
  }
  return define(
      prefix,
      /*properties=*/{},
      /*propResolvers=*/{},
      definitions,
      resolvers,
      self);
}

void runCleanupPasses(std::shared_ptr<Graph>& to_clean) {
  liftClosures(to_clean);
  inlineForkedClosures(to_clean);
  if (getInlineEverythingMode()) {
    Inline(*to_clean);
  }

  // remove any uses of tuples that we inserted that are not needed
  LowerSimpleTuples(to_clean);

  // full constant propagation runs ops with mutable inputs if it can
  // prove that the inputs are not mutated anywhere in the graph.
  // if a mutating node is removed in the graph (e.g. constant prop inlined a
  // a constant if) then the next time constant prop is run it might be able
  // to run nodes it was not able to previously, and the graph may change
  // (jitter) So we run only constant prop w immutable types here bc
  // successive runs of immutable constant prop does not change the graph
  ConstantPropagationImmutableTypes(to_clean);

  // Constant Pooling pass must be after ConstantPropogation, which can create
  // new constants that needs to be pooled.
  ConstantPooling(to_clean);

  // For jitter
  CanonicalizeOutputs(to_clean);

  // Annotate aten::warns so that each has its unique ID. This enables us to
  // mimic Python behavior of only emitting each warning only once.
  AnnotateWarns(to_clean);
}

// we consider _N where N is a number, to be a non-meaningful name
// and do not record it as a unique name. This allows python printing to
// be able to export and import more consistently named graphs
bool meaningfulName(const std::string& name) {
  if (name.size() == 0)
    return false;
  if (name[0] == '$')
    return false;
  if (name[0] != '_')
    return true;
  for (size_t i = 1; i < name.size(); ++i) {
    if (!isdigit(name[i]))
      return true;
  }
  return false;
}

void CompilationUnit::define_interface(
    const c10::QualifiedName& qualifiedName,
    const ClassDef& classDef,
    ResolverPtr rcb,
    bool is_module) {
  ScriptTypeParser typeParser(std::move(rcb));
  InterfaceTypePtr iface =
      InterfaceType::create(c10::QualifiedName(qualifiedName), is_module);
  for (const Stmt& stmt : classDef.body()) {
    if (stmt.kind() != TK_DEF) {
      throw ErrorReport(stmt)
          << "interface declartions can only contain method definitions";
    }
    auto method_def = Def(stmt);
    if (!method_def.decl().return_type().present()) {
      throw ErrorReport(method_def)
          << "interface declarations must have a return type annotated.";
    }
    FunctionSchema schema =
        typeParser.parseSchemaFromDef(method_def, /* skip_self*/ true);
    // need to add self as the first because we skipped it
    std::vector<Argument> arguments;
    arguments.emplace_back(
        Argument(method_def.decl().params()[0].ident().name(), iface));
    arguments.insert(
        arguments.end(), schema.arguments().begin(), schema.arguments().end());
    iface->addMethod(schema.cloneWithArguments(std::move(arguments)));
    if (method_def.statements().size() != 1 ||
        method_def.statements()[0].kind() != TK_PASS) {
      throw ErrorReport(method_def.range())
          << "interfaces declarations should only contain a single 'pass' statement.";
    }
  }
  this->register_type(iface);
}

} // namespace jit
} // namespace torch
