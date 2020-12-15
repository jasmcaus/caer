#include <torch/csrc/jit/ir/scope.h>
#include <ATen/core/function.h>

namespace torch {
namespace jit {

ScopePtr Scope::intrusive_from_this() {
  c10::raw::intrusive_ptr::incref(this); // we are creating a new pointer
                                         // from a raw `this` pointer
                                         // so we need to bump the refcount
                                         // to account for this ownership
  return c10::intrusive_ptr<Scope>::reclaim(this);
}

Scope::Scope() {
  name_ = Symbol::scope("");
}

Scope::Scope(ScopePtr parent, Symbol name) {
  name_ = name;
  parent_ = std::move(parent);
}

ScopePtr Scope::push(Symbol name) {
  return c10::make_intrusive<Scope>(intrusive_from_this(), name);
}

ScopePtr Scope::parent() {
  if (!parent_) {
    throw std::runtime_error("Cannot get parent from Scope with no parent");
  }
  return parent_;
}

bool Scope::isRoot() const {
  return !parent_;
}

bool Scope::isBlank() const {
  static const Symbol blank = Symbol::scope("");
  return isRoot() && name() == blank;
}

ScopePtr Scope::getRoot() {
  ScopePtr current = intrusive_from_this();
  while (current->parent_) {
    current = current->parent_;
  }
  return current;
}

size_t Scope::getDepth() {
  size_t d = 1;
  ScopePtr current = intrusive_from_this();
  while (current->parent_) {
    current = current->parent_;
    d += 1;
  }
  return d;
}

Symbol Scope::name() const {
  return name_;
}

std::string Scope::namesFromRoot(const std::string& separator) const {
  // TODO: I think the answer is we shouldn't have used Symbol here
  std::string out = this->name_.toUnqualString();
  if (this->isRoot()) {
    return out;
  }
  ScopePtr parent = this->parent_;
  while (!parent->isRoot()) {
    // NOLINTNEXTLINE(performance-inefficient-string-concatenation)
    out = std::string(parent->name_.toUnqualString()) + separator + out;
    parent = parent->parent_;
  }
  return out;
}

InlinedCallStackPtr InlinedCallStack::intrusive_from_this() {
  c10::raw::intrusive_ptr::incref(this); // we are creating a new pointer
                                         // from a raw `this` pointer
                                         // so we need to bump the refcount
                                         // to account for this ownership
  return c10::intrusive_ptr<InlinedCallStack>::reclaim(this);
}

InlinedCallStack::InlinedCallStack(Function* fn, SourceRange source_range)
    : fn_(fn), source_range_(std::move(source_range)) {}

InlinedCallStack::InlinedCallStack(
    Function* fn,
    SourceRange source_range,
    c10::optional<ModuleInstanceInfo> module_instance_info)
    : fn_(fn),
      source_range_(std::move(source_range)),
      module_instance_info_(std::move(module_instance_info)) {}

InlinedCallStack::InlinedCallStack(
    InlinedCallStackPtr callee,
    Function* fn,
    SourceRange source_range)
    : callee_(std::move(callee)),
      fn_(fn),
      source_range_(std::move(source_range)) {}

InlinedCallStack::InlinedCallStack(
    InlinedCallStackPtr callee,
    Function* fn,
    SourceRange source_range,
    c10::optional<ModuleInstanceInfo> module_instance_info)
    : callee_(std::move(callee)),
      fn_(fn),
      source_range_(std::move(source_range)),
      module_instance_info_(std::move(module_instance_info)) {}

c10::optional<InlinedCallStackPtr> InlinedCallStack::callee() const {
  return callee_;
}

std::vector<InlinedCallStackEntry> InlinedCallStack::vec() {
  std::vector<InlinedCallStackEntry> r;
  c10::optional<InlinedCallStackPtr> current = intrusive_from_this();
  while (current) {
    r.emplace_back(std::make_tuple(
        (*current)->fn_,
        (*current)->source_range_,
        (*current)->module_instance_info_));
    current = (*current)->callee_;
  }
  return r;
}

ModuleInstanceInfo::ModuleInstanceInfo(
    c10::ClassTypePtr module_type,
    std::string instance_name)
    : module_type_(std::move(module_type)),
      instance_name_(std::move(instance_name)) {}
} // namespace jit
} // namespace torch
