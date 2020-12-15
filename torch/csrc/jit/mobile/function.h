#pragma once
#include <ATen/core/ivalue.h>
//#include <aten/src/Aten/core/operator_name.h>
#include <vector>

namespace torch {
namespace jit {
using Stack = std::vector<c10::IValue>;
enum OpCode : uint8_t;

namespace mobile {
struct Code;

class Function {
 public:
  Function(c10::QualifiedName name);
  bool run(Stack& stack) const;
  c10::IValue operator()(Stack& stack);
  const std::string& name() const;
  const c10::QualifiedName& qualname() const;
  void append_instruction(OpCode op, int X, int N);
  bool append_operator(
      const std::string& name,
      const std::string& overload_name,
      int64_t model_version);
  void set_module_debug_info_list_size(size_t size);
  void set_module_info(const std::string& module_info, size_t pc);
  void append_constant(const c10::IValue& constant);
  void append_type(const c10::TypePtr& type);

  void set_register_size(size_t size);

  std::string get_module_debug_info(size_t pc) const;

 private:
  c10::QualifiedName name_;
  std::shared_ptr<Code> code_;
  std::vector<std::string> pc_to_module_debug_info_;
};

} // namespace mobile
} // namespace jit
} // namespace torch
