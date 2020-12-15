#include <torch/csrc/jit/serialization/export.h>

#include <c10/util/Exception.h>
#include <torch/csrc/jit/ir/attributes.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/type_hashing.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/runtime/instruction.h>
#include <torch/csrc/jit/serialization/import_export_constants.h>
#include <torch/csrc/jit/serialization/import_export_helpers.h>
#include <torch/csrc/jit/serialization/pickle.h>
#include <torch/csrc/jit/serialization/python_print.h>
#include <torch/csrc/jit/serialization/source_range_serialization.h>
#include <torch/csrc/jit/serialization/type_name_uniquer.h>

#include <caffe2/serialize/inline_container.h>

#include <ATen/ATen.h>

#include <ATen/core/jit_type.h>
#include <ATen/core/qualified_name.h>
#include <string>
#include <vector>

namespace torch {
namespace jit {

char const* toString(OpCode op);

namespace {

ExportModuleExtraFilesHook& GetExtraFilesHook() {
  static ExportModuleExtraFilesHook func = nullptr;
  return func;
}

ExportModuleMobileInfoConverter& GetMobileInfoConverter() {
  static ExportModuleMobileInfoConverter func = nullptr;
  return func;
}

static IValue Tup(std::vector<IValue> ivalues) {
  return c10::ivalue::Tuple::create(std::move(ivalues));
}

static IValue Table(
    const std::vector<std::pair<std::string, IValue>>& entries) {
  std::vector<IValue> ivalue_entries;
  ivalue_entries.reserve(entries.size());
  for (const auto& e : entries) {
    ivalue_entries.push_back(Tup({e.first, e.second}));
  }
  return Tup(std::move(ivalue_entries));
}

std::string getModulePath(Node* node, const std::string& root_scope_string) {
  constexpr size_t kFunction = 0;
  constexpr size_t kModuleInstanceInfo = 2;

  if (!node->callstack()) {
    return root_scope_string + ".forward";
  } else {
    std::string module_info = root_scope_string;
    auto callstack_ptr = *(node->callstack());
    const auto& vec = callstack_ptr->vec();

    for (const auto& element : vec) {
      const auto& opt_module_instance_info =
          std::get<kModuleInstanceInfo>(element);
      if (opt_module_instance_info.has_value()) {
        const auto& module_instance_info = opt_module_instance_info.value();
        if (module_instance_info.class_type()) {
          const auto& class_type = module_instance_info.class_type();
          const auto& instance_name = module_instance_info.instance_name();
          auto type_name = class_type->name()->qualifiedName();
          type_name = type_name.substr(type_name.find_last_of('.') + 1);
          module_info.append(".")
              .append(instance_name)
              .append("(")
              .append(type_name)
              .append(")")
              .append(".")
              .append(std::get<kFunction>(element)->name());
        } else {
          module_info += ".(UNKNOWN_INSTANCE(UNKNOWN_TYPE)";
        }
      } else {
        module_info += ".(UNKNOWN_INSTANCE(UNKNOWN_TYPE)";
      }
    }

    return module_info;
  }
}

std::string getModuleTypeName(const Module& module, const std::string& prefix) {
  std::string moduleType = module.type()->str();
  size_t lastDotIndex = moduleType.rfind('.');
  if (lastDotIndex != std::string::npos) {
    moduleType = moduleType.substr(lastDotIndex + 1);
  }
  return prefix + "(" + moduleType + ")";
}

std::pair<IValue, c10::optional<IValue>> getFunctionTuple(
    const Module& module,
    const Function& func,
    bool save_mobile_debug_info) {
  auto graph = func.graph()->copy();

  Inline(*graph);

  torch::jit::Code code(graph, func.name());
  auto instructions_copy = code.instructions();

  // operator names
  std::vector<c10::OperatorName> opnames;
  std::vector<std::string> method_names;
  std::vector<std::string> op_module_paths;
  for (size_t i = 0; i < instructions_copy.size(); ++i) {
    Instruction ins = instructions_copy[i];
    if (ins.op == OP || ins.op == OPN) {
      auto node = code.instructions_source()[i];
      opnames.emplace_back(node->schema().operator_name());
      if (save_mobile_debug_info) {
        std::string root_scope_string = getModuleTypeName(module, "top");
        op_module_paths.emplace_back(getModulePath(node, root_scope_string));
      }
    }
    // CALL nodes at this point represent built-in (i.e. non-Graph)
    // functions that were not inlined. Here we convert the CALL
    // instructions for these functions into INTERFACE_CALL instructions
    // s.t. at runtime, we will look up the Function* on the Type of the
    // 0th argument in the stack and call that directly.
    if (ins.op == CALL) {
      auto node = code.instructions_source()[i];
      if (node->kind() == prim::CallMethod) {
        // NB: replacing instruction
        auto method_name_idx =
            code.constant_table().size() + method_names.size();
        method_names.emplace_back(node->s(attr::name));
        Instruction new_instr{INTERFACE_CALL,
                              static_cast<int32_t>(method_name_idx),
                              static_cast<uint16_t>(node->inputs().size())};
        instructions_copy[i] = new_instr;
      } else {
        TORCH_INTERNAL_ASSERT(
            false, "Unsupported node kind on CALL opcode for mobile");
      }
    } else if (ins.op == RET) {
      auto node = code.instructions_source()[i];
      for (const auto& input : node->inputs()) {
        const auto& input_type = input->type();
        if (input_type->kind() == TypeKind::TupleType) {
          if (const auto& name_typed_input =
                  input_type->cast<at::NamedType>()) {
            TORCH_CHECK(
                !name_typed_input->name(),
                "A named tuple type is not supported in mobile module. ",
                "Workaround: instead of using a named tuple type's fields, ",
                "use a dictionary type's key-value pair itmes or ",
                "a pytorch class (class Foo(torch.nn.Module))'s attributes.'");
          }
        } else if (
            input_type->kind() == TypeKind::ListType ||
            input_type->kind() == TypeKind::DictType) {
          for (const TypePtr& element_type : input_type->containedTypes()) {
            TORCH_CHECK(
                element_type->kind() != TypeKind::ClassType,
                "Returining a list or dictionary with pytorch class type ",
                "is not supported in mobile module "
                "(List[Foo] or Dict[int, Foo] for class Foo(torch.nn.Module)). "
                "Workaround: instead of using pytorch class as their element type, ",
                "use a combination of list, dictionary, and single types.");
          }
        }
      }
    } else {
      TORCH_CHECK(
          ins.op != CREATE_OBJECT,
          "CREATE_OBJECT is not supported in mobile module. ",
          "Workaround: instead of using arbitrary class type (class Foo()), ",
          "define a pytorch class (class Foo(torch.nn.Module)).");
      TORCH_CHECK(
          isOpSupportedInMobile(ins.op),
          toString(ins.op),
          " is not supported in mobile module.");
    }
  }

  // instructions
  std::vector<IValue> instructions;
  instructions.reserve(instructions_copy.size());
  for (Instruction ins : instructions_copy) {
    instructions.emplace_back(Tup({toString(ins.op), ins.X, ins.N}));
  }

  // operators
  std::vector<IValue> operators;
  operators.reserve(opnames.size());
  for (const auto& opname : opnames) {
    operators.emplace_back(Tup({opname.name, opname.overload_name}));
  }

  // constants
  //
  // Make a copy of the constants and append the method names
  // that we emitted for the converted INTERFACE_CALL nodes above.
  auto constants = code.constant_table();
  for (auto& method_name : method_names) {
    constants.emplace_back(std::move(method_name));
  }

  // types
  std::vector<IValue> types;
  types.reserve(code.type_table().size());
  for (const TypePtr& t : code.type_table()) {
    types.emplace_back(t->annotation_str());
  }

  // since the register location is embedded into the bytecode, pass the
  // register size
  auto register_size = static_cast<int>(code.register_size());

  auto table = Table({{"instructions", Tup(instructions)},
                      {"operators", Tup(operators)},
                      {"constants", Tup(constants)},
                      {"types", Tup(types)},
                      {"register_size", register_size}});
  auto bytecode_vals = Tup({func.qualname().qualifiedName(), table});

  c10::optional<IValue> debug_info_vals;
  if (save_mobile_debug_info) {
    // module debug info
    std::vector<IValue> module_paths;
    module_paths.reserve(op_module_paths.size());
    for (auto& path : op_module_paths) {
      module_paths.emplace_back(std::move(path));
    }
    auto module_debug_info = Table({{"module_debug_info", Tup(module_paths)}});
    debug_info_vals = Tup({func.qualname().qualifiedName(), module_debug_info});
  }
  return std::make_pair(bytecode_vals, debug_info_vals);
}

void setstateTuple(
    const Module& module,
    const IValue& ivalue,
    std::vector<c10::IValue>& elements,
    c10::optional<std::vector<c10::IValue>>& debug_info_elements,
    bool save_mobile_debug_info) {
  if (!ivalue.isObject())
    return;
  auto obj = ivalue.toObject();
  auto type = obj->type();
  if (checkHasValidSetGetState(type)) {
    Function& setstate = type->getMethod("__setstate__");
    if (setstate.isGraphFunction()) {
      auto func_tuple =
          getFunctionTuple(module, setstate, save_mobile_debug_info);
      elements.push_back(func_tuple.first);
      if (save_mobile_debug_info) {
        debug_info_elements->push_back(func_tuple.second.value());
      }
    }
  } else {
    for (size_t i = 0, n = type->numAttributes(); i < n; ++i) {
      setstateTuple(
          module,
          obj->getSlot(i),
          elements,
          debug_info_elements,
          save_mobile_debug_info);
    }
  }
}
} // namespace

void moduleMethodsTuple(
    const Module& module,
    std::vector<c10::IValue>& elements,
    c10::optional<std::vector<c10::IValue>>& debug_info_elements,
    bool save_mobile_debug_info) {
  auto methods = module.get_methods();
  // top level methods
  for (const auto& method : methods) {
    auto func_tuple =
        getFunctionTuple(module, method.function(), save_mobile_debug_info);
    elements.push_back(func_tuple.first);
    if (save_mobile_debug_info) {
      debug_info_elements->push_back(func_tuple.second.value());
    }
  }

  // __setstate__ of all components
  setstateTuple(
      module,
      module._ivalue(),
      elements,
      debug_info_elements,
      save_mobile_debug_info);
}

void SetExportModuleExtraFilesHook(ExportModuleExtraFilesHook hook) {
  GetExtraFilesHook() = std::move(hook);
}

void SetExportModuleMobileInfoConverter(
    ExportModuleMobileInfoConverter converter) {
  GetMobileInfoConverter() = std::move(converter);
}

class ScriptModuleSerializer {
 public:
  explicit ScriptModuleSerializer(const std::string& filename)
      : writer_(filename) {}

  explicit ScriptModuleSerializer(
      const std::function<size_t(const void*, size_t)>& writer_func)
      : writer_(writer_func) {}

  void serialize(
      const Module& module,
      const ExtraFilesMap& extra_files,
      bool bytecode_format,
      bool save_mobile_debug_info) {
    C10_LOG_API_USAGE_ONCE("torch.script.save");
    writeExtraFiles(module, extra_files);
    // Serialize the model object
    writeArchive("data", module._ivalue());
    // Then we serialize all code info.
    writeCode(module.type());
    // The tensor constants from the code are written to a separate archive
    // so loading the code does not depend on loading the data
    std::vector<IValue> ivalue_constants(
        constant_table_.begin(), constant_table_.end());
    writeArchive("constants", c10::ivalue::Tuple::create(ivalue_constants));
    if (bytecode_format) {
      writeByteCode(module, save_mobile_debug_info);
      writeMobileMetadata(module, extra_files);
    }

    // Acquires and sets minimum (dynamic) version
    for (auto& item : file_streams_) {
      writer_.setMinVersion(item.value().minVersion());
    }
  }

 private:
  void writeArchive(const std::string& archive_name, const IValue& value) {
    std::vector<char> data;
    // Vector to capture the run-time class types during pickling the IValues
    std::vector<c10::ClassTypePtr> memorizedClassTypes;
    Pickler data_pickle(
        [&](const char* buf, size_t size) {
          data.insert(data.end(), buf, buf + size);
        },
        nullptr,
        [&](const c10::ClassTypePtr& t) {
          return type_name_uniquer_.getUniqueName(t);
        },
        &memorizedClassTypes);
    data_pickle.protocol();
    data_pickle.pushIValue(value);
    data_pickle.stop();
    size_t i = 0;
    std::string prefix = archive_name + "/";
    for (const auto& td : data_pickle.tensorData()) {
      WriteableTensorData writable_td = getWriteableTensorData(td);
      std::string fname = prefix + c10::to_string(i++);
      writer_.writeRecord(fname, writable_td.data(), writable_td.sizeInBytes());
    }
    std::string fname = archive_name + ".pkl";
    writer_.writeRecord(fname, data.data(), data.size());

    // serialize all the captured run-time class types
    for (const c10::ClassTypePtr& wroteType : memorizedClassTypes) {
      convertNamedType(wroteType);
    }
  }

  void writeExtraFiles(const Module& module, const ExtraFilesMap& extra_files) {
    // Write out extra files.
    for (const auto& kv : extra_files) {
      const std::string key = "extra/" + kv.first;
      writer_.writeRecord(key, kv.second.data(), kv.second.size());
    }
    auto hook = GetExtraFilesHook();
    if (hook) {
      ExtraFilesMap hook_files = hook(module);
      for (const auto& kv : hook_files) {
        // Checks if the hooked file is already written in extra files,
        //   if so, skips it and warns
        if (extra_files.find(kv.first) != extra_files.end()) {
          TORCH_WARN_ONCE(
              "An extra files hook attempted to write ",
              kv.first,
              " but ",
              "this is already written in extra files and so will be skipped. ",
              "This warning will only appear once per process.");
          continue;
        }
        const std::string key = "extra/" + kv.first;
        writer_.writeRecord(key, kv.second.data(), kv.second.size());
      }
    }
  }

  void writeMobileMetadata(
      const Module& module,
      const ExtraFilesMap& extra_files) {
    auto hook = GetExtraFilesHook();
    auto converter = GetMobileInfoConverter();
    if (!converter) {
      return;
    }
    ExtraFilesMap files_to_write = extra_files;
    // merge hook files and extra files
    if (hook) {
      ExtraFilesMap hook_files = hook(module);
      files_to_write.insert(hook_files.begin(), hook_files.end());
    }
    auto content_to_write = converter(module, files_to_write);
    if (!content_to_write.empty()) {
      writeArchive("metadata", content_to_write);
    }
  }

  void writeCode(const at::NamedTypePtr& root_type) {
    class_deps_.add(root_type);
    for (size_t i = 0; i < class_deps_.size(); ++i) {
      // note: convertNameType may extend class_deps_, so re-checking
      // .size() is necessary
      convertNamedType(class_deps_[i]);
    }

    // Mapping of filename => src. We need this because multiple classes may go
    // in the same file (e.g. foo.bar.Baz and foo.bar.Qux)
    for (auto& item : file_streams_) {
      const std::string filename = qualifierToArchivePath(item.key(), "code/");

      std::string src = item.value().str();

      // Only compress these records if they're not tiny.
      // The cpu cost of generating zip datastructs and compressing isn't
      // well-spent for very small records.
      static constexpr size_t kMinToCompress = 200;

      writer_.writeRecord(
          filename,
          src.c_str(),
          src.size(),
          src.size() > kMinToCompress /*compress*/);

      // Write out the debug information
      std::string debugFilename = filename + ".debug_pkl";
      SourceRangePickler source_range_pickler;
      auto range_data = source_range_pickler.pickle(item.value().ranges());
      writer_.writeRecord(
          debugFilename,
          range_data.data(),
          range_data.size(),
          range_data.size() > kMinToCompress /*compress*/);
    }
  }

  void writeByteCode(const Module& module, bool save_mobile_debug_info) {
    std::vector<c10::IValue> elements;
    elements.emplace_back(
        static_cast<int64_t>(caffe2::serialize::kProducedBytecodeVersion));
    c10::optional<std::vector<c10::IValue>> debug_info_elements;
    if (save_mobile_debug_info) {
      debug_info_elements = std::vector<c10::IValue>();
      debug_info_elements->emplace_back(
          static_cast<int64_t>(caffe2::serialize::kProducedBytecodeVersion));
    }

    moduleMethodsTuple(
        module, elements, debug_info_elements, save_mobile_debug_info);
    auto telements = Tup(std::move(elements));
    writeArchive("bytecode", telements);
    if (save_mobile_debug_info) {
      auto debug_info_telements = Tup(std::move(debug_info_elements.value()));
      writeArchive("mobile_debug", debug_info_telements);
    }
  }

  void convertNamedType(const c10::NamedTypePtr& class_type) {
    if (converted_types_.count(class_type)) {
      return;
    }
    converted_types_.insert(class_type);
    auto qualname = type_name_uniquer_.getUniqueName(class_type);
    std::string qualifier = qualname.prefix();
    PythonPrint* pp = file_streams_.find(qualifier);

    auto type_printer =
        [&](const c10::ConstTypePtr& t) -> c10::optional<std::string> {
      auto namedType = t->cast<c10::NamedType>();
      if (namedType && namedType->name()) {
        return type_name_uniquer_.getUniqueName(namedType).qualifiedName();
      }
      return c10::nullopt;
    };
    if (!pp) {
      pp = &file_streams_.insert(
          std::move(qualifier),
          PythonPrint(
              constant_table_,
              class_deps_,
              type_printer,
              /*enforce_importable=*/true));
    }
    pp->printNamedType(class_type);
  }

  caffe2::serialize::PyTorchStreamWriter writer_;
  std::vector<at::IValue> constant_table_;
  std::unordered_set<c10::NamedTypePtr> converted_types_;
  PrintDepsTable class_deps_;
  TypeNameUniquer type_name_uniquer_;

  // qualifier, e.g. '__torch__.Bar' -> PythonPrint for the file that will be
  // created
  OrderedDict<std::string, PythonPrint> file_streams_;
};

void ExportModule(
    const Module& module,
    std::ostream& out,
    const ExtraFilesMap& extra_files,
    bool bytecode_format,
    bool save_mobile_debug_info) {
  ScriptModuleSerializer serializer(
      [&](const void* buf, size_t nbytes) -> size_t {
        out.write(static_cast<const char*>(buf), nbytes);
        return !out ? 0 : nbytes;
      });
  serializer.serialize(
      module, extra_files, bytecode_format, save_mobile_debug_info);
}

void ExportModule(
    const Module& module,
    const std::string& filename,
    const ExtraFilesMap& extra_files,
    bool bytecode_format,
    bool save_mobile_debug_info) {
  ScriptModuleSerializer serializer(filename);
  serializer.serialize(
      module, extra_files, bytecode_format, save_mobile_debug_info);
}

void ExportModule(
    const Module& module,
    const std::function<size_t(const void*, size_t)>& writer_func,
    const ExtraFilesMap& extra_files,
    bool bytecode_format,
    bool save_mobile_debug_info) {
  ScriptModuleSerializer serializer(writer_func);
  serializer.serialize(
      module, extra_files, bytecode_format, save_mobile_debug_info);
}

namespace {
void export_opnames(const script::Module& m, std::set<std::string>& opnames) {
  std::vector<c10::IValue> elements;
  c10::optional<std::vector<c10::IValue>> debug_info_elements;
  moduleMethodsTuple(
      m, elements, debug_info_elements, false /* save_mobile_debug_info */);
  for (const auto& element : elements) {
    auto table = element.toTuple()->elements()[1];
    auto row =
        table.toTuple()->elements().at(BYTECODE_INDEX_OPERATOR).toTuple();
    TORCH_INTERNAL_ASSERT(
        row->elements().at(0).toStringRef() == "operators",
        "Expected operators but found ",
        row->elements().at(0).toStringRef());
    const auto& ops_list = row->elements().at(1).toTuple()->elements();
    for (const auto& op : ops_list) {
      auto op_item = op.toTuple()->elements();
      TORCH_CHECK(
          op_item.size() == 2,
          "There should be two parts in an operator name.");
      auto opname = op_item[0].toString()->string();
      auto overload = op_item[1].toString()->string();
      opnames.emplace(overload.empty() ? opname : opname + "." + overload);
    }
  }
}
} // namespace

std::vector<std::string> export_opnames(const script::Module& m) {
  std::set<std::string> names;
  export_opnames(m, names);
  return std::vector<std::string>(names.begin(), names.end());
}

} // namespace jit
} // namespace torch
