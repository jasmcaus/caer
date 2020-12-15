#include <torch/csrc/autograd/python_cpp_function.h>

#include <torch/csrc/python_headers.h>
#include <memory>
#include <cstdio>
#include <typeindex>
#include <unordered_map>

#include <torch/csrc/autograd/python_function.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/autograd/python_hook.h>
#include <torch/csrc/autograd/python_anomaly_mode.h>
#include <pybind11/pybind11.h>
#include <torch/csrc/utils/python_strings.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/Exceptions.h>

using namespace torch::autograd;

namespace torch { namespace autograd {

namespace {

PyObject* THPCppFunction_call(PyObject* self, PyObject* args, PyObject *kwargs)
{
  if (kwargs && PyDict_Size(kwargs) != 0) {
    return PyErr_Format(PyExc_TypeError, "keyword arguments are not supported");
  }

  int num_inputs = PyTuple_GET_SIZE(args);
  int num_inputs_required = ((THPCppFunction*)self)->cdata->num_inputs();
  if (num_inputs != num_inputs_required) {
    return PyErr_Format(PyExc_TypeError, "expected %d arguments, got %d instead",
                        num_inputs_required, num_inputs);
  }
  variable_list vars(num_inputs);
  for (int i = 0; i != num_inputs; ++i) {
    PyObject* arg = PyTuple_GET_ITEM(args, i);
    if (arg == Py_None) {
      continue;
    }
    if (!THPVariable_Check(arg)) {
      return PyErr_Format(PyExc_TypeError, "argument %d is not a Variable", i);
    }
    vars[i] = ((THPVariable*)arg)->cdata;
  }

  variable_list output;

  HANDLE_TH_ERRORS {
    pybind11::gil_scoped_release nogil;
    output = (*((THPCppFunction*)self)->cdata)(std::move(vars));
  }
  END_HANDLE_TH_ERRORS

  int num_outputs = output.size();
  if (num_outputs == 1) {
    // assume we want to unpack one element tuples for now
    return THPVariable_Wrap(output[0]);
  }

  THPObjectPtr tuple(PyTuple_New(num_outputs));
  for (int i = 0; i != num_outputs; ++i) {
    PyTuple_SET_ITEM(tuple.get(), i, THPVariable_Wrap(output[i]));
  }
  return tuple.release();
}

int THPCppFunction_traverse(PyObject* self, visitproc visit, void *arg)
{
  auto& fn = *((THPCppFunction*)self)->cdata;
  for (const auto& hook : fn.pre_hooks()) {
    if (auto pyhook = dynamic_cast<PyFunctionPreHook*>(hook.get())) {
      Py_VISIT(pyhook->dict);
    }
  }
  for (const auto& hook : fn.post_hooks()) {
    if (auto pyhook = dynamic_cast<PyFunctionPostHook*>(hook.get())) {
      Py_VISIT(pyhook->dict);
    }
  }
  return 0;
}

int THPCppFunction_clear(PyObject* self)
{
  auto f = (THPCppFunction*)self;
  // Remove the weak ref of the c++ object if it exist
  if (f->cdata) {
    f->cdata->set_pyobj(nullptr);
  }
  f->cdata.reset();
  return 0;
}

void THPCppFunction_dealloc(PyObject* self)
{
  THPCppFunction_clear(self);
  ((THPCppFunction*)self)->cdata.~shared_ptr();
  Py_TYPE(self)->tp_free(self);
}

} // namespace

PyObject* THPCppFunction_next_functions(THPCppFunction* self, PyObject* hook)
{
  const auto num_next = self->cdata->num_outputs();
  THPObjectPtr py_functions(PyTuple_New(num_next));
  if (!py_functions) return nullptr;
  for (size_t i = 0; i < num_next; ++i) {
    auto& c_tuple = self->cdata->next_edge(i);
    THPObjectPtr tuple(PyTuple_New(2));
    if (!tuple) return nullptr;
    PyObject *py_fn = functionToPyObject(c_tuple.function);
    if (!py_fn) return nullptr;
    PyTuple_SET_ITEM(tuple.get(), 0, py_fn);
    PyObject *py_idx = PyLong_FromLong(c_tuple.input_nr);
    if (!py_idx) return nullptr;
    PyTuple_SET_ITEM(tuple.get(), 1, py_idx);
    PyTuple_SET_ITEM(py_functions.get(), i, tuple.release());
  }
  return py_functions.release();
}

PyObject* THPCppFunction_metadata(THPCppFunction *self, void *_unused)
{
  auto metadata = static_cast<PyAnomalyMetadata*>(self->cdata->metadata())->dict();

  Py_INCREF(metadata);
  return metadata;
}

PyObject* THPCppFunction_requires_grad(THPCppFunction* self, void *unused) {
  Py_RETURN_TRUE;
}

PyObject* THPCppFunction_register_hook_dict(PyObject* self, PyObject* _var)
{
  if (!THPVariable_Check(_var)) {
    return PyErr_Format(PyExc_TypeError, "_register_hook_dict expected a variable");
  }
  auto var = (THPVariable*)_var;
  auto& fn = *((THPCppFunction*)self)->cdata;
  std::unique_ptr<FunctionPreHook> hook(
      new PyFunctionPreHook(var->backward_hooks, var->cdata.output_nr()));
  fn.add_pre_hook(std::move(hook));
  Py_RETURN_NONE;
}

PyObject* THPCppFunction_register_hook(PyObject* self, PyObject* hook)
{
  auto& fn = *((THPCppFunction*)self)->cdata;
  return registerFunctionHook(fn, hook);
}

PyObject* THPCppFunction_name(PyObject* self, PyObject *noargs) {
  auto& fn = *((THPCppFunction*)self)->cdata;
  return THPUtils_packString(fn.name());
}

static struct PyMethodDef default_methods[] = {
  THP_FUNCTION_DEFAULT_METHODS,
  {nullptr}
};

static struct PyGetSetDef default_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {nullptr}
};

PyTypeObject* _initFunctionPyTypeObject(PyTypeObject& type, const char* name,
  PyGetSetDef* function_properties, PyMethodDef* function_methods)
{
  type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC;
  type.tp_name = name;
  type.tp_basicsize = sizeof(THPCppFunction);
  type.tp_call = THPCppFunction_call;
  type.tp_methods = function_methods ? function_methods : default_methods;
  type.tp_getset = function_properties ? function_properties : default_properties;
  type.tp_dealloc = THPCppFunction_dealloc;
  type.tp_traverse = THPCppFunction_traverse;
  type.tp_clear = THPCppFunction_clear;
  if (PyType_Ready(&type) < 0) {
    auto msg = std::string("Unable to instantiate PyTypeObject for ") + name;
    throw std::runtime_error(msg);
  }
  return &type;
}

static std::unordered_map<std::type_index, THPObjectPtr> cpp_function_types;

struct DefaultFunctionType {
  DefaultFunctionType() : type() {
    _initFunctionPyTypeObject(type, "CppFunction", nullptr, nullptr);
    Py_INCREF(&type);
  }

  PyTypeObject type;
};

PyObject* functionToPyObject(const std::shared_ptr<Node>& cdata)
{
  static DefaultFunctionType default_type;

  if (!cdata) {
    Py_RETURN_NONE;
  }

  if (auto pfw = dynamic_cast<PyNode*>(cdata.get())) {
    PyObject* obj = pfw->obj;
    Py_INCREF(obj);
    return obj;
  }

  if (cdata->pyobj()) {
    Py_INCREF(cdata->pyobj());
  } else {
    auto& fn = *cdata;
    auto it = cpp_function_types.find(std::type_index(typeid(fn)));
    PyTypeObject* type;
    if (it == cpp_function_types.end()) {
      type = &default_type.type;
    } else {
      type = (PyTypeObject*)it->second.get();
    }

    THPObjectPtr obj(type->tp_alloc(type, 0));
    if (!obj) return nullptr;
    THPCppFunction* f = (THPCppFunction*)obj.get();
    new (&f->cdata) std::shared_ptr<Node>(cdata);

    // No INCREF here as we only have a weak reference
    cdata->set_pyobj(obj.release());
  }

  return cdata->pyobj();
}

void registerCppFunction(const std::type_info& type, PyTypeObject* pytype)
{
  Py_INCREF((PyObject*)pytype);
  cpp_function_types[std::type_index(type)] = THPObjectPtr((PyObject*)pytype);
}

PyObject* registerFunctionHook(Node& fn, PyObject* hook)
{
  PyObject* dict = Py_None;
  for (const auto& hook : fn.post_hooks()) {
    if (auto pyhook = dynamic_cast<PyFunctionPostHook*>(hook.get())) {
      dict = pyhook->dict;
      break;
    }
  }

  THPObjectPtr register_fn(PyObject_GetAttrString(THPFunctionClass, "_register_hook"));
  if (!register_fn) return nullptr;
  THPObjectPtr res(PyObject_CallFunctionObjArgs(register_fn.get(), dict, hook, nullptr));
  if (!res) return nullptr;

  if (dict == Py_None) {
    dict = PyTuple_GET_ITEM(res.get(), 0);
    std::unique_ptr<FunctionPostHook> hook(new PyFunctionPostHook(dict));
    fn.add_post_hook(std::move(hook));
  }

  PyObject* handle = PyTuple_GET_ITEM(res.get(), 1);
  Py_INCREF(handle);
  return handle;
}

}} // namespace torch::autograd
