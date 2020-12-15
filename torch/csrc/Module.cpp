#include <torch/csrc/python_headers.h>
#include <sys/types.h>

#ifndef _MSC_VER
#include <sys/socket.h>
#endif

#include <unordered_map>
#include <cstdlib>
#include <libshm.h>
#include <TH/TH.h>
#include <c10/util/Logging.h>
#include <ATen/ATen.h>
#include <ATen/ExpandUtils.h>
#include <ATen/dlpack.h>
#include <ATen/DLConvertor.h>
#include <ATen/Parallel.h>
#include <ATen/Utils.h>
#include <ATen/VmapMode.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <torch/csrc/THP.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/Device.h>
#include <torch/csrc/Stream.h>
#include <torch/csrc/Dtype.h>
#include <torch/csrc/DataLoader.h>
#include <torch/csrc/Generator.h>
#include <torch/csrc/Layout.h>
#include <torch/csrc/MemoryFormat.h>
#include <torch/csrc/QScheme.h>
#include <torch/csrc/TypeInfo.h>
#include <torch/csrc/autograd/python_nn_functions.h>
#include <torch/csrc/autograd/python_fft_functions.h>
#include <torch/csrc/autograd/python_linalg_functions.h>
#include <torch/csrc/autograd/python_legacy_variable.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/multiprocessing/init.h>
#include <torch/csrc/tensor/python_tensor.h>
#include <torch/csrc/utils/disable_torch_function.h>
#include <torch/csrc/utils/tensor_dtypes.h>
#include <torch/csrc/utils/python_strings.h>
#include <torch/csrc/utils/tensor_layouts.h>
#include <torch/csrc/utils/tensor_memoryformats.h>
#include <torch/csrc/utils/tensor_qschemes.h>
#include <torch/csrc/utils/tensor_numpy.h>
#include <torch/csrc/utils/python_dispatch.h>
#include <torch/csrc/jit/python/python_tracer.h>
#include <torch/csrc/jit/python/init.h>
#include <torch/csrc/jit/python/python_ir.h>
#include <torch/csrc/onnx/init.h>
#include <torch/csrc/utils/init.h>
#include <torch/csrc/api/include/torch/python/init.h>

#ifdef USE_DISTRIBUTED
#ifdef USE_C10D
#include <torch/csrc/distributed/autograd/python_autograd.h>
#include <torch/csrc/distributed/c10d/c10d.h>
#include <torch/csrc/distributed/rpc/rpc.h>
#include <torch/csrc/distributed/rpc/testing/testing.h>
#endif
#endif

#if defined(USE_VALGRIND)
#include <callgrind.h>
#endif

#define WITH_NUMPY_IMPORT_ARRAY
#include <torch/csrc/utils/numpy_stub.h>

namespace py = pybind11;

PyObject* module;

THPGenerator *THPDefaultCPUGenerator = nullptr;

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

static PyObject * THPModule_initNames(PyObject *self, PyObject *arg)
{
  static std::vector<std::string> names;

  THPObjectPtr types(PySequence_Fast(arg, "expected a sequence"));
  if (!types) return nullptr;

  auto num_classes = PySequence_Fast_GET_SIZE(types.get());
  names.reserve(names.size() + num_classes);
  for (Py_ssize_t i = 0; i < num_classes; i++) {
    PyObject* obj = PySequence_Fast_GET_ITEM(types.get(), i);
    THPUtils_assert(PyType_Check(obj), "expected a PyTypeObject");
    PyTypeObject* type = (PyTypeObject*)obj;

    THPObjectPtr module_name(PyObject_GetAttrString(obj, "__module__"));
    if (!module_name) return nullptr;
    THPUtils_assert(THPUtils_checkString(module_name.get()),
        "expected __module__ to be a string");
    std::string name = THPUtils_unpackString(module_name.get());
    names.push_back(name + "." + type->tp_name);
    type->tp_name = names.back().c_str();
  }
  Py_RETURN_NONE;
}
//
// Callback for python part. Used for additional initialization of python classes
static PyObject * THPModule_initExtension(PyObject *_unused, PyObject *shm_manager_path)
{
  HANDLE_TH_ERRORS
  if (!THPUtils_checkString(shm_manager_path)) {
    THPUtils_setError("initialization error - expected bytes/string object as shm_manager_path!");
    return nullptr;
  }
  torch::utils::initializeLayouts();
  torch::utils::initializeMemoryFormats();
  torch::utils::initializeQSchemes();
  torch::utils::initializeDtypes();
  torch::tensors::initialize_python_bindings();
  std::string path = THPUtils_unpackString(shm_manager_path);
  libshm_init(path.c_str());

  auto module = THPObjectPtr(PyImport_ImportModule("torch"));
  if (!module) throw python_error();

  THPDoubleStorage_postInit(module);
  THPFloatStorage_postInit(module);
  THPHalfStorage_postInit(module);
  THPLongStorage_postInit(module);
  THPIntStorage_postInit(module);
  THPShortStorage_postInit(module);
  THPCharStorage_postInit(module);
  THPByteStorage_postInit(module);
  THPBoolStorage_postInit(module);
  THPQUInt8Storage_postInit(module);
  THPQUInt4x2Storage_postInit(module);
  THPQInt8Storage_postInit(module);
  THPQInt32Storage_postInit(module);
  THPBFloat16Storage_postInit(module);
  THPComplexDoubleStorage_postInit(module);
  THPComplexFloatStorage_postInit(module);
  THPAutograd_initFunctions();
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// The idea behind these two functions is to make it easy to test if we are
// built with ASAN: they're designed not to crash if ASAN is not enabled, but
// to trigger ASAN if it is enabled.  This lets us run a "canary" tests which
// checks if our build environment is misconfigured.

static PyObject * THPModule_crashIfCsrcASAN(PyObject *module, PyObject *arg) {
  THPUtils_assert(THPUtils_checkLong(arg), "crash_if_csrc_asan expects an int, "
          "but got %s", THPUtils_typename(arg));
  //NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays, modernize-avoid-c-arrays)
  volatile char x[3];
  x[static_cast<int>(THPUtils_unpackLong(arg))] = 0;
  return PyLong_FromLong(x[0]);
}

static PyObject * THPModule_crashIfCsrcUBSAN(PyObject *module, PyObject *arg) {
  THPUtils_assert(THPUtils_checkLong(arg), "crash_if_csrc_ubsan expects an int, "
          "but got %s", THPUtils_typename(arg));
  int32_t x = static_cast<int>(THPUtils_unpackLong(arg));
  double y = 1.0 / x;
  return PyLong_FromLong((int)y);
}

static PyObject * THPModule_crashIfATenASAN(PyObject *module, PyObject *arg) {
  THPUtils_assert(THPUtils_checkLong(arg), "crash_if_aten_asan expects an int, "
          "but got %s", THPUtils_typename(arg));
  return PyLong_FromLong(at::_crash_if_asan(static_cast<int>(THPUtils_unpackLong(arg))));
}

static PyObject * THPModule_getNumThreads(PyObject *module, PyObject *noargs)
{
  return PyLong_FromLong(at::get_num_threads());
}

static PyObject * THPModule_setNumThreads(PyObject *module, PyObject *arg)
{
  THPUtils_assert(THPUtils_checkLong(arg), "set_num_threads expects an int, "
          "but got %s", THPUtils_typename(arg));
  int nthreads = (int)THPUtils_unpackLong(arg);
  THPUtils_assert(nthreads > 0, "set_num_threads expects a positive integer");
  at::set_num_threads(nthreads);
  Py_RETURN_NONE;
}

static PyObject * THPModule_getNumInteropThreads(PyObject *module, PyObject *noargs)
{
  return PyLong_FromLong(at::get_num_interop_threads());
}

static PyObject * THPModule_setNumInteropThreads(PyObject *module, PyObject *arg)
{
  THPUtils_assert(THPUtils_checkLong(arg), "set_num_interop_threads expects an int, "
          "but got %s", THPUtils_typename(arg));
  int nthreads = (int)THPUtils_unpackLong(arg);
  THPUtils_assert(nthreads > 0, "set_num_interop_threads expects a positive integer");
  at::set_num_interop_threads(nthreads);
  Py_RETURN_NONE;
}

PyObject * THPModule_setDefaultTensorType(PyObject *_unused, PyObject *type)
{
  HANDLE_TH_ERRORS
  torch::tensors::py_set_default_tensor_type(type);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject * THPModule_setDefaultDtype(PyObject *_unused, PyObject *dtype)
{
  HANDLE_TH_ERRORS
  torch::tensors::py_set_default_dtype(dtype);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject *THPModule_addDocStr(PyObject *_unused, PyObject *args)
{
  // adds a __doc__ string to a function, similar to numpy's arr_add_docstring
  static std::vector<std::string> all_docs;
  PyObject *obj;
  PyObject *doc_obj;
  if (!PyArg_ParseTuple(args, "OO", &obj, &doc_obj)) {
    return nullptr;
  }

  const char* doc_str = "<invalid string>";
  if (THPUtils_checkString(doc_obj)) {
    all_docs.push_back(THPUtils_unpackString(doc_obj));
    doc_str = all_docs.back().c_str();
  }

  if (Py_TYPE(obj) == &PyCFunction_Type) {
    PyCFunctionObject* f = (PyCFunctionObject *)obj;
    if (f->m_ml->ml_doc) {
      return PyErr_Format(PyExc_RuntimeError,
          "function '%s' already has a docstring", f->m_ml->ml_name);
    }
    f->m_ml->ml_doc = doc_str;
  } else if (strcmp(Py_TYPE(obj)->tp_name, "method_descriptor") == 0) {
    PyMethodDescrObject* m = (PyMethodDescrObject *)obj;
    if (m->d_method->ml_doc) {
      return PyErr_Format(PyExc_RuntimeError,
          "method '%s' already has a docstring", m->d_method->ml_name);
    }
    m->d_method->ml_doc = doc_str;
  } else if (strcmp(Py_TYPE(obj)->tp_name, "getset_descriptor") == 0) {
    //NOLINTNEXTLINE(cppcoreguidelines-pro-type-cstyle-cast)
    PyGetSetDescrObject* m = (PyGetSetDescrObject *)obj;
    if (m->d_getset->doc) {
      //NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg)
      return PyErr_Format(PyExc_RuntimeError,
          "attribute '%s' already has a docstring", m->d_getset->name);
    }
    // This field is not const for python < 3.7 yet the content is
    // never modified.
    //NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    m->d_getset->doc = const_cast<char *>(doc_str);
  } else if (Py_TYPE(obj) == &PyType_Type) {
    PyTypeObject* t = (PyTypeObject *)obj;
    if (t->tp_doc) {
      return PyErr_Format(PyExc_RuntimeError,
          "Type '%s' already has a docstring", t->tp_name);
    }
    t->tp_doc = doc_str;
  } else {
    return PyErr_Format(PyExc_TypeError,
        "don't know how to add docstring to type '%s'", Py_TYPE(obj)->tp_name);
  }

  Py_INCREF(obj);
  return obj;
}


PyObject *THPModule_inferSize(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  Py_ssize_t num_args = args ? (Py_ssize_t) PyTuple_Size(args) : 0;
  THPUtils_assert(num_args == 2, "expected exactly 2 arguments");
  PyObject *arg1 = PyTuple_GET_ITEM(args, 0);
  THPUtils_assert(THPSize_Check(arg1), "expected a torch.Size as argument 1");
  PyObject *arg2 = PyTuple_GET_ITEM(args, 1);
  THPUtils_assert(THPSize_Check(arg2), "expected a torch.Size as argument 2");

  auto size1 = THPUtils_unpackLongs(arg1);
  auto size2 = THPUtils_unpackLongs(arg2);
  auto sizes = at::infer_size(size1, size2);
  return THPSize_NewFromSizes(sizes.size(), sizes.data());
  END_HANDLE_TH_ERRORS
}

static PyObject *THPModule_setBackcompatBroadcastWarn(PyObject *module, PyObject *arg) {
  THPUtils_assert(PyBool_Check(arg), "set_backcompat_broadcast_warn expects a bool, "
          "but got %s", THPUtils_typename(arg));
  setBackCompatBroadcastWarn(arg == Py_True);
  Py_RETURN_NONE;
}

static PyObject *THPModule_getBackcompatBroadcastWarn(PyObject *module, PyObject *noargs)
{
  if (getBackCompatBroadcastWarn()) Py_RETURN_TRUE;
  else Py_RETURN_FALSE;
}

static PyObject *THPModule_setBackcompatKeepdimWarn(PyObject *module, PyObject *arg) {
  THPUtils_assert(PyBool_Check(arg), "set_backcompat_keepdim_warn expects a bool, "
          "but got %s", THPUtils_typename(arg));
  setBackCompatKeepdimWarn(arg == Py_True);
  Py_RETURN_NONE;
}

static PyObject *THPModule_getBackcompatKeepdimWarn(PyObject *module, PyObject *noargs)
{
  if (getBackCompatKeepdimWarn()) Py_RETURN_TRUE;
  else Py_RETURN_FALSE;
}

PyObject *THPModule_hasDistributed(PyObject *_unused, PyObject *noargs)
{
#ifdef USE_DISTRIBUTED
  Py_RETURN_TRUE;
#else
  Py_RETURN_FALSE;
#endif
}

static PyObject *THPModule_showConfig(PyObject *module, PyObject *noargs)
{
  HANDLE_TH_ERRORS
  return THPUtils_packString(at::show_config());
  END_HANDLE_TH_ERRORS
}

static PyObject *THPModule_cxxFlags(PyObject *module, PyObject *noargs)
{
  HANDLE_TH_ERRORS
  return THPUtils_packString(at::get_cxx_flags());
  END_HANDLE_TH_ERRORS
}

static PyObject *THPModule_parallelInfo(PyObject *module, PyObject *noargs)
{
  HANDLE_TH_ERRORS
  return THPUtils_packString(at::get_parallel_info());
  END_HANDLE_TH_ERRORS
}

void DLPack_Capsule_Destructor(PyObject* data) {
  HANDLE_TH_ERRORS
  DLManagedTensor * dlMTensor = (DLManagedTensor *)PyCapsule_GetPointer(data, "dltensor");
  if (dlMTensor) {
    // the dlMTensor has not been consumed, call deleter ourselves
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    dlMTensor->deleter(const_cast<DLManagedTensor*>(dlMTensor));
  } else {
    // the dlMTensor has been consumed
    // PyCapsule_GetPointer has set an error indicator
    PyErr_Clear();
  }
  END_HANDLE_TH_ERRORS_RET()
}

PyObject *THPModule_toDLPack(PyObject *_unused, PyObject *data)
{
  HANDLE_TH_ERRORS
  THPUtils_assert(THPVariable_Check(data), "data must be a Tensor");
  DLManagedTensor* dlMTensor = at::toDLPack(THPVariable_Unpack(data));
  return PyCapsule_New(dlMTensor, "dltensor", DLPack_Capsule_Destructor);
  END_HANDLE_TH_ERRORS
}

PyObject *THPModule_fromDLPack(PyObject *_unused, PyObject *data)
{
  using namespace torch::autograd;
  HANDLE_TH_ERRORS
  DLManagedTensor * dlMTensor = (DLManagedTensor *)PyCapsule_GetPointer(data, "dltensor");
  THPUtils_assert(dlMTensor, "from_dlpack received an invalid capsule. "
    "Note that DLTensor capsules can be consumed only once, "
    "so you might have already constructed a tensor from it once.")
  // atensor steals the ownership of the underlying storage. It also passes a
  // destructor function that will be called when the underlying storage goes
  // out of scope. When the destructor is called, the dlMTensor is destructed too.
  auto atensor = at::fromDLPack(dlMTensor);

  // It is possible that the call to at::fromDLPack is the very first
  // call to create a Tensor in PyTorch. If so, then _lazy_init has
  // not been called, and the attempt to call createPyObject will fail
  // because cuda ATen types have not been registered in Python yet.
  // so if we have a cuda tensor, then we need to make sure
  // we have called _lazy_init here
  if(atensor.is_cuda()) {
    py::module::import("torch.cuda").attr("init")();
  }
  // Make sure this capsule will never be used again.
  PyCapsule_SetName(data, "used_dltensor");
  return THPVariable_Wrap(std::move(atensor));
  END_HANDLE_TH_ERRORS
}

PyObject *THPModule_setAllowTF32CuDNN(PyObject *_unused, PyObject *arg)
{
  THPUtils_assert(PyBool_Check(arg), "set_allow_tf32_cublas expects a bool, "
          "but got %s", THPUtils_typename(arg));
  at::globalContext().setAllowTF32CuDNN(arg == Py_True);
  Py_RETURN_NONE;
}

PyObject *THPModule_allowTF32CuDNN(PyObject *_unused, PyObject *noargs)
{
  if (at::globalContext().allowTF32CuDNN()) Py_RETURN_TRUE;
  else Py_RETURN_FALSE;
}

PyObject *THPModule_setUserEnabledCuDNN(PyObject *_unused, PyObject *arg)
{
  THPUtils_assert(PyBool_Check(arg), "set_enabled_cudnn expects a bool, "
          "but got %s", THPUtils_typename(arg));
  at::globalContext().setUserEnabledCuDNN(arg == Py_True);
  Py_RETURN_NONE;
}

PyObject *THPModule_userEnabledCuDNN(PyObject *_unused, PyObject *noargs)
{
  if (at::globalContext().userEnabledCuDNN()) Py_RETURN_TRUE;
  else Py_RETURN_FALSE;
}

PyObject *THPModule_setUserEnabledMkldnn(PyObject *_unused, PyObject *arg)
{
  THPUtils_assert(PyBool_Check(arg), "set_enabled_mkldnn expects a bool, "
          "but got %s", THPUtils_typename(arg));
  at::globalContext().setUserEnabledMkldnn(arg == Py_True);
  Py_RETURN_NONE;
}

PyObject *THPModule_userEnabledMkldnn(PyObject *_unused, PyObject *noargs)
{
  if (at::globalContext().userEnabledMkldnn()) Py_RETURN_TRUE;
  else Py_RETURN_FALSE;
}

PyObject *THPModule_setDeterministicCuDNN(PyObject *_unused, PyObject *arg)
{
  THPUtils_assert(PyBool_Check(arg), "set_deterministic_cudnn expects a bool, "
          "but got %s", THPUtils_typename(arg));
  at::globalContext().setDeterministicCuDNN(arg == Py_True);
  Py_RETURN_NONE;
}

PyObject *THPModule_deterministicCuDNN(PyObject *_unused, PyObject *noargs)
{
  if (at::globalContext().deterministicCuDNN()) Py_RETURN_TRUE;
  else Py_RETURN_FALSE;
}

PyObject *THPModule_setDeterministic(PyObject *_unused, PyObject *arg)
{
  THPUtils_assert(PyBool_Check(arg), "set_deterministic expects a bool, "
          "but got %s", THPUtils_typename(arg));
  at::globalContext().setDeterministic(arg == Py_True);
  Py_RETURN_NONE;
}

PyObject *THPModule_deterministic(PyObject *_unused, PyObject *noargs)
{
  if (at::globalContext().deterministic()) Py_RETURN_TRUE;
  else Py_RETURN_FALSE;
}

PyObject *THPModule_setBenchmarkCuDNN(PyObject *_unused, PyObject *arg)
{
  THPUtils_assert(PyBool_Check(arg), "set_benchmark_cudnn expects a bool, "
          "but got %s", THPUtils_typename(arg));
#ifdef __HIP_PLATFORM_HCC__
  if (arg == Py_False) {
    TORCH_WARN_ONCE("Disabling benchmark mode for MIOpen is NOT supported. Overriding value to True");
    arg = Py_True;
  }
#endif
  at::globalContext().setBenchmarkCuDNN(arg == Py_True);
  Py_RETURN_NONE;
}

PyObject *THPModule_benchmarkCuDNN(PyObject *_unused, PyObject *noargs)
{
  if (at::globalContext().benchmarkCuDNN()) Py_RETURN_TRUE;
  else Py_RETURN_FALSE;
}

PyObject *THPModule_setAllowTF32CuBLAS(PyObject *_unused, PyObject *arg)
{
  THPUtils_assert(PyBool_Check(arg), "set_allow_tf32_cublas expects a bool, "
          "but got %s", THPUtils_typename(arg));
  at::globalContext().setAllowTF32CuBLAS(arg == Py_True);
  Py_RETURN_NONE;
}

PyObject *THPModule_allowTF32CuBLAS(PyObject *_unused, PyObject *noargs)
{
  if (at::globalContext().allowTF32CuBLAS()) Py_RETURN_TRUE;
  else Py_RETURN_FALSE;
}

PyObject *THPModule_setFlushDenormal(PyObject *_unused, PyObject *arg) {
  THPUtils_assert(PyBool_Check(arg), "flush_denormal expects a bool, "
          "but got %s", THPUtils_typename(arg));
  if (!at::globalContext().setFlushDenormal(arg == Py_True)) {
    Py_RETURN_FALSE;
  };
  Py_RETURN_TRUE;
}

PyObject *THPModule_getDefaultDtype(PyObject *_unused, PyObject *arg) {
  HANDLE_TH_ERRORS
  auto scalar_type = torch::tensors::get_default_scalar_type();
  auto dtype = (PyObject*)torch::getTHPDtype(scalar_type);
  Py_INCREF(dtype);
  return dtype;
  END_HANDLE_TH_ERRORS
}

PyObject *THPModule_getDefaultDevice(PyObject *_unused, PyObject *arg) {
  HANDLE_TH_ERRORS
  return THPUtils_packString(
          c10::DeviceTypeName(computeDeviceType(torch::tensors::get_default_dispatch_key()),
                              /*lower_case=*/true));
  END_HANDLE_TH_ERRORS
}

PyObject *THPModule_setQEngine(PyObject */* unused */, PyObject *arg)
{
  THPUtils_assert(THPUtils_checkLong(arg), "set_qengine expects an int, "
          "but got %s", THPUtils_typename(arg));
  auto qengine = static_cast<int>(THPUtils_unpackLong(arg));
  at::globalContext().setQEngine(static_cast<at::QEngine>(qengine));
  Py_RETURN_NONE;
}

PyObject *THPModule_qEngine(PyObject *_unused, PyObject *noargs)
{
  return THPUtils_packInt64(static_cast<int>(at::globalContext().qEngine()));
}

PyObject *THPModule_supportedQEngines(PyObject *_unused, PyObject *noargs)
{
  auto qengines = at::globalContext().supportedQEngines();
  auto list = THPObjectPtr(PyList_New(qengines.size()));
  for (size_t i = 0; i < qengines.size(); ++i) {
    PyObject *i64 = THPUtils_packInt64(static_cast<int>(qengines[i]));
    if (!i64) {
      throw python_error();
    }
    PyList_SET_ITEM(list.get(), i, i64);
  }
  return list.release();
}

PyObject *THPModule_isEnabledXNNPACK(PyObject *_unused, PyObject *noargs)
{
  if (at::globalContext().isXNNPACKAvailable()) Py_RETURN_TRUE;
  else Py_RETURN_FALSE;
}

static PyObject * THPModule_vmapmode_increment_nesting(PyObject* _unused, PyObject *arg) {
  HANDLE_TH_ERRORS
  return THPUtils_packInt64(at::impl::VmapMode::increment_nesting());
  END_HANDLE_TH_ERRORS
}

static PyObject * THPModule_vmapmode_decrement_nesting(PyObject* _unused, PyObject *arg) {
  HANDLE_TH_ERRORS
  return THPUtils_packInt64(at::impl::VmapMode::decrement_nesting());
  END_HANDLE_TH_ERRORS
}

//NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays, modernize-avoid-c-arrays)
static PyMethodDef TorchMethods[] = {
  {"_initExtension",  THPModule_initExtension,   METH_O,       nullptr},
  {"_autograd_init",  THPAutograd_initExtension, METH_NOARGS,  nullptr},
  {"_add_docstr",     THPModule_addDocStr,       METH_VARARGS, nullptr},
  {"_init_names",     THPModule_initNames,       METH_O,       nullptr},
  {"_has_distributed",THPModule_hasDistributed,  METH_NOARGS,  nullptr},
  {"_set_default_tensor_type", THPModule_setDefaultTensorType, METH_O, nullptr},
  {"_set_default_dtype", THPModule_setDefaultDtype, METH_O, nullptr},
  {"_infer_size",     THPModule_inferSize,         METH_VARARGS, nullptr},
  {"_crash_if_csrc_asan", THPModule_crashIfCsrcASAN, METH_O, nullptr},
  {"_crash_if_csrc_ubsan", THPModule_crashIfCsrcUBSAN, METH_O, nullptr},
  {"_crash_if_aten_asan", THPModule_crashIfATenASAN, METH_O, nullptr},
  {"_show_config",    THPModule_showConfig, METH_NOARGS, nullptr},
  {"_cxx_flags", THPModule_cxxFlags, METH_NOARGS, nullptr},
  {"_parallel_info",    THPModule_parallelInfo, METH_NOARGS, nullptr},
  {"_set_backcompat_broadcast_warn", THPModule_setBackcompatBroadcastWarn, METH_O, nullptr},
  {"_get_backcompat_broadcast_warn", THPModule_getBackcompatBroadcastWarn, METH_NOARGS, nullptr},
  {"_set_backcompat_keepdim_warn", THPModule_setBackcompatKeepdimWarn, METH_O, nullptr},
  {"_get_backcompat_keepdim_warn", THPModule_getBackcompatKeepdimWarn, METH_NOARGS, nullptr},
  {"get_num_threads", THPModule_getNumThreads,     METH_NOARGS,  nullptr},
  {"set_num_threads", THPModule_setNumThreads,     METH_O,       nullptr},
  {"get_num_interop_threads", THPModule_getNumInteropThreads,     METH_NOARGS,  nullptr},
  {"set_num_interop_threads", THPModule_setNumInteropThreads,     METH_O,       nullptr},
  {"_get_cudnn_enabled", THPModule_userEnabledCuDNN, METH_NOARGS,     nullptr},
  {"_set_cudnn_enabled", THPModule_setUserEnabledCuDNN, METH_O,  nullptr},
  {"_get_mkldnn_enabled", THPModule_userEnabledMkldnn, METH_NOARGS,     nullptr},
  {"_set_mkldnn_enabled", THPModule_setUserEnabledMkldnn, METH_O,  nullptr},
  {"_get_cudnn_allow_tf32", THPModule_allowTF32CuDNN, METH_NOARGS,     nullptr},
  {"_set_cudnn_allow_tf32", THPModule_setAllowTF32CuDNN, METH_O,  nullptr},
  {"_get_cudnn_benchmark", THPModule_benchmarkCuDNN, METH_NOARGS,     nullptr},
  {"_set_cudnn_benchmark", THPModule_setBenchmarkCuDNN, METH_O,  nullptr},
  {"_get_cudnn_deterministic", THPModule_deterministicCuDNN, METH_NOARGS,     nullptr},
  {"_set_cudnn_deterministic", THPModule_setDeterministicCuDNN, METH_O,  nullptr},
  {"_get_deterministic", THPModule_deterministic, METH_NOARGS,     nullptr},
  {"_set_deterministic", THPModule_setDeterministic, METH_O,  nullptr},
  {"_get_cublas_allow_tf32", THPModule_allowTF32CuBLAS, METH_NOARGS,     nullptr},
  {"_set_cublas_allow_tf32", THPModule_setAllowTF32CuBLAS, METH_O,  nullptr},
  {"_vmapmode_increment_nesting", THPModule_vmapmode_increment_nesting, METH_NOARGS, nullptr},
  {"_vmapmode_decrement_nesting", THPModule_vmapmode_decrement_nesting, METH_NOARGS, nullptr},
  {"_to_dlpack",      THPModule_toDLPack,          METH_O,       nullptr},
  {"_from_dlpack",    THPModule_fromDLPack,        METH_O,       nullptr},
  {"set_flush_denormal", THPModule_setFlushDenormal, METH_O,     nullptr},
  {"get_default_dtype", THPModule_getDefaultDtype, METH_NOARGS,  nullptr},
  {"_get_default_device", THPModule_getDefaultDevice, METH_NOARGS,   nullptr},
  {"_get_qengine", THPModule_qEngine, METH_NOARGS, nullptr},
  {"_set_qengine", THPModule_setQEngine, METH_O, nullptr},
  {"_supported_qengines", THPModule_supportedQEngines, METH_NOARGS, nullptr},
  {"_is_xnnpack_enabled", THPModule_isEnabledXNNPACK, METH_NOARGS, nullptr},
  {"_is_torch_function_enabled", THPModule_isEnabledTorchFunction, METH_NOARGS, nullptr},
  {"_disabled_torch_function_impl", THPModule_disable_torch_function, METH_VARARGS, nullptr},
  {nullptr, nullptr, 0, nullptr}
};

bool THCPDoubleStorage_init(PyObject *module);
bool THCPFloatStorage_init(PyObject *module);
bool THCPHalfStorage_init(PyObject *module);
bool THCPLongStorage_init(PyObject *module);
bool THCPIntStorage_init(PyObject *module);
bool THCPShortStorage_init(PyObject *module);
bool THCPCharStorage_init(PyObject *module);
bool THCPByteStorage_init(PyObject *module);
bool THCPBoolStorage_init(PyObject *module);
bool THCPBFloat16Storage_init(PyObject *module);
bool THCPComplexDoubleStorage_init(PyObject *module);
bool THCPComplexFloatStorage_init(PyObject *module);

void THCPStream_init(PyObject *module);
void THCPEvent_init(PyObject *module);
void THCPGraph_init(PyObject *module);

#ifdef USE_CUDA
PyMethodDef* THCPModule_methods();
namespace torch { namespace cuda {

void initModule(PyObject *module);

}} // namespace torch::cuda
#endif

bool THDPDoubleStorage_init(PyObject *module);
bool THDPFloatStorage_init(PyObject *module);
// TODO: fix
//bool THDPHalfStorage_init(PyObject *module);
bool THDPLongStorage_init(PyObject *module);
bool THDPIntStorage_init(PyObject *module);
bool THDPShortStorage_init(PyObject *module);
bool THDPCharStorage_init(PyObject *module);
bool THDPByteStorage_init(PyObject *module);
bool THDPBoolStorage_init(PyObject *module);
bool THDPBFloat16Storage_init(PyObject *module);
bool THDPComplexDoubleStorage_init(PyObject *module);
bool THDPComplexFloatStorage_init(PyObject *module);

static std::vector<PyMethodDef> methods;

// In Python we can't use the trick of C10_LOG_API_USAGE_ONCE
// Guaranteed to be invoked from Python under GIL, no locking on map needed
static void LogAPIUsageOnceFromPython(const std::string& event) {
  static std::unordered_set<std::string> seen;
  if (!seen.count(event)) {
    seen.insert(event);
    c10::LogAPIUsage(event);
  }
}

extern "C"
#ifdef _WIN32
__declspec(dllexport)
#endif
PyObject* initModule() {
  HANDLE_TH_ERRORS
  at::internal::lazy_init_num_threads();

  C10_LOG_API_USAGE_ONCE("torch.python.import");

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define ASSERT_TRUE(cmd) if (!(cmd)) return nullptr

  THPUtils_addPyMethodDefs(methods, TorchMethods);
  THPUtils_addPyMethodDefs(methods, DataLoaderMethods);
  THPUtils_addPyMethodDefs(methods, torch::autograd::python_functions());
  THPUtils_addPyMethodDefs(methods, torch::multiprocessing::python_functions());
#ifdef USE_CUDA
  THPUtils_addPyMethodDefs(methods, THCPModule_methods());
#endif
#if defined(USE_DISTRIBUTED) && defined(USE_C10D)
  THPUtils_addPyMethodDefs(methods, torch::distributed::c10d::python_functions());
#ifndef _WIN32
  THPUtils_addPyMethodDefs(methods, torch::distributed::rpc::python_functions());
  THPUtils_addPyMethodDefs(
      methods, torch::distributed::autograd::python_functions());
  THPUtils_addPyMethodDefs(methods, torch::distributed::rpc::testing::python_functions());
#endif
#endif

  static struct PyModuleDef torchmodule = {
     PyModuleDef_HEAD_INIT,
     "torch._C",
     nullptr,
     -1,
     methods.data()
  };
  ASSERT_TRUE(module = PyModule_Create(&torchmodule));
  ASSERT_TRUE(THPWrapper_init(module));
  ASSERT_TRUE(THPGenerator_init(module));
  ASSERT_TRUE(THPException_init(module));
  THPSize_init(module);
  THPDtype_init(module);
  THPDTypeInfo_init(module);
  THPLayout_init(module);
  THPMemoryFormat_init(module);
  THPQScheme_init(module);
  THPDevice_init(module);
  THPStream_init(module);
  ASSERT_TRUE(THPVariable_initModule(module));
  ASSERT_TRUE(THPFunction_initModule(module));
  ASSERT_TRUE(THPEngine_initModule(module));
  // NOTE: We need to be able to access OperatorExportTypes from ONNX for use in
  // the export side of JIT, so this ONNX init needs to appear before the JIT
  // init.
  torch::onnx::initONNXBindings(module);
  torch::jit::initJITBindings(module);
  torch::impl::dispatch::initDispatchBindings(module);
  torch::throughput_benchmark::initThroughputBenchmarkBindings(module);
  torch::autograd::initNNFunctions(module);
  torch::autograd::initFFTFunctions(module);
  torch::autograd::initLinalgFunctions(module);
  torch::autograd::init_legacy_variable(module);
  torch::python::init_bindings(module);
#ifdef USE_CUDA
  torch::cuda::initModule(module);
#endif
  ASSERT_TRUE(THPDoubleStorage_init(module));
  ASSERT_TRUE(THPFloatStorage_init(module));
  ASSERT_TRUE(THPHalfStorage_init(module));
  ASSERT_TRUE(THPLongStorage_init(module));
  ASSERT_TRUE(THPIntStorage_init(module));
  ASSERT_TRUE(THPShortStorage_init(module));
  ASSERT_TRUE(THPCharStorage_init(module));
  ASSERT_TRUE(THPByteStorage_init(module));
  ASSERT_TRUE(THPBoolStorage_init(module));
  ASSERT_TRUE(THPQUInt8Storage_init(module));
  ASSERT_TRUE(THPQInt8Storage_init(module));
  ASSERT_TRUE(THPQInt32Storage_init(module));
  ASSERT_TRUE(THPQUInt4x2Storage_init(module));
  ASSERT_TRUE(THPBFloat16Storage_init(module));
  ASSERT_TRUE(THPComplexDoubleStorage_init(module));
  ASSERT_TRUE(THPComplexFloatStorage_init(module));

#ifdef USE_CUDA
  // This will only initialise base classes and attach them to library namespace
  // They won't be ready for real usage until importing cuda module, that will
  // complete the process (but it defines Python classes before calling back into
  // C, so these lines have to execute first)..
  ASSERT_TRUE(THCPDoubleStorage_init(module));
  ASSERT_TRUE(THCPFloatStorage_init(module));
  ASSERT_TRUE(THCPHalfStorage_init(module));
  ASSERT_TRUE(THCPLongStorage_init(module));
  ASSERT_TRUE(THCPIntStorage_init(module));
  ASSERT_TRUE(THCPShortStorage_init(module));
  ASSERT_TRUE(THCPCharStorage_init(module));
  ASSERT_TRUE(THCPByteStorage_init(module));
  ASSERT_TRUE(THCPBoolStorage_init(module));
  ASSERT_TRUE(THCPBFloat16Storage_init(module));
  ASSERT_TRUE(THCPComplexDoubleStorage_init(module));
  ASSERT_TRUE(THCPComplexFloatStorage_init(module));

  THCPStream_init(module);
  THCPEvent_init(module);
  THCPGraph_init(module);
#endif

  auto set_module_attr = [&](const char* name, PyObject* v, bool incref = true) {
    // PyModule_AddObject steals reference
    if (incref) {
      Py_INCREF(v);
    }
    return PyModule_AddObject(module, name, v) == 0;
  };

#if defined(USE_CUDNN) || defined(__HIP_PLATFORM_HCC__)
  PyObject *has_cudnn = Py_True;
#else
  PyObject *has_cudnn = Py_False;
#endif
 ASSERT_TRUE(set_module_attr("has_cudnn", has_cudnn));

  // force ATen to initialize because it handles
  // setting up TH Errors so that they throw C++ exceptions
  at::init();

  // Automatically translate errors thrown from pybind11 functions
  py::register_exception_translator([](std::exception_ptr e) { // NOLINT
    try {
      if (e) {
        std::rethrow_exception(e);
      }
    }
    CATCH_TH_ERRORS()
  });

  auto py_module = py::reinterpret_borrow<py::module>(module);
  py_module.def("_demangle", &c10::demangle);
  py_module.def("_log_api_usage_once", &LogAPIUsageOnceFromPython);

  py_module.def(
    "init_num_threads",
    torch::wrap_pybind_function(at::init_num_threads),
    R"(
init_num_threads()

Initializes the number of parallel threads used on the current thread.

Call this whenever a new thread is created in order to propagate values from
:func:`torch.set_num_threads` onto the new thread.
)");

  ASSERT_TRUE(set_module_attr("has_openmp", at::hasOpenMP() ? Py_True : Py_False));
  ASSERT_TRUE(set_module_attr("has_mkl", at::hasMKL() ? Py_True : Py_False));
  ASSERT_TRUE(set_module_attr("has_lapack", at::hasLAPACK() ? Py_True : Py_False));

  py_module.def(
    "_valgrind_supported_platform", [](){
      #if defined(USE_VALGRIND)
      return true;
      #else
      return false;
      #endif
    }
  );

  py_module.def(
    "_valgrind_toggle", [](){
      #if defined(USE_VALGRIND)
      CALLGRIND_TOGGLE_COLLECT;
      #else
      TORCH_CHECK(false, "Valgrind is not supported.");
      #endif
    }
  );

#ifdef USE_CUDA
  PyObject *has_cuda = Py_True;
#else
  PyObject *has_cuda = Py_False;
#endif
  ASSERT_TRUE(set_module_attr("has_cuda", has_cuda));

  ASSERT_TRUE(set_module_attr("has_mkldnn", at::hasMKLDNN() ? Py_True : Py_False));

#ifdef _GLIBCXX_USE_CXX11_ABI
  ASSERT_TRUE(set_module_attr("_GLIBCXX_USE_CXX11_ABI", _GLIBCXX_USE_CXX11_ABI ? Py_True : Py_False));
#else
  ASSERT_TRUE(set_module_attr("_GLIBCXX_USE_CXX11_ABI", Py_False));
#endif

// See note [Pybind11 ABI constants]
#define SET_STR_DEFINE(name) \
  ASSERT_TRUE(set_module_attr("_" # name, THPUtils_packString(name)))

#ifdef PYBIND11_COMPILER_TYPE
  SET_STR_DEFINE(PYBIND11_COMPILER_TYPE);
#else
  ASSERT_TRUE(set_module_attr("_" C10_STRINGIZE(PYBIND11_COMPILER_TYPE), Py_None));
#endif

#ifdef PYBIND11_STDLIB
  SET_STR_DEFINE(PYBIND11_STDLIB);
#else
  ASSERT_TRUE(set_module_attr("_" C10_STRINGIZE(PYBIND11_STDLIB), Py_None));
#endif

#ifdef PYBIND11_BUILD_ABI
  SET_STR_DEFINE(PYBIND11_BUILD_ABI);
#else
  ASSERT_TRUE(set_module_attr("_" C10_STRINGIZE(PYBIND11_BUILD_ABI), Py_None));
#endif
#undef SET_STR_DEFINE

  const auto& defaultGenerator = at::detail::getDefaultCPUGenerator();
  THPDefaultCPUGenerator = (THPGenerator*)THPGenerator_initDefaultGenerator(defaultGenerator);
  // This reference is meant to be given away, so no need to incref here.
  ASSERT_TRUE(set_module_attr("default_generator", (PyObject*)THPDefaultCPUGenerator, /* incref= */ false));
  ASSERT_TRUE(set_module_attr("DisableTorchFunction", (PyObject*)THPModule_DisableTorchFunctionType(), /* incref= */ false));
  torch::set_disabled_torch_function_impl(PyObject_GetAttrString(module, "_disabled_torch_function_impl"));
  ASSERT_TRUE(torch::disabled_torch_function_impl() != nullptr);
#ifdef USE_NUMPY
  if (_import_array() < 0) return nullptr;
#endif
  return module;
  END_HANDLE_TH_ERRORS
}

// Checks that the _C shared library isn't initialized multiple times. This
// can happen if the same csrc files are compiled into multiple shared
// libraries.
inline void pytorch_duplicate_guard() {
  static int initialized = 0;
  if (initialized) {
    fprintf(stderr, "pytorch: _C shared library re-initialized\n");
    abort();
  }
  initialized = 1;
;}

struct call_duplicate_guard {
  call_duplicate_guard() { pytorch_duplicate_guard(); }
};

static call_duplicate_guard _call_duplicate_guard;
