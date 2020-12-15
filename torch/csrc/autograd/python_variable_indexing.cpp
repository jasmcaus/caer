#include <torch/csrc/autograd/python_variable_indexing.h>

#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/THP_export.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/autograd/utils/wrap_outputs.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/utils/python_compat.h>
#include <torch/csrc/utils/python_numbers.h>
#include <torch/csrc/utils/tensor_new.h>
#include <torch/csrc/utils/python_arg_parser.h>
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/utils/tensor_types.h>

#include <ATen/DeviceGuard.h>
#include <ATen/ExpandUtils.h>
#include <ATen/TensorIndexing.h>
#include <ATen/TracerMode.h>
#include <c10/core/TensorOptions.h>
#include <ATen/core/LegacyTypeDispatch.h>

#include <vector>
#include <tuple>

using namespace at;
using namespace torch::autograd::utils;

namespace torch { namespace autograd {

Py_ssize_t THPVariable_length(PyObject* self) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function(self)) {
    py::object ret = py::reinterpret_steal<py::object>(handle_torch_function(self, "__len__"));
    Py_ssize_t length = PyLong_AsSsize_t(ret.ptr());
    if (PyErr_Occurred()) {
      throw python_error();
    }
    return length;
  }
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  if (self_.dim() == 0) {
    return 0;
  }
  return (Py_ssize_t)self_.size(0);
  END_HANDLE_TH_ERRORS_RET(-1)
}


// We allow indexing by integers, slices, ellipsis, None, Variables,
// and tuples of those types. We also handle bools as if they were a
// Variable[ByteTensor].

static inline int64_t count_specified_dimensions(PyObject* index) {
  // Count the number of indexed dimensions (everything but ellipsis and None)
  int64_t count = 0;
  auto size = PyTuple_GET_SIZE(index); // NOLINT(cppcoreguidelines-pro-type-cstyle-cast)
  for (Py_ssize_t i = 0; i < size; i++) {
    PyObject* obj = PyTuple_GET_ITEM(index, i); // NOLINT(cppcoreguidelines-pro-type-cstyle-cast)
    if (THPVariable_Check(obj)) {
      auto& var = reinterpret_cast<THPVariable*>(obj)->cdata;
      const auto& var_scalar_type = var.scalar_type();
      if (var_scalar_type == kByte || var_scalar_type == kBool) {
        count += var.dim();
      } else {
        count++;
      }
    } else if (obj != Py_None && obj != Py_Ellipsis && obj != Py_True && obj != Py_False) { // NOLINT(cppcoreguidelines-pro-type-cstyle-cast)
      count++;
    }
  }
  return count;
}

[[noreturn]]
static inline void invalid_index(PyObject* obj) {
  throw IndexError(
    "only integers, slices (`:`), ellipsis (`...`), None and long or byte "
    "Variables are valid indices (got %s)", Py_TYPE(obj)->tp_name);
}

static inline Variable sequenceToVariable(c10::DispatchKey dispatch_key, PyObject* seq) {
  return torch::utils::indexing_tensor_from_data(dispatch_key, kLong, c10::nullopt, seq);
}

static inline Variable valueToTensor(c10::TensorOptions options, PyObject* value, const at::Device& device) {
  if (THPVariable_Check(value)) {
    return reinterpret_cast<THPVariable*>(value)->cdata;
  }
  at::AutoNonVariableTypeMode guard;  // TODO: remove
  at::tracer::impl::NoTracerDispatchMode tracer_guard;
  if (THPUtils_checkLong(value) || PyBool_Check(value)) {
    return at::indexing::scalarToTensor(Scalar(THPUtils_unpackLong(value)), options, device);
  }
  if (PyFloat_Check(value)) {
    return at::indexing::scalarToTensor(Scalar(THPUtils_unpackDouble(value)), options, device);
  }
  if (PyComplex_Check(value)) {
    return at::indexing::scalarToTensor(Scalar(THPUtils_unpackComplexDouble(value)), options, device);
  }
  throw TypeError(
    "can't assign a %s to a %s",
    Py_TYPE(value)->tp_name,
    torch::utils::options_to_string(options).c_str());
}

static inline void checkUnpackSlice(PyObject* index, Py_ssize_t* start_ptr, Py_ssize_t* stop_ptr, Py_ssize_t* step_ptr) {
  if (!THPUtils_unpackSlice(index, start_ptr, stop_ptr, step_ptr)) {
    throw python_error();
  }
}

static inline void recordSliceTrace(PyObject* obj) {
  PySliceObject* sliceobj = (PySliceObject*)obj;
  if (THPVariable_Check(sliceobj->start)) {
    torch::jit::tracer::ArgumentStash::stashValue(std::string("start"), 1, THPVariable_Unpack(sliceobj->start), torch::jit::IntType::get());
  }
  if (THPVariable_Check(sliceobj->stop)) {
    torch::jit::tracer::ArgumentStash::stashValue(std::string("end"), 1, THPVariable_Unpack(sliceobj->stop), torch::jit::IntType::get());
  }
  if (THPVariable_Check(sliceobj->step)) {
    torch::jit::tracer::ArgumentStash::stashValue(std::string("step"), 1, THPVariable_Unpack(sliceobj->step), torch::jit::IntType::get());
  }
}

static inline void recordSelectTrace(const Tensor& index_tensor) {
  torch::jit::tracer::ArgumentStash::stashValue(std::string("index"), 1, index_tensor, torch::jit::IntType::get());
}

static inline Variable applySlicing(
    const Variable& self,
    PyObject* index,
    variable_list& outIndices,
    bool is_tracing,
    const at::Device& self_device,
    const IntArrayRef& self_sizes) {
  int64_t size = PyTuple_GET_SIZE(index); // NOLINT(cppcoreguidelines-pro-type-cstyle-cast)
  int64_t dim = 0;
  int64_t specified_dims = count_specified_dimensions(index);

  if (specified_dims > (int64_t)self_sizes.size()) {
    throw IndexError("too many indices for tensor of dimension %d", (int)(self_sizes.size()));
  }

  Variable result = self;
  for (int64_t i = 0; i < size; i++) {
    PyObject* obj = PyTuple_GET_ITEM(index, i); // NOLINT(cppcoreguidelines-pro-type-cstyle-cast)
    result = at::indexing::handleDimInMultiDimIndexing(
      /*prev_dim_result=*/result,
      /*original_tensor=*/self,
      /*index=*/([&]() {
        if (THPUtils_checkLong(obj)) {
          if (is_tracing && THPVariable_Check(obj)) {
            recordSelectTrace(THPVariable_Unpack(obj));
          }
          return at::indexing::TensorIndex(THPUtils_unpackLong(obj));
        } else if (PySlice_Check(obj)) {
          Py_ssize_t start, stop, step;
          checkUnpackSlice(obj, &start, &stop, &step);
          if (is_tracing) {
            recordSliceTrace(obj);
          }
          return at::indexing::TensorIndex(at::indexing::Slice(start, stop, step));
        } else if (obj == Py_Ellipsis) {
          return at::indexing::TensorIndex(at::indexing::Ellipsis);
        } else if (obj == Py_None) {
          return at::indexing::TensorIndex(at::indexing::None);
        } else if (PyBool_Check(obj)) {
          return at::indexing::TensorIndex(obj == Py_True);
        } else if (THPVariable_Check(obj)) {
          Tensor tensor = THPVariable_Unpack(obj);
          if (is_tracing) {
            auto scalar_type = tensor.scalar_type();
            if (tensor.dim() == 0 && at::isIntegralType(scalar_type, /*includeBool=*/false) && scalar_type != at::kByte) {
              recordSelectTrace(tensor);
            }
          }
          return at::indexing::TensorIndex(std::move(tensor));
        } else if (PySequence_Check(obj)) {
          // TODO: Naughty naughty get out of jail free
          // (Fixing this means I have to fix the call chain though :/)
          return at::indexing::TensorIndex(sequenceToVariable(legacyExtractDispatchKey(self), obj));
        } else {
          auto idx = THPObjectPtr(PyNumber_Index(obj));
          if (!idx) {
            PyErr_Clear();
            invalid_index(obj);
          }
          if (is_tracing && THPVariable_Check(idx)) {
            recordSelectTrace(THPVariable_Unpack(idx));
          }
          return at::indexing::TensorIndex(THPUtils_unpackLong(idx));
        }
      })(),
      /*dim_ptr=*/&dim,
      /*specified_dims_ptr=*/&specified_dims,
      /*real_dim=*/i,
      /*outIndices=*/outIndices,
      // See NOTE [ Setting `disable_slice_optimization` when calling C++ tensor indexing functions from Python ]
      /*disable_slice_optimization=*/is_tracing,
      /*original_tensor_device=*/self_device,
      /*prev_dim_result_sizes=*/result.sizes());
  }
  return result;
}

static inline bool treatSequenceAsTuple(PyObject* index) {
  if (PyTuple_Check(index)) {
    return true;
  }
  if (!PySequence_Check(index)) {
    return false;
  }
  // This uses a heuristics from NumPy for determining whether to treat
  // non-tuple sequences as if they were a tuple. From the NumPy code comments:
  //
  // "At this point, we're left with a non-tuple, non-array, sequence:
  //  typically, a list. We use some somewhat-arbitrary heuristics from here
  //  onwards to decided whether to treat that list as a single index, or a
  //  list of indices. Backwards compatibility only takes effect for short
  //  sequences - otherwise we treat it like any other scalar."
  auto n = PySequence_Size(index);
  if (n < 0) {
    // Negative size indicates a Python error in the PySequence_Size call.
    PyErr_Clear();
    return false;
  }
  if (n >= 32) {
    return false;
  }
  for (Py_ssize_t i = 0; i < n; i++) {
    auto obj = THPObjectPtr{PySequence_GetItem(index, i)};
    if (!obj.get()) {
      PyErr_Clear();
      return false;
    }
    if (THPVariable_Check(obj.get()) || PySequence_Check(obj.get()) || PySlice_Check(obj.get())) {
      return true;
    }
    if (obj.get() == Py_Ellipsis || obj.get() == Py_None) {
      return true;
    }
  }
  return false;
}

static inline THPObjectPtr wrapTuple(PyObject* index) {
  THPObjectPtr res;
  if (treatSequenceAsTuple(index)) {
    res = PySequence_Tuple(index);
  } else {
    res = PyTuple_Pack(1, index); // NOLINT(cppcoreguidelines-pro-type-cstyle-cast)
  }
  if (!res) throw python_error();
  return res;
}

// NOTE: Here is the dispatch structure for `THPVariable_getitem`:
//
// 1. Python 1-D getter calls C++ `at::indexing::get_item` after
// converting Python index to C++ TensorIndex.
//
// 2. Python N-D getter calls C++ `at::indexing::handleDimInMultiDimIndexing`
// for each dim, after converting Python index to C++ TensorIndex. If advanced
// indexing is needed, it calls C++ `at::indexing::dispatch_index`.
PyObject* THPVariable_getitem(PyObject* self, PyObject* index) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function(self)) {
    py::tuple args_ = py::make_tuple(py::handle(index));
    return handle_torch_function(self, "__getitem__", args_.ptr());
  }
  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  OptionalDeviceGuard device_guard(device_of(self_));

  // handle simple types: none, ellipsis
  if (index == Py_None) {
    return THPVariable_Wrap(
      at::indexing::get_item(self_, {at::indexing::TensorIndex(at::indexing::None)}));
  } else if (index == Py_Ellipsis) {
    return THPVariable_Wrap(
      at::indexing::get_item(self_, {at::indexing::TensorIndex(at::indexing::Ellipsis)}));
  }

  bool is_tracing = torch::jit::tracer::isTracing();

  // handle simple types: integers, slices, bool
  if (THPUtils_checkLong(index)) {
    if (is_tracing && THPVariable_Check(index)) {
      recordSelectTrace(THPVariable_Unpack(index));
    }
    return THPVariable_Wrap(
      at::indexing::get_item(self_, {at::indexing::TensorIndex(THPUtils_unpackLong(index))}));
  } else if (PySlice_Check(index)) {
    Py_ssize_t start, stop, step;
    checkUnpackSlice(index, &start, &stop, &step);
    if (is_tracing) {
      recordSliceTrace(index);
    }
    return THPVariable_Wrap(
      at::indexing::get_item(self_, {at::indexing::TensorIndex(at::indexing::Slice(start, stop, step))}));
  } else if (index == Py_False || index == Py_True) {
    return THPVariable_Wrap(([&]() {
      pybind11::gil_scoped_release no_gil;
      return at::indexing::get_item(self_, {at::indexing::TensorIndex(index == Py_True)});
    })());
  }

  // wrap index in a tuple if it's not already one
  THPObjectPtr holder = wrapTuple(index);

  variable_list variableIndices;
  Variable sliced = applySlicing(
    self_, holder.get(), variableIndices, /*is_tracing=*/is_tracing, self_.device(), self_.sizes());
  if (variableIndices.empty()) {
    if (sliced.is_same(self_)) {
      // ensure we return a shallow copy for things like x[...]
      sliced = at::alias(sliced);
    }
    return THPVariable_Wrap(std::move(sliced));
  }

  // indexing by tensors ("advanced" indexing)
  return THPVariable_Wrap(([&]() {
    pybind11::gil_scoped_release no_gil;
    return at::indexing::dispatch_index(sliced, std::move(variableIndices));
  })());

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// NOTE: Here is the dispatch structure for `THPVariable_setitem`:
//
// 1. Python 1-D setter calls C++ `at::indexing::set_item` after
// converting Python index to C++ TensorIndex.
//
// 2. Python N-D setter calls C++ `at::indexing::handleDimInMultiDimIndexing`
// for each dim, after converting Python index to C++ TensorIndex. If advanced
// indexing is needed, it calls C++ `at::indexing::dispatch_index_put_`.
int THPVariable_setitem(PyObject* self, PyObject* index, PyObject* py_value) {
  HANDLE_TH_ERRORS
  if (py_value == nullptr) {
    throw TypeError("Tensor does not support deleting items");
  }
  if (check_has_torch_function(self)) {
    py::tuple args_ = py::make_tuple(py::handle(index), py::handle(py_value));
    py::object ret = py::reinterpret_steal<py::object>(handle_torch_function(self, "__setitem__", args_.ptr()));
    return 0;
  }

  auto& self_ = reinterpret_cast<THPVariable*>(self)->cdata;
  OptionalDeviceGuard device_guard(device_of(self_));
  at::Device self_device = self_.device();
  Variable value;
  // TODO: This qint special case looks very suspicious...
  if (isQIntType(self_.scalar_type())) {
    value = valueToTensor(device(kCPU).dtype(kFloat), py_value, at::Device(kCPU));
  } else {
    value = valueToTensor(self_.options(), py_value, self_device);
  }

  // handle simple types: ellipsis, none, bool
  if (index == Py_False) { // NOLINT(cppcoreguidelines-pro-type-cstyle-cast)
    // do nothing for false (technically we should check the size, but we don't have
    // real 0-sized shapes.
    return 0;
  } else if (index == Py_Ellipsis) {
    at::indexing::set_item(self_, {at::indexing::TensorIndex(at::indexing::Ellipsis)}, value);
    return 0;
  } else if (index == Py_None) {
    at::indexing::set_item(self_, {at::indexing::TensorIndex(at::indexing::None)}, value);
    return 0;
  } else if (index == Py_True) {
    at::indexing::set_item(self_, {at::indexing::TensorIndex(true)}, value);
    return 0;
  }

  bool is_tracing = torch::jit::tracer::isTracing();

  // handle simple types: integers, slices
  if (THPUtils_checkLong(index)) {
    if (is_tracing && THPVariable_Check(index)) {
      recordSelectTrace(THPVariable_Unpack(index));
    }
    at::indexing::set_item(self_, {at::indexing::TensorIndex(THPUtils_unpackLong(index))}, value);
    return 0;
  } else if (PySlice_Check(index)) {
    Py_ssize_t start, stop, step;
    checkUnpackSlice(index, &start, &stop, &step);
    if (is_tracing) {
      recordSliceTrace(index);
    }
    // See NOTE [ Setting `disable_slice_optimization` when calling C++ tensor indexing functions from Python ]
    at::indexing::set_item(
      self_, {at::indexing::TensorIndex(at::indexing::Slice(start, stop, step))}, value, /*disable_slice_optimization=*/is_tracing);
    return 0;
  }

  // wrap index in a tuple if it's not already one
  THPObjectPtr holder = wrapTuple(index);

  variable_list variableIndices;
  Variable sliced = applySlicing(
    self_, holder.get(), variableIndices, /*is_tracing=*/is_tracing, self_device, self_.sizes());
  if (variableIndices.empty()) {
    at::indexing::copy_to(sliced, value);
    return 0;
  }

  IntArrayRef valueSizes = value.sizes();
  IntArrayRef slicedValueSizes = at::indexing::slicePrefix1sSize(valueSizes);
  torch::autograd::Variable valuesSliced;
  if (!valueSizes.equals(slicedValueSizes)) {
    valuesSliced = value.view(slicedValueSizes);
  } else {
    valuesSliced = value;
  }
  {
    pybind11::gil_scoped_release no_gil;
    at::indexing::dispatch_index_put_(sliced, std::move(variableIndices), valuesSliced);
  }
  return 0;
  END_HANDLE_TH_ERRORS_RET(-1)
}

}} // namespace torch::autograd
