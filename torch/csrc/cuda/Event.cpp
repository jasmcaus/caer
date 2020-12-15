#include <pybind11/pybind11.h>
#include <torch/csrc/cuda/Event.h>
#include <torch/csrc/cuda/Module.h>
#include <torch/csrc/cuda/Stream.h>
#include <torch/csrc/Device.h>
#include <torch/csrc/THP.h>
#include <torch/csrc/utils/pycfunction_helpers.h>
#include <torch/csrc/utils/python_arg_parser.h>

#include <c10/cuda/CUDAGuard.h>

#include <structmember.h>
#include <cuda_runtime_api.h>

PyObject *THCPEventClass = nullptr;

static PyObject * THCPEvent_pynew(
    PyTypeObject *type, PyObject *args, PyObject *kwargs) {
  HANDLE_TH_ERRORS
  unsigned char enable_timing = 0;
  unsigned char blocking = 0;
  unsigned char interprocess = 0;

  static char *kwlist[] =
    {"enable_timing", "blocking", "interprocess", nullptr};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|bbb", kwlist,
      &enable_timing, &blocking, &interprocess)) {
    return nullptr;
  }

  THPObjectPtr ptr(type->tp_alloc(type, 0));
  if (!ptr) {
    return nullptr;
  }

  THCPEvent* self = (THCPEvent *)ptr.get();
  unsigned int flags =
    (blocking ? cudaEventBlockingSync : cudaEventDefault) |
    (enable_timing ? cudaEventDefault : cudaEventDisableTiming) |
    (interprocess ? cudaEventInterprocess : cudaEventDefault);

  new (&self->cuda_event) at::cuda::CUDAEvent(flags);

  return (PyObject *)ptr.release();
  END_HANDLE_TH_ERRORS
}

static PyObject * THCPEvent_from_ipc_handle(
    PyObject *_type, PyObject *args, PyObject *kwargs) {
  HANDLE_TH_ERRORS
  auto type = (PyTypeObject*)_type;

  static torch::PythonArgParser parser({
    "from_ipc_handle(Device device, std::string ipc_handle)",
  });
  torch::ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  at::Device device = r.device(0);
  std::string handle_string = r.string(1);

  TORCH_CHECK(handle_string.size() == sizeof(cudaIpcEventHandle_t),
    "cudaIpcEventHandle_t expects byte-like object of size ",
    sizeof(cudaIpcEventHandle_t), ", but got ", handle_string.size());
  TORCH_CHECK(device.type() == at::kCUDA, "Event can only be created on "
    "CUDA devices, but got device type ", device.type())

  THPObjectPtr ptr(type->tp_alloc(type, 0));
  if (!ptr) {
    return nullptr;
  }
  THCPEvent* self = (THCPEvent *)ptr.get();

  cudaIpcEventHandle_t handle;
  std::memcpy(&handle, handle_string.c_str(), handle_string.size());
  new (&self->cuda_event) at::cuda::CUDAEvent(device.index(), &handle);

  return (PyObject *)ptr.release();
  END_HANDLE_TH_ERRORS
}

static void THCPEvent_dealloc(THCPEvent *self) {
  self->cuda_event.~CUDAEvent();
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject * THCPEvent_get_cuda_event(THCPEvent *self, void *unused) {
  HANDLE_TH_ERRORS
  return PyLong_FromVoidPtr(self->cuda_event.event());
  END_HANDLE_TH_ERRORS
}

static PyObject * THCPEvent_get_device(THCPEvent *self, void *unused) {
  HANDLE_TH_ERRORS
  at::optional<at::Device> device = self->cuda_event.device();
  if (!device) {
    Py_RETURN_NONE;
  }
  return THPDevice_New(device.value());
  END_HANDLE_TH_ERRORS
}

static PyObject * THCPEvent_record(PyObject *_self, PyObject *_stream) {
  HANDLE_TH_ERRORS
  auto self = (THCPEvent*)_self;
  auto stream = (THCPStream*)_stream;
  self->cuda_event.record(stream->cuda_stream);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject * THCPEvent_wait(PyObject *_self, PyObject *_stream) {
  HANDLE_TH_ERRORS
  {
    auto self = (THCPEvent*)_self;
    auto stream = (THCPStream*)_stream;
    pybind11::gil_scoped_release no_gil;
    self->cuda_event.block(stream->cuda_stream);
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject * THCPEvent_query(PyObject *_self, PyObject *noargs) {
  HANDLE_TH_ERRORS
  auto self = (THCPEvent*)_self;
  return PyBool_FromLong(self->cuda_event.query());
  END_HANDLE_TH_ERRORS
}

static PyObject * THCPEvent_elapsed_time(PyObject *_self, PyObject *_other) {
  HANDLE_TH_ERRORS
  auto self = (THCPEvent*)_self;
  auto other = (THCPEvent*)_other;
  return PyFloat_FromDouble(self->cuda_event.elapsed_time(other->cuda_event));
  END_HANDLE_TH_ERRORS
}

static PyObject * THCPEvent_synchronize(PyObject *_self, PyObject *noargs) {
  HANDLE_TH_ERRORS
  {
    auto self = (THCPEvent*)_self;
    pybind11::gil_scoped_release no_gil;
    self->cuda_event.synchronize();
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject * THCPEvent_ipc_handle(PyObject *_self, PyObject *noargs) {
  HANDLE_TH_ERRORS
  auto self = (THCPEvent*)_self;
  cudaIpcEventHandle_t handle;
  self->cuda_event.ipc_handle(&handle);
  return PyBytes_FromStringAndSize((const char *)&handle, sizeof(handle));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef THCPEvent_properties[] = {
  {"device", (getter)THCPEvent_get_device, nullptr, nullptr, nullptr},
  {"cuda_event", (getter)THCPEvent_get_cuda_event, nullptr, nullptr, nullptr},
  {nullptr}
};

static PyMethodDef THCPEvent_methods[] = {
  {(char*)"from_ipc_handle",
    castPyCFunctionWithKeywords(THCPEvent_from_ipc_handle),
    METH_CLASS | METH_VARARGS | METH_KEYWORDS, nullptr},
  {(char*)"record", THCPEvent_record, METH_O, nullptr},
  {(char*)"wait", THCPEvent_wait, METH_O, nullptr},
  {(char*)"query", THCPEvent_query, METH_NOARGS, nullptr},
  {(char*)"elapsed_time", THCPEvent_elapsed_time, METH_O, nullptr},
  {(char*)"synchronize", THCPEvent_synchronize,
    METH_NOARGS, nullptr},
  {(char*)"ipc_handle", THCPEvent_ipc_handle,
    METH_NOARGS, nullptr},
  {nullptr}
};

PyTypeObject THCPEventType = {
  PyVarObject_HEAD_INIT(nullptr, 0)
  "torch._C._CudaEventBase",             /* tp_name */
  sizeof(THCPEvent),                     /* tp_basicsize */
  0,                                     /* tp_itemsize */
  (destructor)THCPEvent_dealloc,         /* tp_dealloc */
  0,                                     /* tp_vectorcall_offset */
  0,                                     /* tp_getattr */
  0,                                     /* tp_setattr */
  0,                                     /* tp_reserved */
  0,                                     /* tp_repr */
  0,                                     /* tp_as_number */
  0,                                     /* tp_as_sequence */
  0,                                     /* tp_as_mapping */
  0,                                     /* tp_hash  */
  0,                                     /* tp_call */
  0,                                     /* tp_str */
  0,                                     /* tp_getattro */
  0,                                     /* tp_setattro */
  0,                                     /* tp_as_buffer */
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
  nullptr,                                  /* tp_doc */
  0,                                     /* tp_traverse */
  0,                                     /* tp_clear */
  0,                                     /* tp_richcompare */
  0,                                     /* tp_weaklistoffset */
  0,                                     /* tp_iter */
  0,                                     /* tp_iternext */
  THCPEvent_methods,                     /* tp_methods */
  0,                                     /* tp_members */
  THCPEvent_properties,                  /* tp_getset */
  0,                                     /* tp_base */
  0,                                     /* tp_dict */
  0,                                     /* tp_descr_get */
  0,                                     /* tp_descr_set */
  0,                                     /* tp_dictoffset */
  0,                                     /* tp_init */
  0,                                     /* tp_alloc */
  THCPEvent_pynew,                       /* tp_new */
};

void THCPEvent_init(PyObject *module) {
  THCPEventClass = (PyObject*)&THCPEventType;
  if (PyType_Ready(&THCPEventType) < 0) {
    throw python_error();
  }
  Py_INCREF(&THCPEventType);
  if (PyModule_AddObject(
      module, "_CudaEventBase", (PyObject *)&THCPEventType) < 0) {
    throw python_error();
  }
}
