#include <torch/csrc/utils/tensor_layouts.h>
#include <ATen/Layout.h>
#include <c10/core/ScalarType.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/Layout.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/object_ptr.h>

namespace torch { namespace utils {

void initializeLayouts() {
  auto torch_module = THPObjectPtr(PyImport_ImportModule("torch"));
  if (!torch_module) throw python_error();

  PyObject *strided_layout = THPLayout_New(at::Layout::Strided, "torch.strided");
  Py_INCREF(strided_layout);
  if (PyModule_AddObject(torch_module, "strided", strided_layout) != 0) {
    throw python_error();
  }
  registerLayoutObject((THPLayout*)strided_layout, at::Layout::Strided);

  PyObject *sparse_coo_layout = THPLayout_New(at::Layout::Sparse, "torch.sparse_coo");
  Py_INCREF(sparse_coo_layout);
  if (PyModule_AddObject(torch_module, "sparse_coo", sparse_coo_layout) != 0) {
    throw python_error();
  }
  registerLayoutObject((THPLayout*)sparse_coo_layout, at::Layout::Sparse);

  PyObject *mkldnn_layout = THPLayout_New(at::Layout::Mkldnn, "torch._mkldnn");
  Py_INCREF(mkldnn_layout);
  if (PyModule_AddObject(torch_module, "_mkldnn", mkldnn_layout) != 0) {
    throw python_error();
  }
  registerLayoutObject((THPLayout*)mkldnn_layout, at::Layout::Mkldnn);
}

}} // namespace torch::utils
