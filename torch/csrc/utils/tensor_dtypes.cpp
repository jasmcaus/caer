#include <torch/csrc/utils/tensor_dtypes.h>
#include <torch/csrc/Dtype.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/autograd/generated/VariableType.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/tensor_types.h>

namespace torch {
namespace utils {

std::pair<std::string, std::string> getDtypeNames(
    at::ScalarType scalarType) {
  switch (scalarType) {
    case at::ScalarType::Byte:
      // no "byte" because byte is signed in numpy and we overload
      // byte to mean bool often
      return std::make_pair("uint8", "");
    case at::ScalarType::Char:
      // no "char" because it is not consistently signed or unsigned; we want
      // to move to int8
      return std::make_pair("int8", "");
    case at::ScalarType::Double:
      return std::make_pair("float64", "double");
    case at::ScalarType::Float:
      return std::make_pair("float32", "float");
    case at::ScalarType::Int:
      return std::make_pair("int32", "int");
    case at::ScalarType::Long:
      return std::make_pair("int64", "long");
    case at::ScalarType::Short:
      return std::make_pair("int16", "short");
    case at::ScalarType::Half:
      return std::make_pair("float16", "half");
    case at::ScalarType::ComplexHalf:
      return std::make_pair("complex32", "");
    case at::ScalarType::ComplexFloat:
      return std::make_pair("complex64", "cfloat");
    case at::ScalarType::ComplexDouble:
      return std::make_pair("complex128", "cdouble");
    case at::ScalarType::Bool:
      return std::make_pair("bool", "");
    case at::ScalarType::QInt8:
      return std::make_pair("qint8", "");
    case at::ScalarType::QUInt8:
      return std::make_pair("quint8", "");
    case at::ScalarType::QInt32:
      return std::make_pair("qint32", "");
    case at::ScalarType::BFloat16:
      return std::make_pair("bfloat16", "");
    case at::ScalarType::QUInt4x2:
      return std::make_pair("quint4x2", "");
    default:
      throw std::runtime_error("Unimplemented scalar type");
  }
}

void initializeDtypes() {
  auto torch_module = THPObjectPtr(PyImport_ImportModule("torch"));
  if (!torch_module)
    throw python_error();

#define DEFINE_SCALAR_TYPE(_1, n) at::ScalarType::n,

  at::ScalarType all_scalar_types[] = {
      AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(DEFINE_SCALAR_TYPE)};

  for (at::ScalarType scalarType : all_scalar_types) {
    std::string primary_name, legacy_name;
    std::tie(primary_name, legacy_name) = getDtypeNames(scalarType);
    PyObject *dtype = THPDtype_New(scalarType, primary_name);
    torch::registerDtypeObject((THPDtype*)dtype, scalarType);
    Py_INCREF(dtype);
    if (PyModule_AddObject(torch_module.get(), primary_name.c_str(), dtype) !=
        0) {
      throw python_error();
    }
    if (legacy_name != "") {
      Py_INCREF(dtype);
      if (PyModule_AddObject(torch_module.get(), legacy_name.c_str(), dtype) !=
          0) {
        throw python_error();
      }
    }
  }
}

} // namespace utils
} // namespace torch
