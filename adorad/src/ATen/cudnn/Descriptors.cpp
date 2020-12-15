#include <ATen/cudnn/Descriptors.h>

#include <ATen/ATen.h>

#include <ostream>
#include <sstream>

namespace at { namespace native {

namespace {

inline cudnnDataType_t getDataType(const at::Tensor& t) {
  auto scalar_type = t.scalar_type();
  if (scalar_type == at::kFloat) {
    return CUDNN_DATA_FLOAT;
  } else if (scalar_type == at::kHalf) {
    return CUDNN_DATA_HALF;
  } else if (scalar_type == at::kDouble) {
    return CUDNN_DATA_DOUBLE;
  }
  throw std::runtime_error("TensorDescriptor only supports double, float and half tensors");
}

} // anonymous namespace


void TensorDescriptor::set(const at::Tensor &t, size_t pad) {
  set(getDataType(t), t.sizes(), t.strides(), pad, t.suggest_memory_format() == at::MemoryFormat::ChannelsLast);
}

void TensorDescriptor::set(cudnnDataType_t datatype, IntArrayRef t_sizes, IntArrayRef t_strides, size_t pad) {
  set(datatype, t_sizes, t_strides, pad, is_channels_last_strides_2d(t_sizes, t_strides));
}

void TensorDescriptor::set(cudnnDataType_t datatype, IntArrayRef t_sizes, IntArrayRef t_strides, size_t pad, bool nhwc) {
  size_t dim = t_sizes.size();
  if (dim > CUDNN_DIM_MAX || pad > CUDNN_DIM_MAX)
#define _STR(X) #X
#define STR(X) _STR(X)
    throw std::runtime_error("cuDNN supports only up to " STR(CUDNN_DIM_MAX) " dimensions");
#undef _STR
#undef STR
  int size[CUDNN_DIM_MAX];
  int stride[CUDNN_DIM_MAX];
  for (size_t i = 0; i < dim; ++i) {
    size[i] = static_cast<int>(t_sizes[i]);
    stride[i] = static_cast<int>(t_strides[i]);
  }
  for (size_t i = dim; i < pad; ++i) {
    size[i] = 1;
    stride[i] = 1;
  }
  set(datatype, static_cast<int>(std::max(dim, pad)), size, stride, nhwc);
}

std::string cudnnTypeToString(cudnnDataType_t dtype) {
  switch (dtype) {
    case CUDNN_DATA_FLOAT:
      return "CUDNN_DATA_FLOAT";
    case CUDNN_DATA_DOUBLE:
      return "CUDNN_DATA_DOUBLE";
    case CUDNN_DATA_HALF:
      return "CUDNN_DATA_HALF";
    case CUDNN_DATA_INT8:
      return "CUDNN_DATA_INT8";
    case CUDNN_DATA_INT32:
      return "CUDNN_DATA_INT32";
    case CUDNN_DATA_INT8x4:
      return "CUDNN_DATA_INT8x4";
#if CUDNN_VERSION >= 7100
    case CUDNN_DATA_UINT8:
      return "CUDNN_DATA_UINT8";
    case CUDNN_DATA_UINT8x4:
      return "CUDNN_DATA_UINT8x4";
#endif
    default:
      std::ostringstream oss;
      oss << "(unknown data-type " << static_cast<int>(dtype) << ")";
      return oss.str();
  }
}

std::ostream& operator<<(std::ostream & out, const TensorDescriptor& d) {
  out << "TensorDescriptor " << static_cast<void*>(d.desc()) << "\n";
  int nbDims;
  int dimA[CUDNN_DIM_MAX];
  int strideA[CUDNN_DIM_MAX];
  cudnnDataType_t dtype;
  cudnnGetTensorNdDescriptor(d.desc(), CUDNN_DIM_MAX, &dtype, &nbDims, dimA, strideA);
  out << "    type = " << cudnnTypeToString(dtype) << "\n";
  out << "    nbDims = " << nbDims << "\n";
  // Read out only nbDims of the arrays!
  out << "    dimA = ";
  for (auto i : ArrayRef<int>{dimA, static_cast<size_t>(nbDims)}) {
    out << i << ", ";
  }
  out << "\n";
  out << "    strideA = ";
  for (auto i : ArrayRef<int>{strideA, static_cast<size_t>(nbDims)}) {
    out << i << ", ";
  }
  out << "\n";
  return out;
}

void TensorDescriptor::print() { std::cout << *this; }

void FilterDescriptor::set(const at::Tensor &t, int64_t pad, bool force_nhwc) {
  auto dim = t.ndimension();
  if (dim > CUDNN_DIM_MAX || pad > CUDNN_DIM_MAX)
#define _STR(X) #X
#define STR(X) _STR(X)
    throw std::runtime_error("cuDNN supports only up to " STR(CUDNN_DIM_MAX) " dimensions");
#undef _STR
#undef STR
  auto memory_format = force_nhwc ? at::MemoryFormat::ChannelsLast : t.suggest_memory_format();
  // NB: It is possible for this test to be insufficient, because the
  // Tensor passed in to set the filter descriptor may not be the actual
  // Tensor whose data pointer is passed to cuDNN.  Nevertheless,
  // that is the common case, so we can catch most client errors with this test.
  TORCH_CHECK(t.is_contiguous(memory_format),
      "cuDNN filters (a.k.a. weights) must be contiguous in desired memory_format");

  int size[CUDNN_DIM_MAX];
  for (int i = 0; i < dim; ++i) {
    size[i] = (int) t.size(i);
  }
  for (int i = dim; i < pad; ++i) {
    size[i] = (int) 1;
  }
  dim = std::max(dim, pad);
  cudnnTensorFormat_t filter_format;
  switch(memory_format) {
  case at::MemoryFormat::Contiguous:
    filter_format = CUDNN_TENSOR_NCHW;
    break;
  case at::MemoryFormat::ChannelsLast:
    filter_format = CUDNN_TENSOR_NHWC;
    break;
  default:
    TORCH_INTERNAL_ASSERT(false, "unsurpported memory_format for cuDNN filters");
  }
  set(getDataType(t), (int) dim, size, filter_format);
}

std::string cudnnMemoryFormatToString(cudnnTensorFormat_t tformat) {
  switch (tformat) {
    case CUDNN_TENSOR_NCHW:
      return "CUDNN_TENSOR_NCHW";
    case CUDNN_TENSOR_NHWC:
      return "CUDNN_TENSOR_NHWC";
    default:
      std::ostringstream oss;
      oss << "(unknown cudnn tensor format " << static_cast<int>(tformat) << ")";
      return oss.str();
  }
}

std::ostream& operator<<(std::ostream & out, const FilterDescriptor& d) {
  out << "FilterDescriptor " << static_cast<void*>(d.desc()) << "\n";
  int nbDims;
  int dimA[CUDNN_DIM_MAX];
  cudnnDataType_t dtype;
  cudnnTensorFormat_t tformat;
  cudnnGetFilterNdDescriptor(d.desc(), CUDNN_DIM_MAX, &dtype, &tformat, &nbDims, dimA);
  out << "    type = " << cudnnTypeToString(dtype) << "\n";
  out << "    tensor_format = " << cudnnMemoryFormatToString(tformat) << "\n";
  out << "    nbDims = " << nbDims << "\n";
  // Read out only nbDims of the arrays!
  out << "    dimA = ";
  for (auto i : ArrayRef<int>{dimA, static_cast<size_t>(nbDims)}) {
    out << i << ", ";
  }
  out << "\n";
  return out;
}

void FilterDescriptor::print() { std::cout << *this; }

}}
