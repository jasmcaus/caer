#include <ATen/LegacyTHFunctionsCPU.h>

#include <ATen/ATen.h>
#include <ATen/Utils.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/CPUGeneratorImpl.h>
#include <ATen/ExpandUtils.h>
#include <TH/TH.h>
#include <TH/THTensor.hpp>


namespace at {
namespace native {
namespace legacy {
namespace cpu {

namespace {
  ScalarType infer_scalar_type(const Tensor & t) {
    return t.scalar_type();
  }
  ScalarType infer_scalar_type(const TensorList & tl) {
    TORCH_CHECK(tl.size() > 0, "expected a non-empty list of Tensors");
    return tl[0].scalar_type();
  }

  TensorOptions options(ScalarType s) {
    return TensorOptions().dtype(s)
                          .device(DeviceType::CPU)
                          .layout(kStrided);
  }

  Allocator* allocator() {
    return getCPUAllocator();
  }
}

Tensor & _th_masked_scatter_(Tensor & self, const Tensor & mask, const Tensor & source) {
    // DeviceGuard omitted
    auto dispatch_scalar_type = infer_scalar_type(self);

    switch (dispatch_scalar_type) {
        case ScalarType::Bool: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_masked_scatter_", false, DeviceType::CPU, dispatch_scalar_type);
            auto mask_ = checked_dense_tensor_unwrap(mask, "mask", 2, "_th_masked_scatter_", false, DeviceType::CPU, ScalarType::Byte);
            auto source_ = checked_dense_tensor_unwrap(source, "source", 3, "_th_masked_scatter_", false, DeviceType::CPU, dispatch_scalar_type);
            THBoolTensor_maskedCopy(self_, mask_, source_);
            break;
        }
        case ScalarType::Byte: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_masked_scatter_", false, DeviceType::CPU, dispatch_scalar_type);
            auto mask_ = checked_dense_tensor_unwrap(mask, "mask", 2, "_th_masked_scatter_", false, DeviceType::CPU, ScalarType::Byte);
            auto source_ = checked_dense_tensor_unwrap(source, "source", 3, "_th_masked_scatter_", false, DeviceType::CPU, dispatch_scalar_type);
            THByteTensor_maskedCopy(self_, mask_, source_);
            break;
        }
        case ScalarType::Char: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_masked_scatter_", false, DeviceType::CPU, dispatch_scalar_type);
            auto mask_ = checked_dense_tensor_unwrap(mask, "mask", 2, "_th_masked_scatter_", false, DeviceType::CPU, ScalarType::Byte);
            auto source_ = checked_dense_tensor_unwrap(source, "source", 3, "_th_masked_scatter_", false, DeviceType::CPU, dispatch_scalar_type);
            THCharTensor_maskedCopy(self_, mask_, source_);
            break;
        }
        case ScalarType::Double: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_masked_scatter_", false, DeviceType::CPU, dispatch_scalar_type);
            auto mask_ = checked_dense_tensor_unwrap(mask, "mask", 2, "_th_masked_scatter_", false, DeviceType::CPU, ScalarType::Byte);
            auto source_ = checked_dense_tensor_unwrap(source, "source", 3, "_th_masked_scatter_", false, DeviceType::CPU, dispatch_scalar_type);
            THDoubleTensor_maskedCopy(self_, mask_, source_);
            break;
        }
        case ScalarType::Float: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_masked_scatter_", false, DeviceType::CPU, dispatch_scalar_type);
            auto mask_ = checked_dense_tensor_unwrap(mask, "mask", 2, "_th_masked_scatter_", false, DeviceType::CPU, ScalarType::Byte);
            auto source_ = checked_dense_tensor_unwrap(source, "source", 3, "_th_masked_scatter_", false, DeviceType::CPU, dispatch_scalar_type);
            THFloatTensor_maskedCopy(self_, mask_, source_);
            break;
        }
        case ScalarType::Int: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_masked_scatter_", false, DeviceType::CPU, dispatch_scalar_type);
            auto mask_ = checked_dense_tensor_unwrap(mask, "mask", 2, "_th_masked_scatter_", false, DeviceType::CPU, ScalarType::Byte);
            auto source_ = checked_dense_tensor_unwrap(source, "source", 3, "_th_masked_scatter_", false, DeviceType::CPU, dispatch_scalar_type);
            THIntTensor_maskedCopy(self_, mask_, source_);
            break;
        }
        case ScalarType::Long: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_masked_scatter_", false, DeviceType::CPU, dispatch_scalar_type);
            auto mask_ = checked_dense_tensor_unwrap(mask, "mask", 2, "_th_masked_scatter_", false, DeviceType::CPU, ScalarType::Byte);
            auto source_ = checked_dense_tensor_unwrap(source, "source", 3, "_th_masked_scatter_", false, DeviceType::CPU, dispatch_scalar_type);
            THLongTensor_maskedCopy(self_, mask_, source_);
            break;
        }
        case ScalarType::Short: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_masked_scatter_", false, DeviceType::CPU, dispatch_scalar_type);
            auto mask_ = checked_dense_tensor_unwrap(mask, "mask", 2, "_th_masked_scatter_", false, DeviceType::CPU, ScalarType::Byte);
            auto source_ = checked_dense_tensor_unwrap(source, "source", 3, "_th_masked_scatter_", false, DeviceType::CPU, dispatch_scalar_type);
            THShortTensor_maskedCopy(self_, mask_, source_);
            break;
        }
        case ScalarType::BFloat16: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_masked_scatter_", false, DeviceType::CPU, dispatch_scalar_type);
            auto mask_ = checked_dense_tensor_unwrap(mask, "mask", 2, "_th_masked_scatter_", false, DeviceType::CPU, ScalarType::Byte);
            auto source_ = checked_dense_tensor_unwrap(source, "source", 3, "_th_masked_scatter_", false, DeviceType::CPU, dispatch_scalar_type);
            THBFloat16Tensor_maskedCopy(self_, mask_, source_);
            break;
        }
        default:
            AT_ERROR("_th_masked_scatter_ not supported on CPUType for ", dispatch_scalar_type);
    }
    return self;
}
Tensor & _th_masked_scatter_bool_(Tensor & self, const Tensor & mask, const Tensor & source) {
    // DeviceGuard omitted
    auto dispatch_scalar_type = infer_scalar_type(self);

    switch (dispatch_scalar_type) {
        case ScalarType::Bool: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_masked_scatter_bool_", false, DeviceType::CPU, dispatch_scalar_type);
            auto mask_ = checked_dense_tensor_unwrap(mask, "mask", 2, "_th_masked_scatter_bool_", false, DeviceType::CPU, ScalarType::Bool);
            auto source_ = checked_dense_tensor_unwrap(source, "source", 3, "_th_masked_scatter_bool_", false, DeviceType::CPU, dispatch_scalar_type);
            THBoolTensor_maskedCopyBool(self_, mask_, source_);
            break;
        }
        case ScalarType::Byte: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_masked_scatter_bool_", false, DeviceType::CPU, dispatch_scalar_type);
            auto mask_ = checked_dense_tensor_unwrap(mask, "mask", 2, "_th_masked_scatter_bool_", false, DeviceType::CPU, ScalarType::Bool);
            auto source_ = checked_dense_tensor_unwrap(source, "source", 3, "_th_masked_scatter_bool_", false, DeviceType::CPU, dispatch_scalar_type);
            THByteTensor_maskedCopyBool(self_, mask_, source_);
            break;
        }
        case ScalarType::Char: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_masked_scatter_bool_", false, DeviceType::CPU, dispatch_scalar_type);
            auto mask_ = checked_dense_tensor_unwrap(mask, "mask", 2, "_th_masked_scatter_bool_", false, DeviceType::CPU, ScalarType::Bool);
            auto source_ = checked_dense_tensor_unwrap(source, "source", 3, "_th_masked_scatter_bool_", false, DeviceType::CPU, dispatch_scalar_type);
            THCharTensor_maskedCopyBool(self_, mask_, source_);
            break;
        }
        case ScalarType::Double: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_masked_scatter_bool_", false, DeviceType::CPU, dispatch_scalar_type);
            auto mask_ = checked_dense_tensor_unwrap(mask, "mask", 2, "_th_masked_scatter_bool_", false, DeviceType::CPU, ScalarType::Bool);
            auto source_ = checked_dense_tensor_unwrap(source, "source", 3, "_th_masked_scatter_bool_", false, DeviceType::CPU, dispatch_scalar_type);
            THDoubleTensor_maskedCopyBool(self_, mask_, source_);
            break;
        }
        case ScalarType::Float: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_masked_scatter_bool_", false, DeviceType::CPU, dispatch_scalar_type);
            auto mask_ = checked_dense_tensor_unwrap(mask, "mask", 2, "_th_masked_scatter_bool_", false, DeviceType::CPU, ScalarType::Bool);
            auto source_ = checked_dense_tensor_unwrap(source, "source", 3, "_th_masked_scatter_bool_", false, DeviceType::CPU, dispatch_scalar_type);
            THFloatTensor_maskedCopyBool(self_, mask_, source_);
            break;
        }
        case ScalarType::Int: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_masked_scatter_bool_", false, DeviceType::CPU, dispatch_scalar_type);
            auto mask_ = checked_dense_tensor_unwrap(mask, "mask", 2, "_th_masked_scatter_bool_", false, DeviceType::CPU, ScalarType::Bool);
            auto source_ = checked_dense_tensor_unwrap(source, "source", 3, "_th_masked_scatter_bool_", false, DeviceType::CPU, dispatch_scalar_type);
            THIntTensor_maskedCopyBool(self_, mask_, source_);
            break;
        }
        case ScalarType::Long: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_masked_scatter_bool_", false, DeviceType::CPU, dispatch_scalar_type);
            auto mask_ = checked_dense_tensor_unwrap(mask, "mask", 2, "_th_masked_scatter_bool_", false, DeviceType::CPU, ScalarType::Bool);
            auto source_ = checked_dense_tensor_unwrap(source, "source", 3, "_th_masked_scatter_bool_", false, DeviceType::CPU, dispatch_scalar_type);
            THLongTensor_maskedCopyBool(self_, mask_, source_);
            break;
        }
        case ScalarType::Short: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_masked_scatter_bool_", false, DeviceType::CPU, dispatch_scalar_type);
            auto mask_ = checked_dense_tensor_unwrap(mask, "mask", 2, "_th_masked_scatter_bool_", false, DeviceType::CPU, ScalarType::Bool);
            auto source_ = checked_dense_tensor_unwrap(source, "source", 3, "_th_masked_scatter_bool_", false, DeviceType::CPU, dispatch_scalar_type);
            THShortTensor_maskedCopyBool(self_, mask_, source_);
            break;
        }
        case ScalarType::BFloat16: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_masked_scatter_bool_", false, DeviceType::CPU, dispatch_scalar_type);
            auto mask_ = checked_dense_tensor_unwrap(mask, "mask", 2, "_th_masked_scatter_bool_", false, DeviceType::CPU, ScalarType::Bool);
            auto source_ = checked_dense_tensor_unwrap(source, "source", 3, "_th_masked_scatter_bool_", false, DeviceType::CPU, dispatch_scalar_type);
            THBFloat16Tensor_maskedCopyBool(self_, mask_, source_);
            break;
        }
        default:
            AT_ERROR("_th_masked_scatter_bool_ not supported on CPUType for ", dispatch_scalar_type);
    }
    return self;
}
Tensor & _th_nonzero_out(Tensor & result, const Tensor & self) {
    // DeviceGuard omitted
    auto dispatch_scalar_type = infer_scalar_type(self);

    switch (dispatch_scalar_type) {
        case ScalarType::Bool: {
            auto result_ = checked_dense_tensor_unwrap(result, "result", 0, "_th_nonzero_out", false, DeviceType::CPU, ScalarType::Long);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_nonzero_out", false, DeviceType::CPU, dispatch_scalar_type);
            THBoolTensor_nonzero(result_, self_);
            break;
        }
        case ScalarType::Byte: {
            auto result_ = checked_dense_tensor_unwrap(result, "result", 0, "_th_nonzero_out", false, DeviceType::CPU, ScalarType::Long);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_nonzero_out", false, DeviceType::CPU, dispatch_scalar_type);
            THByteTensor_nonzero(result_, self_);
            break;
        }
        case ScalarType::Char: {
            auto result_ = checked_dense_tensor_unwrap(result, "result", 0, "_th_nonzero_out", false, DeviceType::CPU, ScalarType::Long);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_nonzero_out", false, DeviceType::CPU, dispatch_scalar_type);
            THCharTensor_nonzero(result_, self_);
            break;
        }
        case ScalarType::Double: {
            auto result_ = checked_dense_tensor_unwrap(result, "result", 0, "_th_nonzero_out", false, DeviceType::CPU, ScalarType::Long);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_nonzero_out", false, DeviceType::CPU, dispatch_scalar_type);
            THDoubleTensor_nonzero(result_, self_);
            break;
        }
        case ScalarType::Float: {
            auto result_ = checked_dense_tensor_unwrap(result, "result", 0, "_th_nonzero_out", false, DeviceType::CPU, ScalarType::Long);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_nonzero_out", false, DeviceType::CPU, dispatch_scalar_type);
            THFloatTensor_nonzero(result_, self_);
            break;
        }
        case ScalarType::Int: {
            auto result_ = checked_dense_tensor_unwrap(result, "result", 0, "_th_nonzero_out", false, DeviceType::CPU, ScalarType::Long);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_nonzero_out", false, DeviceType::CPU, dispatch_scalar_type);
            THIntTensor_nonzero(result_, self_);
            break;
        }
        case ScalarType::Long: {
            auto result_ = checked_dense_tensor_unwrap(result, "result", 0, "_th_nonzero_out", false, DeviceType::CPU, ScalarType::Long);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_nonzero_out", false, DeviceType::CPU, dispatch_scalar_type);
            THLongTensor_nonzero(result_, self_);
            break;
        }
        case ScalarType::Short: {
            auto result_ = checked_dense_tensor_unwrap(result, "result", 0, "_th_nonzero_out", false, DeviceType::CPU, ScalarType::Long);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_nonzero_out", false, DeviceType::CPU, dispatch_scalar_type);
            THShortTensor_nonzero(result_, self_);
            break;
        }
        case ScalarType::Half: {
            auto result_ = checked_dense_tensor_unwrap(result, "result", 0, "_th_nonzero_out", false, DeviceType::CPU, ScalarType::Long);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_nonzero_out", false, DeviceType::CPU, dispatch_scalar_type);
            THHalfTensor_nonzero(result_, self_);
            break;
        }
        case ScalarType::BFloat16: {
            auto result_ = checked_dense_tensor_unwrap(result, "result", 0, "_th_nonzero_out", false, DeviceType::CPU, ScalarType::Long);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_nonzero_out", false, DeviceType::CPU, dispatch_scalar_type);
            THBFloat16Tensor_nonzero(result_, self_);
            break;
        }
        default:
            AT_ERROR("_th_nonzero_out not supported on CPUType for ", dispatch_scalar_type);
    }
    return result;
}
Tensor _th_nonzero(const Tensor & self) {
    // DeviceGuard omitted
    auto dispatch_scalar_type = infer_scalar_type(self);
    auto result_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(c10::Storage::use_byte_size_t(), 0, allocator(), true),DispatchKey::CPU, scalarTypeToTypeMeta(ScalarType::Long)).release();
    auto result = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(result_));
    switch (dispatch_scalar_type) {
        case ScalarType::Bool: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_nonzero", false, DeviceType::CPU, dispatch_scalar_type);
            THBoolTensor_nonzero(result_, self_);
            break;
        }
        case ScalarType::Byte: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_nonzero", false, DeviceType::CPU, dispatch_scalar_type);
            THByteTensor_nonzero(result_, self_);
            break;
        }
        case ScalarType::Char: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_nonzero", false, DeviceType::CPU, dispatch_scalar_type);
            THCharTensor_nonzero(result_, self_);
            break;
        }
        case ScalarType::Double: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_nonzero", false, DeviceType::CPU, dispatch_scalar_type);
            THDoubleTensor_nonzero(result_, self_);
            break;
        }
        case ScalarType::Float: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_nonzero", false, DeviceType::CPU, dispatch_scalar_type);
            THFloatTensor_nonzero(result_, self_);
            break;
        }
        case ScalarType::Int: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_nonzero", false, DeviceType::CPU, dispatch_scalar_type);
            THIntTensor_nonzero(result_, self_);
            break;
        }
        case ScalarType::Long: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_nonzero", false, DeviceType::CPU, dispatch_scalar_type);
            THLongTensor_nonzero(result_, self_);
            break;
        }
        case ScalarType::Short: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_nonzero", false, DeviceType::CPU, dispatch_scalar_type);
            THShortTensor_nonzero(result_, self_);
            break;
        }
        case ScalarType::Half: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_nonzero", false, DeviceType::CPU, dispatch_scalar_type);
            THHalfTensor_nonzero(result_, self_);
            break;
        }
        case ScalarType::BFloat16: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_nonzero", false, DeviceType::CPU, dispatch_scalar_type);
            THBFloat16Tensor_nonzero(result_, self_);
            break;
        }
        default:
            AT_ERROR("_th_nonzero not supported on CPUType for ", dispatch_scalar_type);
    }
    return result;
}
Tensor & _th_index_copy_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) {
    // DeviceGuard omitted
    auto dispatch_scalar_type = infer_scalar_type(self);

    switch (dispatch_scalar_type) {
        case ScalarType::Bool: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_index_copy_", false, DeviceType::CPU, dispatch_scalar_type);
            auto index_ = checked_dense_tensor_unwrap(index, "index", 3, "_th_index_copy_", false, DeviceType::CPU, ScalarType::Long);
            auto source_ = checked_dense_tensor_unwrap(source, "source", 4, "_th_index_copy_", false, DeviceType::CPU, dispatch_scalar_type);
            THBoolTensor_indexCopy(self_, dim, index_, source_);
            break;
        }
        case ScalarType::Byte: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_index_copy_", false, DeviceType::CPU, dispatch_scalar_type);
            auto index_ = checked_dense_tensor_unwrap(index, "index", 3, "_th_index_copy_", false, DeviceType::CPU, ScalarType::Long);
            auto source_ = checked_dense_tensor_unwrap(source, "source", 4, "_th_index_copy_", false, DeviceType::CPU, dispatch_scalar_type);
            THByteTensor_indexCopy(self_, dim, index_, source_);
            break;
        }
        case ScalarType::Char: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_index_copy_", false, DeviceType::CPU, dispatch_scalar_type);
            auto index_ = checked_dense_tensor_unwrap(index, "index", 3, "_th_index_copy_", false, DeviceType::CPU, ScalarType::Long);
            auto source_ = checked_dense_tensor_unwrap(source, "source", 4, "_th_index_copy_", false, DeviceType::CPU, dispatch_scalar_type);
            THCharTensor_indexCopy(self_, dim, index_, source_);
            break;
        }
        case ScalarType::Double: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_index_copy_", false, DeviceType::CPU, dispatch_scalar_type);
            auto index_ = checked_dense_tensor_unwrap(index, "index", 3, "_th_index_copy_", false, DeviceType::CPU, ScalarType::Long);
            auto source_ = checked_dense_tensor_unwrap(source, "source", 4, "_th_index_copy_", false, DeviceType::CPU, dispatch_scalar_type);
            THDoubleTensor_indexCopy(self_, dim, index_, source_);
            break;
        }
        case ScalarType::Float: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_index_copy_", false, DeviceType::CPU, dispatch_scalar_type);
            auto index_ = checked_dense_tensor_unwrap(index, "index", 3, "_th_index_copy_", false, DeviceType::CPU, ScalarType::Long);
            auto source_ = checked_dense_tensor_unwrap(source, "source", 4, "_th_index_copy_", false, DeviceType::CPU, dispatch_scalar_type);
            THFloatTensor_indexCopy(self_, dim, index_, source_);
            break;
        }
        case ScalarType::Int: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_index_copy_", false, DeviceType::CPU, dispatch_scalar_type);
            auto index_ = checked_dense_tensor_unwrap(index, "index", 3, "_th_index_copy_", false, DeviceType::CPU, ScalarType::Long);
            auto source_ = checked_dense_tensor_unwrap(source, "source", 4, "_th_index_copy_", false, DeviceType::CPU, dispatch_scalar_type);
            THIntTensor_indexCopy(self_, dim, index_, source_);
            break;
        }
        case ScalarType::Long: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_index_copy_", false, DeviceType::CPU, dispatch_scalar_type);
            auto index_ = checked_dense_tensor_unwrap(index, "index", 3, "_th_index_copy_", false, DeviceType::CPU, ScalarType::Long);
            auto source_ = checked_dense_tensor_unwrap(source, "source", 4, "_th_index_copy_", false, DeviceType::CPU, dispatch_scalar_type);
            THLongTensor_indexCopy(self_, dim, index_, source_);
            break;
        }
        case ScalarType::Short: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_index_copy_", false, DeviceType::CPU, dispatch_scalar_type);
            auto index_ = checked_dense_tensor_unwrap(index, "index", 3, "_th_index_copy_", false, DeviceType::CPU, ScalarType::Long);
            auto source_ = checked_dense_tensor_unwrap(source, "source", 4, "_th_index_copy_", false, DeviceType::CPU, dispatch_scalar_type);
            THShortTensor_indexCopy(self_, dim, index_, source_);
            break;
        }
        default:
            AT_ERROR("_th_index_copy_ not supported on CPUType for ", dispatch_scalar_type);
    }
    return self;
}
Tensor & _th_put_(Tensor & self, const Tensor & index, const Tensor & source, bool accumulate) {
    // DeviceGuard omitted
    auto dispatch_scalar_type = infer_scalar_type(self);

    switch (dispatch_scalar_type) {
        case ScalarType::Bool: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_put_", false, DeviceType::CPU, dispatch_scalar_type);
            auto index_ = checked_dense_tensor_unwrap(index, "index", 2, "_th_put_", false, DeviceType::CPU, ScalarType::Long);
            auto source_ = checked_dense_tensor_unwrap(source, "source", 3, "_th_put_", false, DeviceType::CPU, dispatch_scalar_type);
            THBoolTensor_put(self_, index_, source_, accumulate);
            break;
        }
        case ScalarType::Byte: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_put_", false, DeviceType::CPU, dispatch_scalar_type);
            auto index_ = checked_dense_tensor_unwrap(index, "index", 2, "_th_put_", false, DeviceType::CPU, ScalarType::Long);
            auto source_ = checked_dense_tensor_unwrap(source, "source", 3, "_th_put_", false, DeviceType::CPU, dispatch_scalar_type);
            THByteTensor_put(self_, index_, source_, accumulate);
            break;
        }
        case ScalarType::Char: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_put_", false, DeviceType::CPU, dispatch_scalar_type);
            auto index_ = checked_dense_tensor_unwrap(index, "index", 2, "_th_put_", false, DeviceType::CPU, ScalarType::Long);
            auto source_ = checked_dense_tensor_unwrap(source, "source", 3, "_th_put_", false, DeviceType::CPU, dispatch_scalar_type);
            THCharTensor_put(self_, index_, source_, accumulate);
            break;
        }
        case ScalarType::Double: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_put_", false, DeviceType::CPU, dispatch_scalar_type);
            auto index_ = checked_dense_tensor_unwrap(index, "index", 2, "_th_put_", false, DeviceType::CPU, ScalarType::Long);
            auto source_ = checked_dense_tensor_unwrap(source, "source", 3, "_th_put_", false, DeviceType::CPU, dispatch_scalar_type);
            THDoubleTensor_put(self_, index_, source_, accumulate);
            break;
        }
        case ScalarType::Float: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_put_", false, DeviceType::CPU, dispatch_scalar_type);
            auto index_ = checked_dense_tensor_unwrap(index, "index", 2, "_th_put_", false, DeviceType::CPU, ScalarType::Long);
            auto source_ = checked_dense_tensor_unwrap(source, "source", 3, "_th_put_", false, DeviceType::CPU, dispatch_scalar_type);
            THFloatTensor_put(self_, index_, source_, accumulate);
            break;
        }
        case ScalarType::Int: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_put_", false, DeviceType::CPU, dispatch_scalar_type);
            auto index_ = checked_dense_tensor_unwrap(index, "index", 2, "_th_put_", false, DeviceType::CPU, ScalarType::Long);
            auto source_ = checked_dense_tensor_unwrap(source, "source", 3, "_th_put_", false, DeviceType::CPU, dispatch_scalar_type);
            THIntTensor_put(self_, index_, source_, accumulate);
            break;
        }
        case ScalarType::Long: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_put_", false, DeviceType::CPU, dispatch_scalar_type);
            auto index_ = checked_dense_tensor_unwrap(index, "index", 2, "_th_put_", false, DeviceType::CPU, ScalarType::Long);
            auto source_ = checked_dense_tensor_unwrap(source, "source", 3, "_th_put_", false, DeviceType::CPU, dispatch_scalar_type);
            THLongTensor_put(self_, index_, source_, accumulate);
            break;
        }
        case ScalarType::Short: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_put_", false, DeviceType::CPU, dispatch_scalar_type);
            auto index_ = checked_dense_tensor_unwrap(index, "index", 2, "_th_put_", false, DeviceType::CPU, ScalarType::Long);
            auto source_ = checked_dense_tensor_unwrap(source, "source", 3, "_th_put_", false, DeviceType::CPU, dispatch_scalar_type);
            THShortTensor_put(self_, index_, source_, accumulate);
            break;
        }
        default:
            AT_ERROR("_th_put_ not supported on CPUType for ", dispatch_scalar_type);
    }
    return self;
}
Tensor & _th_index_fill_(Tensor & self, int64_t dim, const Tensor & index, Scalar value) {
    // DeviceGuard omitted
    auto dispatch_scalar_type = infer_scalar_type(self);

    switch (dispatch_scalar_type) {
        case ScalarType::Bool: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_index_fill_", false, DeviceType::CPU, dispatch_scalar_type);
            auto index_ = checked_dense_tensor_unwrap(index, "index", 3, "_th_index_fill_", false, DeviceType::CPU, ScalarType::Long);
            auto value_ = value.toBool();
            THBoolTensor_indexFill(self_, dim, index_, value_);
            break;
        }
        case ScalarType::Byte: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_index_fill_", false, DeviceType::CPU, dispatch_scalar_type);
            auto index_ = checked_dense_tensor_unwrap(index, "index", 3, "_th_index_fill_", false, DeviceType::CPU, ScalarType::Long);
            auto value_ = value.toByte();
            THByteTensor_indexFill(self_, dim, index_, value_);
            break;
        }
        case ScalarType::Char: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_index_fill_", false, DeviceType::CPU, dispatch_scalar_type);
            auto index_ = checked_dense_tensor_unwrap(index, "index", 3, "_th_index_fill_", false, DeviceType::CPU, ScalarType::Long);
            auto value_ = value.toChar();
            THCharTensor_indexFill(self_, dim, index_, value_);
            break;
        }
        case ScalarType::Double: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_index_fill_", false, DeviceType::CPU, dispatch_scalar_type);
            auto index_ = checked_dense_tensor_unwrap(index, "index", 3, "_th_index_fill_", false, DeviceType::CPU, ScalarType::Long);
            auto value_ = value.toDouble();
            THDoubleTensor_indexFill(self_, dim, index_, value_);
            break;
        }
        case ScalarType::Float: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_index_fill_", false, DeviceType::CPU, dispatch_scalar_type);
            auto index_ = checked_dense_tensor_unwrap(index, "index", 3, "_th_index_fill_", false, DeviceType::CPU, ScalarType::Long);
            auto value_ = value.toFloat();
            THFloatTensor_indexFill(self_, dim, index_, value_);
            break;
        }
        case ScalarType::Int: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_index_fill_", false, DeviceType::CPU, dispatch_scalar_type);
            auto index_ = checked_dense_tensor_unwrap(index, "index", 3, "_th_index_fill_", false, DeviceType::CPU, ScalarType::Long);
            auto value_ = value.toInt();
            THIntTensor_indexFill(self_, dim, index_, value_);
            break;
        }
        case ScalarType::Long: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_index_fill_", false, DeviceType::CPU, dispatch_scalar_type);
            auto index_ = checked_dense_tensor_unwrap(index, "index", 3, "_th_index_fill_", false, DeviceType::CPU, ScalarType::Long);
            auto value_ = value.toLong();
            THLongTensor_indexFill(self_, dim, index_, value_);
            break;
        }
        case ScalarType::Short: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_index_fill_", false, DeviceType::CPU, dispatch_scalar_type);
            auto index_ = checked_dense_tensor_unwrap(index, "index", 3, "_th_index_fill_", false, DeviceType::CPU, ScalarType::Long);
            auto value_ = value.toShort();
            THShortTensor_indexFill(self_, dim, index_, value_);
            break;
        }
        default:
            AT_ERROR("_th_index_fill_ not supported on CPUType for ", dispatch_scalar_type);
    }
    return self;
}
std::tuple<Tensor &,Tensor &> _th_mode_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim, bool keepdim) {
    // DeviceGuard omitted
    auto dispatch_scalar_type = infer_scalar_type(self);

    switch (dispatch_scalar_type) {
        case ScalarType::Byte: {
            auto values_ = checked_dense_tensor_unwrap(values, "values", 0, "_th_mode_out", false, DeviceType::CPU, dispatch_scalar_type);
            auto indices_ = checked_dense_tensor_unwrap(indices, "indices", 0, "_th_mode_out", false, DeviceType::CPU, ScalarType::Long);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_mode_out", false, DeviceType::CPU, dispatch_scalar_type);
            THByteTensor_mode(values_, indices_, self_, dim, keepdim);
            break;
        }
        case ScalarType::Char: {
            auto values_ = checked_dense_tensor_unwrap(values, "values", 0, "_th_mode_out", false, DeviceType::CPU, dispatch_scalar_type);
            auto indices_ = checked_dense_tensor_unwrap(indices, "indices", 0, "_th_mode_out", false, DeviceType::CPU, ScalarType::Long);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_mode_out", false, DeviceType::CPU, dispatch_scalar_type);
            THCharTensor_mode(values_, indices_, self_, dim, keepdim);
            break;
        }
        case ScalarType::Double: {
            auto values_ = checked_dense_tensor_unwrap(values, "values", 0, "_th_mode_out", false, DeviceType::CPU, dispatch_scalar_type);
            auto indices_ = checked_dense_tensor_unwrap(indices, "indices", 0, "_th_mode_out", false, DeviceType::CPU, ScalarType::Long);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_mode_out", false, DeviceType::CPU, dispatch_scalar_type);
            THDoubleTensor_mode(values_, indices_, self_, dim, keepdim);
            break;
        }
        case ScalarType::Float: {
            auto values_ = checked_dense_tensor_unwrap(values, "values", 0, "_th_mode_out", false, DeviceType::CPU, dispatch_scalar_type);
            auto indices_ = checked_dense_tensor_unwrap(indices, "indices", 0, "_th_mode_out", false, DeviceType::CPU, ScalarType::Long);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_mode_out", false, DeviceType::CPU, dispatch_scalar_type);
            THFloatTensor_mode(values_, indices_, self_, dim, keepdim);
            break;
        }
        case ScalarType::Int: {
            auto values_ = checked_dense_tensor_unwrap(values, "values", 0, "_th_mode_out", false, DeviceType::CPU, dispatch_scalar_type);
            auto indices_ = checked_dense_tensor_unwrap(indices, "indices", 0, "_th_mode_out", false, DeviceType::CPU, ScalarType::Long);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_mode_out", false, DeviceType::CPU, dispatch_scalar_type);
            THIntTensor_mode(values_, indices_, self_, dim, keepdim);
            break;
        }
        case ScalarType::Long: {
            auto values_ = checked_dense_tensor_unwrap(values, "values", 0, "_th_mode_out", false, DeviceType::CPU, dispatch_scalar_type);
            auto indices_ = checked_dense_tensor_unwrap(indices, "indices", 0, "_th_mode_out", false, DeviceType::CPU, ScalarType::Long);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_mode_out", false, DeviceType::CPU, dispatch_scalar_type);
            THLongTensor_mode(values_, indices_, self_, dim, keepdim);
            break;
        }
        case ScalarType::Short: {
            auto values_ = checked_dense_tensor_unwrap(values, "values", 0, "_th_mode_out", false, DeviceType::CPU, dispatch_scalar_type);
            auto indices_ = checked_dense_tensor_unwrap(indices, "indices", 0, "_th_mode_out", false, DeviceType::CPU, ScalarType::Long);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_mode_out", false, DeviceType::CPU, dispatch_scalar_type);
            THShortTensor_mode(values_, indices_, self_, dim, keepdim);
            break;
        }
        default:
            AT_ERROR("_th_mode_out not supported on CPUType for ", dispatch_scalar_type);
    }
    return std::tuple<Tensor &, Tensor &>(values, indices);
}
std::tuple<Tensor,Tensor> _th_mode(const Tensor & self, int64_t dim, bool keepdim) {
    // DeviceGuard omitted
    auto dispatch_scalar_type = infer_scalar_type(self);
    auto values_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(c10::Storage::use_byte_size_t(), 0, allocator(), true),DispatchKey::CPU, scalarTypeToTypeMeta(dispatch_scalar_type)).release();
    auto values = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(values_));
    auto indices_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(c10::Storage::use_byte_size_t(), 0, allocator(), true),DispatchKey::CPU, scalarTypeToTypeMeta(ScalarType::Long)).release();
    auto indices = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(indices_));
    switch (dispatch_scalar_type) {
        case ScalarType::Byte: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_mode", false, DeviceType::CPU, dispatch_scalar_type);
            THByteTensor_mode(values_, indices_, self_, dim, keepdim);
            break;
        }
        case ScalarType::Char: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_mode", false, DeviceType::CPU, dispatch_scalar_type);
            THCharTensor_mode(values_, indices_, self_, dim, keepdim);
            break;
        }
        case ScalarType::Double: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_mode", false, DeviceType::CPU, dispatch_scalar_type);
            THDoubleTensor_mode(values_, indices_, self_, dim, keepdim);
            break;
        }
        case ScalarType::Float: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_mode", false, DeviceType::CPU, dispatch_scalar_type);
            THFloatTensor_mode(values_, indices_, self_, dim, keepdim);
            break;
        }
        case ScalarType::Int: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_mode", false, DeviceType::CPU, dispatch_scalar_type);
            THIntTensor_mode(values_, indices_, self_, dim, keepdim);
            break;
        }
        case ScalarType::Long: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_mode", false, DeviceType::CPU, dispatch_scalar_type);
            THLongTensor_mode(values_, indices_, self_, dim, keepdim);
            break;
        }
        case ScalarType::Short: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_mode", false, DeviceType::CPU, dispatch_scalar_type);
            THShortTensor_mode(values_, indices_, self_, dim, keepdim);
            break;
        }
        default:
            AT_ERROR("_th_mode not supported on CPUType for ", dispatch_scalar_type);
    }
    return std::tuple<Tensor, Tensor>(values, indices);
}
Tensor _th_var(const Tensor & self, bool unbiased) {
    // DeviceGuard omitted
    auto dispatch_scalar_type = infer_scalar_type(self);

    switch (dispatch_scalar_type) {
        case ScalarType::Double: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_var", false, DeviceType::CPU, dispatch_scalar_type);
            return at::scalar_tensor(convert<double>(THDoubleTensor_var_all(self_, unbiased)), options(ScalarType::Double));
            break;
        }
        case ScalarType::Float: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_var", false, DeviceType::CPU, dispatch_scalar_type);
            return at::scalar_tensor(convert<float>(THFloatTensor_var_all(self_, unbiased)), options(ScalarType::Float));
            break;
        }
        default:
            AT_ERROR("_th_var not supported on CPUType for ", dispatch_scalar_type);
    }
}
Tensor _th_std(const Tensor & self, bool unbiased) {
    // DeviceGuard omitted
    auto dispatch_scalar_type = infer_scalar_type(self);

    switch (dispatch_scalar_type) {
        case ScalarType::Double: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_std", false, DeviceType::CPU, dispatch_scalar_type);
            return at::scalar_tensor(convert<double>(THDoubleTensor_std_all(self_, unbiased)), options(ScalarType::Double));
            break;
        }
        case ScalarType::Float: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_std", false, DeviceType::CPU, dispatch_scalar_type);
            return at::scalar_tensor(convert<float>(THFloatTensor_std_all(self_, unbiased)), options(ScalarType::Float));
            break;
        }
        default:
            AT_ERROR("_th_std not supported on CPUType for ", dispatch_scalar_type);
    }
}
Tensor & _th_renorm_out(Tensor & result, const Tensor & self, Scalar p, int64_t dim, Scalar maxnorm) {
    // DeviceGuard omitted
    auto dispatch_scalar_type = infer_scalar_type(self);

    switch (dispatch_scalar_type) {
        case ScalarType::Double: {
            auto result_ = checked_dense_tensor_unwrap(result, "result", 0, "_th_renorm_out", false, DeviceType::CPU, dispatch_scalar_type);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_renorm_out", false, DeviceType::CPU, dispatch_scalar_type);
            auto p_ = p.toDouble();
            auto maxnorm_ = maxnorm.toDouble();
            THDoubleTensor_renorm(result_, self_, p_, dim, maxnorm_);
            break;
        }
        case ScalarType::Float: {
            auto result_ = checked_dense_tensor_unwrap(result, "result", 0, "_th_renorm_out", false, DeviceType::CPU, dispatch_scalar_type);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_renorm_out", false, DeviceType::CPU, dispatch_scalar_type);
            auto p_ = p.toFloat();
            auto maxnorm_ = maxnorm.toFloat();
            THFloatTensor_renorm(result_, self_, p_, dim, maxnorm_);
            break;
        }
        default:
            AT_ERROR("_th_renorm_out not supported on CPUType for ", dispatch_scalar_type);
    }
    return result;
}
Tensor _th_renorm(const Tensor & self, Scalar p, int64_t dim, Scalar maxnorm) {
    // DeviceGuard omitted
    auto dispatch_scalar_type = infer_scalar_type(self);
    auto result_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(c10::Storage::use_byte_size_t(), 0, allocator(), true),DispatchKey::CPU, scalarTypeToTypeMeta(dispatch_scalar_type)).release();
    auto result = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(result_));
    switch (dispatch_scalar_type) {
        case ScalarType::Double: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_renorm", false, DeviceType::CPU, dispatch_scalar_type);
            auto p_ = p.toDouble();
            auto maxnorm_ = maxnorm.toDouble();
            THDoubleTensor_renorm(result_, self_, p_, dim, maxnorm_);
            break;
        }
        case ScalarType::Float: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_renorm", false, DeviceType::CPU, dispatch_scalar_type);
            auto p_ = p.toFloat();
            auto maxnorm_ = maxnorm.toFloat();
            THFloatTensor_renorm(result_, self_, p_, dim, maxnorm_);
            break;
        }
        default:
            AT_ERROR("_th_renorm not supported on CPUType for ", dispatch_scalar_type);
    }
    return result;
}
Tensor & _th_renorm_(Tensor & self, Scalar p, int64_t dim, Scalar maxnorm) {
    // DeviceGuard omitted
    auto dispatch_scalar_type = infer_scalar_type(self);

    switch (dispatch_scalar_type) {
        case ScalarType::Double: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_renorm_", false, DeviceType::CPU, dispatch_scalar_type);
            auto p_ = p.toDouble();
            auto maxnorm_ = maxnorm.toDouble();
            THDoubleTensor_renorm(self_, self_, p_, dim, maxnorm_);
            break;
        }
        case ScalarType::Float: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_renorm_", false, DeviceType::CPU, dispatch_scalar_type);
            auto p_ = p.toFloat();
            auto maxnorm_ = maxnorm.toFloat();
            THFloatTensor_renorm(self_, self_, p_, dim, maxnorm_);
            break;
        }
        default:
            AT_ERROR("_th_renorm_ not supported on CPUType for ", dispatch_scalar_type);
    }
    return self;
}
Tensor & _th_histc_out(Tensor & result, const Tensor & self, int64_t bins, Scalar min, Scalar max) {
    // DeviceGuard omitted
    auto dispatch_scalar_type = infer_scalar_type(self);

    switch (dispatch_scalar_type) {
        case ScalarType::Double: {
            auto result_ = checked_dense_tensor_unwrap(result, "result", 0, "_th_histc_out", false, DeviceType::CPU, dispatch_scalar_type);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_histc_out", false, DeviceType::CPU, dispatch_scalar_type);
            auto min_ = min.toDouble();
            auto max_ = max.toDouble();
            THDoubleTensor_histc(result_, self_, bins, min_, max_);
            break;
        }
        case ScalarType::Float: {
            auto result_ = checked_dense_tensor_unwrap(result, "result", 0, "_th_histc_out", false, DeviceType::CPU, dispatch_scalar_type);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_histc_out", false, DeviceType::CPU, dispatch_scalar_type);
            auto min_ = min.toFloat();
            auto max_ = max.toFloat();
            THFloatTensor_histc(result_, self_, bins, min_, max_);
            break;
        }
        default:
            AT_ERROR("_th_histc_out not supported on CPUType for ", dispatch_scalar_type);
    }
    return result;
}
Tensor _th_histc(const Tensor & self, int64_t bins, Scalar min, Scalar max) {
    // DeviceGuard omitted
    auto dispatch_scalar_type = infer_scalar_type(self);
    auto result_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(c10::Storage::use_byte_size_t(), 0, allocator(), true),DispatchKey::CPU, scalarTypeToTypeMeta(dispatch_scalar_type)).release();
    auto result = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(result_));
    switch (dispatch_scalar_type) {
        case ScalarType::Double: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_histc", false, DeviceType::CPU, dispatch_scalar_type);
            auto min_ = min.toDouble();
            auto max_ = max.toDouble();
            THDoubleTensor_histc(result_, self_, bins, min_, max_);
            break;
        }
        case ScalarType::Float: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_histc", false, DeviceType::CPU, dispatch_scalar_type);
            auto min_ = min.toFloat();
            auto max_ = max.toFloat();
            THFloatTensor_histc(result_, self_, bins, min_, max_);
            break;
        }
        default:
            AT_ERROR("_th_histc not supported on CPUType for ", dispatch_scalar_type);
    }
    return result;
}

std::tuple<Tensor &,Tensor &> _th_gels_out(Tensor & res1, Tensor & res2, const Tensor & self, const Tensor & A) {
    // DeviceGuard omitted
    auto dispatch_scalar_type = infer_scalar_type(self);

    switch (dispatch_scalar_type) {
        case ScalarType::Double: {
            auto res1_ = checked_dense_tensor_unwrap(res1, "res1", 0, "_th_gels_out", false, DeviceType::CPU, dispatch_scalar_type);
            auto res2_ = checked_dense_tensor_unwrap(res2, "res2", 0, "_th_gels_out", false, DeviceType::CPU, dispatch_scalar_type);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_gels_out", false, DeviceType::CPU, dispatch_scalar_type);
            auto A_ = checked_dense_tensor_unwrap(A, "A", 2, "_th_gels_out", false, DeviceType::CPU, dispatch_scalar_type);
            THDoubleTensor_gels(res1_, res2_, self_, A_);
            break;
        }
        case ScalarType::Float: {
            auto res1_ = checked_dense_tensor_unwrap(res1, "res1", 0, "_th_gels_out", false, DeviceType::CPU, dispatch_scalar_type);
            auto res2_ = checked_dense_tensor_unwrap(res2, "res2", 0, "_th_gels_out", false, DeviceType::CPU, dispatch_scalar_type);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_gels_out", false, DeviceType::CPU, dispatch_scalar_type);
            auto A_ = checked_dense_tensor_unwrap(A, "A", 2, "_th_gels_out", false, DeviceType::CPU, dispatch_scalar_type);
            THFloatTensor_gels(res1_, res2_, self_, A_);
            break;
        }
        default:
            AT_ERROR("_th_gels_out not supported on CPUType for ", dispatch_scalar_type);
    }
    return std::tuple<Tensor &, Tensor &>(res1, res2);
}
std::tuple<Tensor,Tensor> _th_gels(const Tensor & self, const Tensor & A) {
    // DeviceGuard omitted
    auto dispatch_scalar_type = infer_scalar_type(self);
    auto res1_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(c10::Storage::use_byte_size_t(), 0, allocator(), true),DispatchKey::CPU, scalarTypeToTypeMeta(dispatch_scalar_type)).release();
    auto res1 = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(res1_));
    auto res2_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(c10::Storage::use_byte_size_t(), 0, allocator(), true),DispatchKey::CPU, scalarTypeToTypeMeta(dispatch_scalar_type)).release();
    auto res2 = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(res2_));
    switch (dispatch_scalar_type) {
        case ScalarType::Double: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_gels", false, DeviceType::CPU, dispatch_scalar_type);
            auto A_ = checked_dense_tensor_unwrap(A, "A", 2, "_th_gels", false, DeviceType::CPU, dispatch_scalar_type);
            THDoubleTensor_gels(res1_, res2_, self_, A_);
            break;
        }
        case ScalarType::Float: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_gels", false, DeviceType::CPU, dispatch_scalar_type);
            auto A_ = checked_dense_tensor_unwrap(A, "A", 2, "_th_gels", false, DeviceType::CPU, dispatch_scalar_type);
            THFloatTensor_gels(res1_, res2_, self_, A_);
            break;
        }
        default:
            AT_ERROR("_th_gels not supported on CPUType for ", dispatch_scalar_type);
    }
    return std::tuple<Tensor, Tensor>(res1, res2);
}
Tensor & _th_potri_out(Tensor & output, const Tensor & self, bool upper) {
    // DeviceGuard omitted
    auto dispatch_scalar_type = infer_scalar_type(self);

    switch (dispatch_scalar_type) {
        case ScalarType::Double: {
            auto output_ = checked_dense_tensor_unwrap(output, "output", 0, "_th_potri_out", false, DeviceType::CPU, dispatch_scalar_type);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_potri_out", false, DeviceType::CPU, dispatch_scalar_type);
            THDoubleTensor_potri(output_, self_, upper);
            break;
        }
        case ScalarType::Float: {
            auto output_ = checked_dense_tensor_unwrap(output, "output", 0, "_th_potri_out", false, DeviceType::CPU, dispatch_scalar_type);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_potri_out", false, DeviceType::CPU, dispatch_scalar_type);
            THFloatTensor_potri(output_, self_, upper);
            break;
        }
        default:
            AT_ERROR("_th_potri_out not supported on CPUType for ", dispatch_scalar_type);
    }
    return output;
}
Tensor _th_potri(const Tensor & self, bool upper) {
    // DeviceGuard omitted
    auto dispatch_scalar_type = infer_scalar_type(self);
    auto output_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(c10::Storage::use_byte_size_t(), 0, allocator(), true),DispatchKey::CPU, scalarTypeToTypeMeta(dispatch_scalar_type)).release();
    auto output = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(output_));
    switch (dispatch_scalar_type) {
        case ScalarType::Double: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_potri", false, DeviceType::CPU, dispatch_scalar_type);
            THDoubleTensor_potri(output_, self_, upper);
            break;
        }
        case ScalarType::Float: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_potri", false, DeviceType::CPU, dispatch_scalar_type);
            THFloatTensor_potri(output_, self_, upper);
            break;
        }
        default:
            AT_ERROR("_th_potri not supported on CPUType for ", dispatch_scalar_type);
    }
    return output;
}
std::tuple<Tensor &,Tensor &> _th_geqrf_out(Tensor & res1, Tensor & res2, const Tensor & self) {
    // DeviceGuard omitted
    auto dispatch_scalar_type = infer_scalar_type(self);

    switch (dispatch_scalar_type) {
        case ScalarType::Double: {
            auto res1_ = checked_dense_tensor_unwrap(res1, "res1", 0, "_th_geqrf_out", false, DeviceType::CPU, dispatch_scalar_type);
            auto res2_ = checked_dense_tensor_unwrap(res2, "res2", 0, "_th_geqrf_out", false, DeviceType::CPU, dispatch_scalar_type);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_geqrf_out", false, DeviceType::CPU, dispatch_scalar_type);
            THDoubleTensor_geqrf(res1_, res2_, self_);
            break;
        }
        case ScalarType::Float: {
            auto res1_ = checked_dense_tensor_unwrap(res1, "res1", 0, "_th_geqrf_out", false, DeviceType::CPU, dispatch_scalar_type);
            auto res2_ = checked_dense_tensor_unwrap(res2, "res2", 0, "_th_geqrf_out", false, DeviceType::CPU, dispatch_scalar_type);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_geqrf_out", false, DeviceType::CPU, dispatch_scalar_type);
            THFloatTensor_geqrf(res1_, res2_, self_);
            break;
        }
        default:
            AT_ERROR("_th_geqrf_out not supported on CPUType for ", dispatch_scalar_type);
    }
    return std::tuple<Tensor &, Tensor &>(res1, res2);
}
std::tuple<Tensor,Tensor> _th_geqrf(const Tensor & self) {
    // DeviceGuard omitted
    auto dispatch_scalar_type = infer_scalar_type(self);
    auto res1_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(c10::Storage::use_byte_size_t(), 0, allocator(), true),DispatchKey::CPU, scalarTypeToTypeMeta(dispatch_scalar_type)).release();
    auto res1 = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(res1_));
    auto res2_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(c10::Storage::use_byte_size_t(), 0, allocator(), true),DispatchKey::CPU, scalarTypeToTypeMeta(dispatch_scalar_type)).release();
    auto res2 = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(res2_));
    switch (dispatch_scalar_type) {
        case ScalarType::Double: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_geqrf", false, DeviceType::CPU, dispatch_scalar_type);
            THDoubleTensor_geqrf(res1_, res2_, self_);
            break;
        }
        case ScalarType::Float: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_geqrf", false, DeviceType::CPU, dispatch_scalar_type);
            THFloatTensor_geqrf(res1_, res2_, self_);
            break;
        }
        default:
            AT_ERROR("_th_geqrf not supported on CPUType for ", dispatch_scalar_type);
    }
    return std::tuple<Tensor, Tensor>(res1, res2);
}
Tensor & _th_orgqr_out(Tensor & result, const Tensor & self, const Tensor & input2) {
    // DeviceGuard omitted
    auto dispatch_scalar_type = infer_scalar_type(self);

    switch (dispatch_scalar_type) {
        case ScalarType::Double: {
            auto result_ = checked_dense_tensor_unwrap(result, "result", 0, "_th_orgqr_out", false, DeviceType::CPU, dispatch_scalar_type);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_orgqr_out", false, DeviceType::CPU, dispatch_scalar_type);
            auto input2_ = checked_dense_tensor_unwrap(input2, "input2", 2, "_th_orgqr_out", false, DeviceType::CPU, dispatch_scalar_type);
            THDoubleTensor_orgqr(result_, self_, input2_);
            break;
        }
        case ScalarType::Float: {
            auto result_ = checked_dense_tensor_unwrap(result, "result", 0, "_th_orgqr_out", false, DeviceType::CPU, dispatch_scalar_type);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_orgqr_out", false, DeviceType::CPU, dispatch_scalar_type);
            auto input2_ = checked_dense_tensor_unwrap(input2, "input2", 2, "_th_orgqr_out", false, DeviceType::CPU, dispatch_scalar_type);
            THFloatTensor_orgqr(result_, self_, input2_);
            break;
        }
        default:
            AT_ERROR("_th_orgqr_out not supported on CPUType for ", dispatch_scalar_type);
    }
    return result;
}
Tensor _th_orgqr(const Tensor & self, const Tensor & input2) {
    // DeviceGuard omitted
    auto dispatch_scalar_type = infer_scalar_type(self);
    auto result_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(c10::Storage::use_byte_size_t(), 0, allocator(), true),DispatchKey::CPU, scalarTypeToTypeMeta(dispatch_scalar_type)).release();
    auto result = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(result_));
    switch (dispatch_scalar_type) {
        case ScalarType::Double: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_orgqr", false, DeviceType::CPU, dispatch_scalar_type);
            auto input2_ = checked_dense_tensor_unwrap(input2, "input2", 2, "_th_orgqr", false, DeviceType::CPU, dispatch_scalar_type);
            THDoubleTensor_orgqr(result_, self_, input2_);
            break;
        }
        case ScalarType::Float: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_orgqr", false, DeviceType::CPU, dispatch_scalar_type);
            auto input2_ = checked_dense_tensor_unwrap(input2, "input2", 2, "_th_orgqr", false, DeviceType::CPU, dispatch_scalar_type);
            THFloatTensor_orgqr(result_, self_, input2_);
            break;
        }
        default:
            AT_ERROR("_th_orgqr not supported on CPUType for ", dispatch_scalar_type);
    }
    return result;
}
Tensor & _th_ormqr_out(Tensor & result, const Tensor & self, const Tensor & input2, const Tensor & input3, bool left, bool transpose) {
    // DeviceGuard omitted
    auto dispatch_scalar_type = infer_scalar_type(self);

    switch (dispatch_scalar_type) {
        case ScalarType::Double: {
            auto result_ = checked_dense_tensor_unwrap(result, "result", 0, "_th_ormqr_out", false, DeviceType::CPU, dispatch_scalar_type);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_ormqr_out", false, DeviceType::CPU, dispatch_scalar_type);
            auto input2_ = checked_dense_tensor_unwrap(input2, "input2", 2, "_th_ormqr_out", false, DeviceType::CPU, dispatch_scalar_type);
            auto input3_ = checked_dense_tensor_unwrap(input3, "input3", 3, "_th_ormqr_out", false, DeviceType::CPU, dispatch_scalar_type);
            THDoubleTensor_ormqr(result_, self_, input2_, input3_, left, transpose);
            break;
        }
        case ScalarType::Float: {
            auto result_ = checked_dense_tensor_unwrap(result, "result", 0, "_th_ormqr_out", false, DeviceType::CPU, dispatch_scalar_type);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_ormqr_out", false, DeviceType::CPU, dispatch_scalar_type);
            auto input2_ = checked_dense_tensor_unwrap(input2, "input2", 2, "_th_ormqr_out", false, DeviceType::CPU, dispatch_scalar_type);
            auto input3_ = checked_dense_tensor_unwrap(input3, "input3", 3, "_th_ormqr_out", false, DeviceType::CPU, dispatch_scalar_type);
            THFloatTensor_ormqr(result_, self_, input2_, input3_, left, transpose);
            break;
        }
        default:
            AT_ERROR("_th_ormqr_out not supported on CPUType for ", dispatch_scalar_type);
    }
    return result;
}
Tensor _th_ormqr(const Tensor & self, const Tensor & input2, const Tensor & input3, bool left, bool transpose) {
    // DeviceGuard omitted
    auto dispatch_scalar_type = infer_scalar_type(self);
    auto result_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(c10::Storage::use_byte_size_t(), 0, allocator(), true),DispatchKey::CPU, scalarTypeToTypeMeta(dispatch_scalar_type)).release();
    auto result = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(result_));
    switch (dispatch_scalar_type) {
        case ScalarType::Double: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_ormqr", false, DeviceType::CPU, dispatch_scalar_type);
            auto input2_ = checked_dense_tensor_unwrap(input2, "input2", 2, "_th_ormqr", false, DeviceType::CPU, dispatch_scalar_type);
            auto input3_ = checked_dense_tensor_unwrap(input3, "input3", 3, "_th_ormqr", false, DeviceType::CPU, dispatch_scalar_type);
            THDoubleTensor_ormqr(result_, self_, input2_, input3_, left, transpose);
            break;
        }
        case ScalarType::Float: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_ormqr", false, DeviceType::CPU, dispatch_scalar_type);
            auto input2_ = checked_dense_tensor_unwrap(input2, "input2", 2, "_th_ormqr", false, DeviceType::CPU, dispatch_scalar_type);
            auto input3_ = checked_dense_tensor_unwrap(input3, "input3", 3, "_th_ormqr", false, DeviceType::CPU, dispatch_scalar_type);
            THFloatTensor_ormqr(result_, self_, input2_, input3_, left, transpose);
            break;
        }
        default:
            AT_ERROR("_th_ormqr not supported on CPUType for ", dispatch_scalar_type);
    }
    return result;
}
std::tuple<Tensor &,Tensor &> _th_multinomial_alias_setup_out(Tensor & J, Tensor & q, const Tensor & probs) {
    // DeviceGuard omitted
    auto dispatch_scalar_type = infer_scalar_type(J);

    switch (dispatch_scalar_type) {
        case ScalarType::Double: {
            auto probs_ = checked_dense_tensor_unwrap(probs, "probs", 1, "_th_multinomial_alias_setup_out", false, DeviceType::CPU, dispatch_scalar_type);
            auto J_ = checked_dense_tensor_unwrap(J, "J", 1, "_th_multinomial_alias_setup_out", false, DeviceType::CPU, ScalarType::Long);
            auto q_ = checked_dense_tensor_unwrap(q, "q", 1, "_th_multinomial_alias_setup_out", false, DeviceType::CPU, dispatch_scalar_type);
            THDoubleTensor_multinomialAliasSetup(probs_, J_, q_);
            break;
        }
        case ScalarType::Float: {
            auto probs_ = checked_dense_tensor_unwrap(probs, "probs", 1, "_th_multinomial_alias_setup_out", false, DeviceType::CPU, dispatch_scalar_type);
            auto J_ = checked_dense_tensor_unwrap(J, "J", 1, "_th_multinomial_alias_setup_out", false, DeviceType::CPU, ScalarType::Long);
            auto q_ = checked_dense_tensor_unwrap(q, "q", 1, "_th_multinomial_alias_setup_out", false, DeviceType::CPU, dispatch_scalar_type);
            THFloatTensor_multinomialAliasSetup(probs_, J_, q_);
            break;
        }
        default:
            AT_ERROR("_th_multinomial_alias_setup_out not supported on CPUType for ", dispatch_scalar_type);
    }
    return std::tuple<Tensor &, Tensor &>(J, q);
}
std::tuple<Tensor,Tensor> _th_multinomial_alias_setup(const Tensor & probs) {
    // DeviceGuard omitted
    auto dispatch_scalar_type = infer_scalar_type(probs);
    auto J_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(c10::Storage::use_byte_size_t(), 0, allocator(), true),DispatchKey::CPU, scalarTypeToTypeMeta(ScalarType::Long)).release();
    auto J = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(J_));
    auto q_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(c10::Storage::use_byte_size_t(), 0, allocator(), true),DispatchKey::CPU, scalarTypeToTypeMeta(dispatch_scalar_type)).release();
    auto q = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(q_));
    switch (dispatch_scalar_type) {
        case ScalarType::Double: {
            auto probs_ = checked_dense_tensor_unwrap(probs, "probs", 1, "_th_multinomial_alias_setup", false, DeviceType::CPU, dispatch_scalar_type);
            THDoubleTensor_multinomialAliasSetup(probs_, J_, q_);
            break;
        }
        case ScalarType::Float: {
            auto probs_ = checked_dense_tensor_unwrap(probs, "probs", 1, "_th_multinomial_alias_setup", false, DeviceType::CPU, dispatch_scalar_type);
            THFloatTensor_multinomialAliasSetup(probs_, J_, q_);
            break;
        }
        default:
            AT_ERROR("_th_multinomial_alias_setup not supported on CPUType for ", dispatch_scalar_type);
    }
    return std::tuple<Tensor, Tensor>(J, q);
}
Tensor & _th_multinomial_alias_draw_out(Tensor & result, const Tensor & q, const Tensor & J, int64_t num_samples, c10::optional<Generator> generator) {
    // DeviceGuard omitted
    auto dispatch_scalar_type = infer_scalar_type(result);

    switch (dispatch_scalar_type) {
        case ScalarType::Double: {
            auto result_ = checked_dense_tensor_unwrap(result, "result", 0, "_th_multinomial_alias_draw_out", false, DeviceType::CPU, ScalarType::Long);
            auto q_ = checked_dense_tensor_unwrap(q, "q", 1, "_th_multinomial_alias_draw_out", false, DeviceType::CPU, dispatch_scalar_type);
            auto J_ = checked_dense_tensor_unwrap(J, "J", 2, "_th_multinomial_alias_draw_out", false, DeviceType::CPU, ScalarType::Long);
            THDoubleTensor_multinomialAliasDraw(result_, q_, J_, num_samples, generator);
            break;
        }
        case ScalarType::Float: {
            auto result_ = checked_dense_tensor_unwrap(result, "result", 0, "_th_multinomial_alias_draw_out", false, DeviceType::CPU, ScalarType::Long);
            auto q_ = checked_dense_tensor_unwrap(q, "q", 1, "_th_multinomial_alias_draw_out", false, DeviceType::CPU, dispatch_scalar_type);
            auto J_ = checked_dense_tensor_unwrap(J, "J", 2, "_th_multinomial_alias_draw_out", false, DeviceType::CPU, ScalarType::Long);
            THFloatTensor_multinomialAliasDraw(result_, q_, J_, num_samples, generator);
            break;
        }
        default:
            AT_ERROR("_th_multinomial_alias_draw_out not supported on CPUType for ", dispatch_scalar_type);
    }
    return result;
}
Tensor _th_multinomial_alias_draw(const Tensor & q, const Tensor & J, int64_t num_samples, c10::optional<Generator> generator) {
    // DeviceGuard omitted
    auto dispatch_scalar_type = infer_scalar_type(q);
    auto result_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(c10::Storage::use_byte_size_t(), 0, allocator(), true),DispatchKey::CPU, scalarTypeToTypeMeta(ScalarType::Long)).release();
    auto result = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(result_));
    switch (dispatch_scalar_type) {
        case ScalarType::Double: {
            auto q_ = checked_dense_tensor_unwrap(q, "q", 1, "_th_multinomial_alias_draw", false, DeviceType::CPU, dispatch_scalar_type);
            auto J_ = checked_dense_tensor_unwrap(J, "J", 2, "_th_multinomial_alias_draw", false, DeviceType::CPU, ScalarType::Long);
            THDoubleTensor_multinomialAliasDraw(result_, q_, J_, num_samples, generator);
            break;
        }
        case ScalarType::Float: {
            auto q_ = checked_dense_tensor_unwrap(q, "q", 1, "_th_multinomial_alias_draw", false, DeviceType::CPU, dispatch_scalar_type);
            auto J_ = checked_dense_tensor_unwrap(J, "J", 2, "_th_multinomial_alias_draw", false, DeviceType::CPU, ScalarType::Long);
            THFloatTensor_multinomialAliasDraw(result_, q_, J_, num_samples, generator);
            break;
        }
        default:
            AT_ERROR("_th_multinomial_alias_draw not supported on CPUType for ", dispatch_scalar_type);
    }
    return result;
}

} // namespace th
} // namespace legacy
} // namespace native
} // namespace at
