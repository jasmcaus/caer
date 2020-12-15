#pragma once

// Indexing tensors by by tensors

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>

namespace at {
  struct TensorIterator;
}

namespace at { namespace native {

enum class SCATTER_GATHER_OP: uint8_t {REDUCE_ADD, REDUCE_MULTIPLY};

using index_fn = void(*)(TensorIterator &, IntArrayRef indexed_sizes, IntArrayRef indexed_strides);
using index_put_fn = void(*)(TensorIterator &, IntArrayRef indexed_sizes, IntArrayRef indexed_strides, bool accumulate);
using index_put_accum_fn = void(*)(Tensor &, TensorList , const Tensor &, bool unsafe);
using masked_fill_fn = void(*)(TensorIterator &, Scalar scalar);
using masked_select_fn = void(*)(TensorIterator &, int64_t orig_stride);

using gather_fn = void (*)(Tensor & result, const Tensor & self, int64_t dim, const Tensor & index);
using scatter_fn = void(*)(Tensor& self, int64_t dim, const Tensor& index, const Tensor& src);
using scatter_fill_fn = void(*)(Tensor& self, int64_t dim, const Tensor& index, Scalar src);
using scatter_add_fn = void(*)(Tensor& self, int64_t dim, const Tensor& index, const Tensor& src);
using scatter_reduce_fn = void(*)(Tensor& self, const int64_t dim, const Tensor& index,
                                  const Tensor& src, const SCATTER_GATHER_OP& reduce);
using scatter_scalar_reduce_fn = void(*)(Tensor& self, const int64_t dim, const Tensor& index,
                                         Scalar& value, const SCATTER_GATHER_OP& reduce);

DECLARE_DISPATCH(index_fn, index_stub);
DECLARE_DISPATCH(index_put_fn, index_put_stub);
DECLARE_DISPATCH(index_put_accum_fn, index_put_accum_stub);
DECLARE_DISPATCH(masked_fill_fn, masked_fill_stub);
DECLARE_DISPATCH(masked_select_fn, masked_select_serial_stub);
DECLARE_DISPATCH(masked_select_fn, masked_select_stub);

DECLARE_DISPATCH(gather_fn, gather_stub);
DECLARE_DISPATCH(scatter_fn, scatter_stub);
DECLARE_DISPATCH(scatter_fill_fn, scatter_fill_stub);
DECLARE_DISPATCH(scatter_add_fn, scatter_add_stub);
DECLARE_DISPATCH(scatter_reduce_fn, scatter_reduce_stub);
DECLARE_DISPATCH(scatter_scalar_reduce_fn, scatter_scalar_reduce_stub);

TORCH_API Tensor& index_out(Tensor& result, const Tensor & self, TensorList indices);

}} // namespace at::native
