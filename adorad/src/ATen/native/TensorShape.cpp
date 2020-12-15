#include <algorithm>
#include <vector>
#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/ExpandUtils.h>
#include <ATen/InferSize.h>
#include <ATen/NativeFunctions.h>
#include <ATen/WrapDimUtils.h>
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>
#include <ATen/native/Resize.h>
#include <ATen/native/TypeProperties.h>
#include <ATen/SparseTensorUtils.h>
#include <ATen/quantized/QTensorImpl.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/TypeProperties.h>
#include <ATen/native/cpu/CatKernel.h>
#include <ATen/native/Copy.h>
#include <ATen/MemoryOverlap.h>

namespace at {
namespace native {

DEFINE_DISPATCH(cat_serial_stub);

Tensor _reshape_from_tensor(const Tensor& self, const Tensor& shape_tensor) {
  TORCH_CHECK(shape_tensor.dim() == 1);
  std::vector<int64_t> shape;
  auto accessor = shape_tensor.accessor<int64_t, 1>();
  for (size_t i = 0; i < shape_tensor.numel(); ++i) {
    shape.push_back(accessor[i]);
  }
  return self.reshape(IntArrayRef(shape));
}

Tensor _shape_as_tensor(const Tensor& self) {
  auto options = TensorOptions(at::kLong);
  return at::tensor(self.sizes(), options);
}

Tensor& set_(Tensor& result, Storage source) {
  int64_t new_size =
      static_cast<int64_t>(source.nbytes() / result.dtype().itemsize());
  return result.set_(source, 0, new_size, {});
}

// unify with cuda implementation?  This is not done to avoid a dispatch in resize_impl_cpu_
Tensor& set_storage_cpu_(Tensor& result, Storage storage, int64_t storage_offset, IntArrayRef size, IntArrayRef stride) {
  checkSetStorage(result, storage, storage_offset, size, stride);

  result.unsafeGetTensorImpl()->set_storage_offset(storage_offset);
  c10::optional<IntArrayRef> stride_opt = stride.data() != nullptr ?
                                          c10::optional<IntArrayRef>(stride) : c10::nullopt;
  at::native::resize_impl_cpu_(result.unsafeGetTensorImpl(), size, stride_opt);
  return result;
}

Tensor& set_tensor_(Tensor& result, const Tensor& source) {
  if (result.unsafeGetTensorImpl() != source.unsafeGetTensorImpl()) {
    return result.set_(source.storage(), source.storage_offset(), source.sizes(), source.strides());
  }
  return result;
}

// this needs to be split along CPU/CUDA lines because we don't have a consistent
// way of getting the allocator to use for a device (c10::GetAllocator is not
// the same as at::cuda::getCUDADeviceAllocator().
Tensor& set_cpu_(Tensor& result) {
  caffe2::TypeMeta dtype = result.dtype();
  Storage storage(
      Storage::use_byte_size_t(),
      0,
      c10::GetAllocator(kCPU),
      true);
  result.set_(storage, 0, {0}, {});
  TORCH_INTERNAL_ASSERT(dtype == result.dtype());
  return result;
}

std::vector<Tensor> broadcast_tensors(TensorList tensors) {
  return expand_outplace(tensors);
}

// Check to see if the shape of tensors is compatible
// for being concatenated along a given dimension.
static inline void check_cat_shape_except_dim(const Tensor & first, const Tensor & second, int64_t dimension, int64_t index) {
  int64_t first_dims = first.dim();
  int64_t second_dims = second.dim();
  TORCH_CHECK(first_dims == second_dims, "Tensors must have same number of dimensions: got ",
              first_dims, " and ", second_dims);
  for (int64_t dim = 0; dim < first_dims; dim++) {
    if (dim == dimension) {
      continue;
    }
    int64_t first_dim_size = first.size(dim);
    int64_t second_dim_size = second.size(dim);
    TORCH_CHECK(first_dim_size == second_dim_size, "Sizes of tensors must match except in dimension ",
                dimension, ". Got ", first_dim_size, " and ", second_dim_size, " in dimension ", dim,
                " (The offending index is ", index, ")");
  }
}

Tensor & _cat_out_cpu(Tensor& result, TensorList tensors, int64_t dim) {
  // previously, size [0] tensors were the only possible empty tensors; thus, it wasn't possible
  // to cat empty tensors unless all the other tensors were 1-dimensional, so we allowed these tensors
  // to be "skipped".  We maintain this behavior for backwards compatibility, but only for this specific
  // size (i.e. other empty sizes are not skipped).
  // FIXME: warn if this is the case
  bool allSkipped = true;
  bool allContiguous = true;
  Tensor notSkippedTensor;

  // Inputs cannot alias the output tensor
  for (int64_t i = 0; i < tensors.size(); i++) {
    auto lap = at::get_overlap_status(result, tensors[i]);
    TORCH_CHECK(lap != at::MemOverlapStatus::PARTIAL &&
        lap != at::MemOverlapStatus::FULL, 0,
        "unsupported operation: the input tensors cannot refer to any of the "
        "output memory locations. Found overlap in input tensor ", i);
  }
  at::assert_no_internal_overlap(result);

  auto should_skip = [](const Tensor& t) { return t.numel() == 0 && t.dim() == 1; };
  for (auto const &tensor : tensors) {
    if (should_skip(tensor)) {
      continue;
    }
    // we've found a non-empty tensor
    allSkipped = false;
    notSkippedTensor = tensor;
    break;
  }
  if (allSkipped) {
    return result;
  }

  TORCH_CHECK(tensors.size() > 0, "expected a non-empty list of Tensors");
  TORCH_CHECK(dim <= notSkippedTensor.dim(), "dimension ", dim, "out of range");

  // when the input tensors are of the same size and strides,
  // reuse the same iterator for all input tensors
  bool reuse_iterator = true;
  bool no_type_promotion = true;
  // Check the type of the result
  no_type_promotion = result.dtype() == notSkippedTensor.dtype();

  // compute size of the result in the cat dimension
  int64_t cat_dim_size = 0;
  auto first_tensor_mem_format = tensors[0].suggest_memory_format();
  for (int i = 0; i < tensors.size(); i++) {
    auto const &tensor = tensors[i];
    if (should_skip(tensor)) {
      // don't use fast path for empty tensor
      allContiguous = false;
      continue;
    }
    check_cat_shape_except_dim(notSkippedTensor, tensor, dim, i);
    cat_dim_size += tensor.size(dim);

    if (!tensor.is_contiguous(first_tensor_mem_format)) {
      allContiguous = false;
    }

    if (tensor.sizes() != notSkippedTensor.sizes() ||
        tensor.strides() != notSkippedTensor.strides()) {
      reuse_iterator = false;
    }
    if (tensor.dtype() != notSkippedTensor.dtype()) {
      no_type_promotion = false;
    }
  }
  // compute the size of the result
  auto result_size = notSkippedTensor.sizes().vec();
  result_size[dim] = cat_dim_size;
  result.resize_(result_size, first_tensor_mem_format);
  if (result.numel() == 0) {
    return result;
  }

  // fast path for single thread when both inputs and result are contiguous and not empty
  allContiguous = allContiguous && result.is_contiguous(first_tensor_mem_format);
  bool use_serial_kernel = result.numel() < at::internal::GRAIN_SIZE || at::get_num_threads() == 1;
  ScalarType dtype = notSkippedTensor.scalar_type();
  if (use_serial_kernel && allContiguous && no_type_promotion && (dtype == ScalarType::Double || dtype == ScalarType::Float)) {
    cat_serial_stub(kCPU, result, tensors, dim);
    return result;
  }

  int64_t offset = 0;
  if (reuse_iterator &&
      result.is_contiguous(first_tensor_mem_format) &&
      no_type_promotion) {
    auto source_slice = notSkippedTensor;
    auto slice_dim_size = source_slice.size(dim);
    auto result_slice = result.narrow(dim, 0, slice_dim_size);
    auto result_slice_data = result_slice.data_ptr();
    auto result_stride_bytes = result.stride(dim) * elementSize(result.scalar_type());

    auto iter = TensorIteratorConfig()
      .set_check_mem_overlap(false)  // Already checked above
      .resize_outputs(false)
      .add_output(result_slice)
      .add_input(source_slice)
      .enforce_safe_casting_to_output(true)
      .build();

    for (auto const &tensor : tensors) {
      if (should_skip(tensor)) {
        continue;
      }
      auto source_data = static_cast<char*>(tensor.data_ptr());
      auto result_data = static_cast<char*>(result_slice_data) + offset * result_stride_bytes;
      iter.unsafe_replace_operand(0, result_data);
      iter.unsafe_replace_operand(1, source_data);
      copy_stub(iter.device_type(), iter, false);
      offset += slice_dim_size;
    }
  } else {
    for (auto const &tensor: tensors) {
      if (should_skip(tensor)) {
        continue;
      }
      auto slice_dim_size = tensor.size(dim);
      auto result_slice = result.narrow(dim, offset, slice_dim_size);

      auto iter = TensorIteratorConfig()
        .set_check_mem_overlap(false)  // Already checked above
        .resize_outputs(false)
        .add_output(result_slice)
        .add_input(tensor)
        .promote_inputs_to_common_dtype(true)
        .cast_common_dtype_to_outputs(true)
        .enforce_safe_casting_to_output(true)
        .build();
      copy_stub(iter.device_type(), iter, false);
      offset += slice_dim_size;
    }
  }

  return result;
}

Tensor _cat_cpu(TensorList tensors, int64_t dim) {
  ScalarType high_type = result_type(tensors);
  Tensor result = at::empty({0}, tensors[0].options().dtype(high_type));
  return native::_cat_out_cpu(result, tensors, dim);
}

static void check_cat_no_zero_dim(TensorList tensors) {
  for(size_t i = 0; i < tensors.size(); ++i) {
    auto& t = tensors[i];
    TORCH_CHECK(t.dim() > 0,
             "zero-dimensional tensor (at position ", i, ") cannot be concatenated");
  }
}

Tensor & cat_out(Tensor & result, TensorList tensors, int64_t dim) {
  check_cat_no_zero_dim(tensors);
  dim = legacy_cat_wrap_dim(dim, tensors);
  auto maybe_outnames = namedinference::compute_cat_outnames(tensors);
  {
    NoNamesGuard guard;
    at::_cat_out(result, tensors, dim);
  }
  namedinference::propagate_names_if_nonempty(result, maybe_outnames);
  return result;
}

Tensor& cat_out(Tensor& result, TensorList tensors, Dimname dim) {
  TORCH_CHECK(!tensors.empty(), "expected a non-empty list of Tensors");
  return at::cat_out(result, tensors, dimname_to_position(tensors[0], dim));
}

Tensor cat(TensorList tensors, Dimname dim) {
  TORCH_CHECK(!tensors.empty(), "expected a non-empty list of Tensors");
  return at::cat(tensors, dimname_to_position(tensors[0], dim));
}

static bool sizes_match_except(IntArrayRef s1, IntArrayRef s2, int64_t dim_except /* should already be wrapped */) {
  if (s1.size() != s2.size()) {
    return false;
  }
  for (int64_t i = 0; i < s1.size(); ++i) {
    if (i != dim_except && s1[i] != s2[i]) {
      return false;
    }
  }
  return true;
}

// Check to see if the shape of tensors is compatible
// for being concatenated along a given dimension.
static void check_cat_sparse_dims(Tensor const &t,
  int64_t pos /* used only for debug messages */,
  IntArrayRef sizes,
  int64_t wrapped,
  int64_t sparse_dim,
  int64_t dense_dim) {
    TORCH_CHECK(t.is_sparse(),
            "Concatenating sparse tensors, but a dense tensor was found at position ", pos, ".");
    TORCH_CHECK(sizes_match_except(sizes, t.sizes(), wrapped),
            "All tensors must have the same shape: ", sizes, " (except in the concatenating dimension),"
            " but found shape: ", t.sizes(), " at position ", pos, ".");
    TORCH_CHECK(t.sparse_dim() == sparse_dim && t.dense_dim() == dense_dim,
            "All tensors must have the same sparse_dim and dense_dim: ", sparse_dim, ", ", dense_dim,
            ", but tensor at position ", pos, " has ", t.sparse_dim(), ", ", t.dense_dim(), ".");
}

static Tensor cat_sparse(TensorList tensors, int64_t dim) {
  std::vector<Tensor> indices;
  std::vector<Tensor> values;
  int64_t wrapped = maybe_wrap_dim(dim, tensors[0].dim());
  int64_t sparse_dim = tensors[0].sparse_dim();
  int64_t dense_dim = tensors[0].dense_dim();
  IntArrayRef sizes = tensors[0].sizes();
  if (wrapped < sparse_dim) {
    for (size_t i = 0; i < tensors.size(); ++i) {
      auto const &t = tensors[i];
      check_cat_sparse_dims(t, i, sizes, wrapped, sparse_dim, dense_dim);
      indices.push_back(t._indices());
      values.push_back(t._values());
    }
    Tensor idxs = at::cat(indices, 1);
    Tensor vals = at::cat(values, 0);

    // We now need to move the indices of each
    // input tensor up along `dim` by an appropriate amount.
    // E.g., if t1 has indices [[2,3,4],[5,6,7]],
    // and sizes [10, 7]
    // then torch.cat((t1,t1,t1),1) should have indices
    // [[2,3,4,2,3,4,2,3,4],[5,6,7,12,13,14,19,20,21]],
    // so we need to increase idxs[1][3:6] by 7
    // and idxs[1][6:9] by 14.
    int64_t col = 0;
    int64_t cumulative_offset = 0;
    for (size_t i = 0; i < tensors.size(); ++i) {
      auto const &t = tensors[i];
      int64_t this_piece_size = t._nnz();
      // cumulative_offset is zero for the first piece, so
      // don't waste time doing this operation unless i > 0.
      if (i > 0) {
        idxs[wrapped].narrow(0, col, this_piece_size) += cumulative_offset;
      }
      cumulative_offset += t.size(wrapped);
      col += this_piece_size;
    }
    auto sizes_copy = sizes.vec();
    sizes_copy[wrapped] = cumulative_offset;
    return native::sparse_coo_tensor(idxs, vals, sizes_copy, tensors[0].options());
  }
  else {
    // Catting along a dense dimension requires us to create new values.
    // For illustration, consider the sparse 3d tensors t1 and t2,
    // given by t1 = [[[1,2],[3,4]], ... (zeros) ..., [[5,6],[7,8]]]
    // and t2 = [... (zeros) ..., [[9, 10], [11,12]], ... (zeros) ...],
    // Their concatenation along dimension 2 is:
    // [[[1,2,0,0],[3,4,0,0]], ... (zeros) ..., [[0,0,9,10],[0,0,11,12]], ... (zeros) ..., [[5,6,0,0],[7,8,0,0]]]
    //
    // Their values tensors are, respectively,
    // [[[1,2],[3,4]],[[5,6],[7,8]]] and [[[9,10],[11,12]]].
    //
    // and so the values tensor of their concatenation along dim 2 will be:
    // [[[1,2,0,0],[3,4,0,0]],[[5,6,0,0],[7,8,0,0]],[[0,0,9,10],[0,0,11,12]]]
    //
    // which we can get by taking the values tensor of each tensor, catting it with zeros of the appropriate size on the left and right,
    // and then catting all those results together.

    // The dimension in each tensor's values object that corresponds to the overall dimension along which we're catting.
    int64_t values_dim = wrapped - sparse_dim + 1;
    // The final size along the catted dimension.
    const int64_t total_size = std::accumulate(tensors.begin(), tensors.end(), static_cast<int64_t>(0), [values_dim](int64_t l, Tensor const &r) {
      return l + r._values().size(values_dim);
    });
    auto zeros_sizes = tensors[0]._values().sizes().vec();
    int64_t cumulative_size = 0;
    std::vector<Tensor> vals_pieces;
    std::vector<Tensor> idxs_pieces;
    for (size_t i = 0; i < tensors.size(); ++i) {
      auto const &t = tensors[i];
      check_cat_sparse_dims(t, i, sizes, wrapped, sparse_dim, dense_dim);
      // dimension 0 of values corresponds to the number of values,
      // rather than to any logical dimension of the sparse tensor.
      zeros_sizes[0] = t._values().size(0);
      zeros_sizes[values_dim] = cumulative_size;
      cumulative_size += t._values().size(values_dim);
      auto z1 = native::zeros(zeros_sizes, t._values().options());
      zeros_sizes[values_dim] = total_size - cumulative_size;
      auto z2 = native::zeros(zeros_sizes, t._values().options());
      vals_pieces.push_back(native::cat({z1, t._values(), z2}, values_dim));
      idxs_pieces.push_back(t._indices());
    }
    auto sizes_copy = sizes.vec();
    sizes_copy[wrapped] = total_size;
    // This can create an uncoalesced tensor
    return native::sparse_coo_tensor(native::cat(idxs_pieces, 1), native::cat(vals_pieces), sizes_copy, tensors[0].options());
  }
}

Tensor cat(TensorList tensors, int64_t dim) {
  if (tensors.size() > 0 &&
        tensors[0].is_sparse()) {
    return cat_sparse(tensors, dim);
  }

  check_cat_no_zero_dim(tensors);
  dim = legacy_cat_wrap_dim(dim, tensors);
  auto maybe_outnames = namedinference::compute_cat_outnames(tensors);
  Tensor result;
  {
    NoNamesGuard guard;
    result = at::_cat(tensors, dim);
  }
  namedinference::propagate_names_if_nonempty(result, maybe_outnames);
  return result;
}

Tensor block_diag(TensorList tensors) {
  Tensor result;
  if (tensors.size() == 0) {
    result = at::empty({1, 0});
    return result;
  }

  const Device& device = tensors[0].device();

  for (size_t tensor_idx = 1; tensor_idx < tensors.size(); tensor_idx++) {
    const Tensor& tensor = tensors[tensor_idx];

    TORCH_CHECK(
      tensor.device() == device,
      "torch.block_diag: input tensors must all be on the same device.",
      " Input 0 is on device ", device,
      " and input ", tensor_idx, " is on device ", tensor.device()
    );
  }

  ScalarType output_scalar_type = native::result_type(tensors);
  int64_t result_dim0 = 0;
  int64_t result_dim1 = 0;
  std::vector<Tensor> tensors_2D(tensors.size());

  // Sum the dimensions of the tensors, check tensor sizes,
  // and expand all 0-D and 1-D tensors so that everything
  // is 2-D
  for (size_t tensor_idx = 0; tensor_idx < tensors.size(); tensor_idx++) {
    const Tensor& tensor = tensors[tensor_idx];
    int64_t ndims = tensor.dim();
    TORCH_CHECK(
      ndims <= 2,
      "torch.block_diag: Input tensors must have 2 or fewer dimensions. Input ",
      tensor_idx, " has ", ndims, " dimensions"
    );

    int64_t dim0 = 1;
    int64_t dim1 = 1;

    if (ndims == 2) {
      dim0 = tensor.size(0);
      dim1 = tensor.size(1);
      tensors_2D[tensor_idx] = tensor;
    } else if (ndims == 1) {
      // Switching dim 0 to dim 1 is intentional
      dim1 = tensor.size(0);
      tensors_2D[tensor_idx] = tensor.expand({dim0, dim1});
    } else {
      tensors_2D[tensor_idx] = tensor.expand({dim0, dim1});
    }
    result_dim0 += dim0;
    result_dim1 += dim1;
  }

  result = at::zeros(
    {result_dim0, result_dim1},
    tensors[0].options().dtype(output_scalar_type)
  );

  int64_t cur_dim0 = 0;
  int64_t cur_dim1 = 0;

  // Copy each tensor into the appropriate location in the result matrix
  for (auto iter = tensors_2D.begin(); iter != tensors_2D.end(); iter++) {
    const Tensor& tensor = *iter;

    int64_t dim0 = tensor.size(0);
    int64_t dim1 = tensor.size(1);
    result.slice(0, cur_dim0, cur_dim0+dim0).slice(1, cur_dim1, cur_dim1+dim1).copy_(tensor);

    cur_dim0 += dim0;
    cur_dim1 += dim1;
  }

  return result;
}

std::vector<Tensor> chunk(const Tensor& self, int64_t chunks, int64_t dim) {
  TORCH_CHECK(self.dim() > 0,
           "chunk expects at least a 1-dimensional tensor");
  TORCH_CHECK(chunks > 0,
           "chunk expects `chunks` to be greater than 0, got: ", chunks);

  int64_t split_size = (self.size(dim) + chunks - 1) / chunks;

  // We need to call split_with_sizes in the case where split_size and dimension size are 0, because
  // a call to split would discard the number of chunks (because we can have an arbitrary number of
  // 0-sized chunks adding up to 0).  So, call split_with_sizes with the correct number of chunks,
  // eventually we will do this for all cases.
  if (split_size == 0 && self.size(dim) == 0) {
    std::vector<int64_t> split_sizes(chunks, split_size);
    split_sizes[chunks - 1] = split_size - (split_size * chunks - self.size(dim));
    return self.split_with_sizes(split_sizes, dim);
  } else {
    return self.split(split_size, dim);
  }
}

std::vector<Tensor> tensor_split(const Tensor& self, int64_t sections, int64_t dim) {
  TORCH_CHECK(self.dim() > 0, "expected at least a 1-dimensional tensor");
  int64_t dim_ = maybe_wrap_dim(dim, self.dim());
  TORCH_CHECK(sections > 0, "number of sections must be larger than 0, got ", sections);
  std::vector<Tensor> splits(sections);
  int64_t min_split_size = self.size(dim_) / sections;
  int64_t num_splits_one_extra = self.size(dim_) % sections;
  int64_t start_idx = 0;
  for (int64_t split_idx = 0; split_idx < sections; split_idx++) {
    int64_t split_size = (split_idx < num_splits_one_extra) ? (min_split_size + 1) : min_split_size;
    splits[split_idx] = at::slice(self, dim_, start_idx, start_idx + split_size);
    start_idx += split_size;
  }
  return splits;
}

std::vector<Tensor> tensor_split(const Tensor& self, IntArrayRef indices, int64_t dim) {
  TORCH_CHECK(self.dim() > 0, "expected at least a 1-dimensional tensor");
  int64_t dim_ = maybe_wrap_dim(dim, self.dim());
  int64_t num_indices = indices.size();
  std::vector<Tensor> splits(num_indices + 1);
  int64_t start_idx = 0;
  for (int64_t split_idx = 0; split_idx < num_indices; split_idx++) {
    int64_t end_idx = indices[split_idx];
    splits[split_idx] = at::slice(self, dim_, start_idx, end_idx);
    start_idx = end_idx;
  }
  splits[num_indices] = at::slice(self, dim_, start_idx, self.size(dim_));
  return splits;
}

std::vector<Tensor> unsafe_chunk(const Tensor& self, int64_t chunks, int64_t dim) {
  TORCH_CHECK(self.dim() > 0,
           "chunk expects at least a 1-dimensional tensor");
  TORCH_CHECK(chunks > 0,
           "chunk expects `chunks` to be greater than 0, got: ", chunks);

  std::vector<Tensor> result;
  int64_t split_size = (self.size(dim) + chunks - 1) / chunks;

  // See the comment above in chunk(...)
  if (split_size == 0 && self.size(dim) == 0) {
    std::vector<int64_t> split_sizes(chunks, split_size);
    split_sizes[chunks - 1] = split_size - (split_size * chunks - self.size(dim));
    return self.unsafe_split_with_sizes(split_sizes, dim);
  } else {
    return self.unsafe_split(split_size, dim);
  }
}

Tensor diagflat(const Tensor& self, int64_t offset) {
  return self.contiguous().view(-1).diag(offset);
}

Tensor diagonal(const Tensor& self, int64_t offset, int64_t dim1_, int64_t dim2_) {
  int64_t nDims = self.dim();
  int64_t dim1 = maybe_wrap_dim(dim1_, nDims);
  int64_t dim2 = maybe_wrap_dim(dim2_, nDims);
  TORCH_CHECK(dim1 != dim2, "diagonal dimensions cannot be identical ", dim1_, ", ", dim2_);
  auto outnames = namedinference::compute_diagonal_outnames(self, dim1, dim2);
  NoNamesGuard no_names_guard;

  int64_t diag_size;
  int64_t storage_offset = self.storage_offset();
  // compute storage offset and size for the diagonal
  // for positive values of offset (above the main diagonal)
  // "leftmost columns" (along dim2) are dropped
  // for negative values of offset (below the main diagonal)
  // "topmost rows" (along dim1) are dropped.
  // Note that we invert +/- in the second to absorb the negative
  // sign in the offset.
  if (offset >= 0) {
    diag_size = std::max<int64_t>(std::min(self.size(dim1), self.size(dim2)-offset), 0);
  } else {
    diag_size = std::max<int64_t>(std::min(self.size(dim1)+offset, self.size(dim2)), 0);
  }

  // NumPy allows you to specify offsets "off the end"; let's just be careful not to
  // set a ridiculous storage_offset in that case (technically it shouldn't matter
  // because there are no elements in the tensor, but let's be kosher).
  if (diag_size == 0) {
    // skip
  } else if (offset >= 0) {
    storage_offset += offset * self.stride(dim2);
  } else {
    storage_offset -= offset * self.stride(dim1);
  }

  // construct new size and stride: we drop dim1 and dim2 (maximum first for not changing the index of the minimum)
  // the new ("joint") dimension is appended to the end of the shape / stride to match numpy semantics
  auto sizes = self.sizes().vec();
  auto strides = self.strides().vec();
  sizes.erase(sizes.begin() + std::max(dim1, dim2));
  strides.erase(strides.begin() + std::max(dim1, dim2));
  sizes.erase(sizes.begin() + std::min(dim1, dim2));
  strides.erase(strides.begin() + std::min(dim1, dim2));
  sizes.push_back(diag_size);
  strides.push_back(self.stride(dim1)+self.stride(dim2));

  // return view with new parameters
  auto result = self.as_strided(sizes, strides, storage_offset);

  no_names_guard.reset();
  namedinference::propagate_names_if_nonempty(result, outnames);
  return result;
}

Tensor diagonal(const Tensor& self, Dimname outdim, Dimname dim1, Dimname dim2, int64_t offset) {
  auto result = at::diagonal(
      self,
      offset,
      dimname_to_position(self, dim1),
      dimname_to_position(self, dim2));
  // This is slower than it needs to be because there is no way to modify
  // the names of a tensor in-place right now. In the future we should consider
  // offering that functionality.
  std::vector<Dimname> new_names = result.names().vec();
  new_names[new_names.size() - 1] = outdim;
  return result.refine_names(new_names);
}

Tensor diag_embed(const Tensor& self, int64_t offset, int64_t dim1_, int64_t dim2_) {
  int64_t nDims = self.dim() + 1;
  int64_t dim1 = maybe_wrap_dim(dim1_, nDims);
  int64_t dim2 = maybe_wrap_dim(dim2_, nDims);
  TORCH_CHECK(dim1 != dim2, "diagonal dimensions cannot be identical ", dim1_, ", ", dim2_);
  int64_t new_dim_len = std::abs(offset) + self.size(-1);
  auto sizes = self.sizes().vec();
  sizes.pop_back();
  sizes.insert(sizes.begin() + std::min(dim1, dim2), new_dim_len);
  sizes.insert(sizes.begin() + std::max(dim1, dim2), new_dim_len);
  auto result = at::zeros(sizes, self.options());
  auto diag = result.diagonal(offset, dim1, dim2);
  diag.copy_(self);
  return result;
}

Tensor expand(const Tensor& self, IntArrayRef size, bool implicit) {
  // [expand implicit]
  // The implicit flag is set to true for any expand calls inserted by broadcast
  // operators in ExpandUtils.h This flag is recorded by the tracer to
  // distinguish between expands inserted by broadcasts and those explicitly
  // requested by the user, because it is legal to remove implicit expands
  // from the graph, but not legal to remove the explicit ones.
  TORCH_CHECK(size.size() >= (size_t)self.dim(),
           "expand(", self.toString(), "{", self.sizes(), "}, size=", size,
           "): the number of sizes provided (", size.size(), ") ",
           "must be greater or equal to the number of dimensions in the tensor (",
           self.dim(), ")");

  std::vector<int64_t> expandedSizes;
  std::vector<int64_t> expandedStrides;
  std::tie(expandedSizes, expandedStrides) = inferExpandGeometry(self.sizes(), self.strides(), size);

  auto result = self.as_strided(expandedSizes, expandedStrides);
  namedinference::propagate_names_for_expand(result, self);
  return result;
}

Tensor expand_as(const Tensor& self, const Tensor& other) {
  return self.expand(other.sizes());
}

Tensor sum_to_size(const Tensor& self, IntArrayRef size) {
  TORCH_CHECK(is_expandable_to(size, self.sizes()),
           "size {", size, "} is not expandable to size {", self.sizes(), "}.");

  return sum_to(self, size);
}

// We currently do not support per-channel quant for unfold, diagonal, expand, permute.
// TODO: Make this an aten function and replace as_strided_qtensorimpl once that is done.
Tensor make_qtensor(const Tensor& self, IntArrayRef size, IntArrayRef stride, QuantizerPtr quantizer) {
  auto result = detail::make_tensor<QTensorImpl>(
      Storage(self.storage()), self.key_set(), self.dtype(), quantizer);
  setStrided(result, size, stride, self.storage_offset());
  return result;
}

Tensor as_strided_tensorimpl(const Tensor& self, IntArrayRef size, IntArrayRef stride, optional<int64_t> storage_offset_) {
  auto storage_offset = storage_offset_.value_or(self.storage_offset());
  auto result = detail::make_tensor<TensorImpl>(
      Storage(self.storage()), self.key_set(), self.dtype());
  setStrided(result, size, stride, storage_offset);
  return result;
}

Tensor as_strided_qtensorimpl(const Tensor& self, IntArrayRef size, IntArrayRef stride, optional<int64_t> storage_offset_) {
  auto storage_offset = storage_offset_.value_or(self.storage_offset());
  auto quantizer = get_qtensorimpl(self)->quantizer();
  TORCH_CHECK(
      quantizer->qscheme() == QScheme::PER_TENSOR_AFFINE,
      "Setting strides is possible only on uniformly quantized tensor");
  auto result = detail::make_tensor<QTensorImpl>(
      Storage(self.storage()), self.key_set(), self.dtype(), quantizer);
  setStrided(result, size, stride, storage_offset);
  return result;
}

Tensor &as_strided_(Tensor& self, IntArrayRef size, IntArrayRef stride, optional<int64_t> storage_offset_) {
  auto storage_offset = storage_offset_.value_or(self.storage_offset());
  setStrided(self, size, stride, storage_offset);
  return self;
}

Tensor narrow_copy_sparse(const Tensor& self, int64_t dim, int64_t start, int64_t length) {
  int64_t allDim = self.dim();
  int64_t end = start+length;
  TORCH_CHECK(allDim > 0, "narrow() cannot be applied to a 0-dim tensor.");
  TORCH_CHECK(dim >= 0 && dim < allDim,
    "Dimension ", dim, " out of range. Expecting 0 <= dim < ", allDim, ".");
  TORCH_CHECK(start >= 0 && length >= 0 && end <= self.size(dim),
    "Invalid range to narrow. range(start, start+length) must be a subset of range(0, ", self.size(dim), ").")
  Tensor indices = self._indices();
  int64_t sparse_dim = self.sparse_dim();

  std::vector<int64_t> new_sizes = self.sizes().vec();
  new_sizes[dim] = length;

  Tensor new_values;
  Tensor new_indices;
  if (dim < sparse_dim) {
    Tensor mask = (indices[dim] >= start).__and__((indices[dim] < end));
    new_indices = indices.masked_select(mask).view({sparse_dim, -1});
    new_indices[dim].sub_(start);
    Tensor nzIndices = mask.nonzero().view(-1);
    new_values = self._values().index_select(0, nzIndices);
  } else {
    /* This means we are narrowing on a dense dim, which is in effect just a
        regular narrow on _values() */
    new_indices = indices;
    int64_t dense_dim = dim - sparse_dim + 1;
    new_values = self._values().narrow_copy(dense_dim, start, length);
  }

  auto newTensor = at::sparse_coo_tensor(new_indices, new_values, new_sizes);
  return newTensor._coalesced_(self.is_coalesced());
}

Tensor narrow_copy_dense(const Tensor& self, int64_t dim, int64_t start, int64_t length){
    return self.narrow(dim, start, length).clone(at::MemoryFormat::Contiguous);
}

Tensor narrow(const Tensor& self, int64_t dim, int64_t start, int64_t length) {
  TORCH_CHECK(self.dim() > 0, "narrow() cannot be applied to a 0-dim tensor.");
  auto cur_size = self.size(dim);
  if (start != cur_size) {  // start being the end is valid, but not a valid dim specification.
    start = maybe_wrap_dim(start, cur_size);
  }
  TORCH_CHECK(length >= 0 && start <= cur_size - length,
           "start (", start, ") + length (", length, ") exceeds dimension size (", cur_size, ").");
  return at::slice(self, dim, start, start + length, 1);
}

Tensor narrow(const Tensor& self, int64_t dim, const Tensor& start, int64_t length) {
  TORCH_CHECK(start.dim() == 0 && isIntegralType(start.scalar_type(), /*includeBool=*/false),
              "start must be an 0-dim integral Tensor.");
  int64_t st = start.item<int64_t>();
  return at::narrow(self, dim, st, length);
}

Tensor permute(const Tensor& self, IntArrayRef dims) {
  auto nDims = self.dim();
  TORCH_CHECK(dims.size() == (size_t)nDims,
           "number of dims don't match in permute");
  auto oldSizes = self.sizes();
  auto oldStrides = self.strides();
  std::vector<int64_t> newSizes(nDims);
  std::vector<int64_t> newStrides(nDims);
  std::vector<bool> seen(nDims);
  for (int64_t i = 0; i < nDims; i++) {
    auto dim = maybe_wrap_dim(dims[i], nDims);
    TORCH_CHECK(!seen[dim],
             "repeated dim in permute");
    seen[dim] = true;
    newSizes[i] = oldSizes[dim];
    newStrides[i] = oldStrides[dim];
  }
  return self.as_strided(newSizes, newStrides);
}

Tensor repeat(const Tensor& self, IntArrayRef repeats) {
  TORCH_CHECK(repeats.size() >= (size_t)self.dim(),
           "Number of dimensions of repeat dims can not be smaller than number of dimensions of tensor");

  // Add new leading dimensions to the tensor if the
  // number of target dimensions is larger than the
  // number of source dimensions.
  int64_t num_new_dimensions = repeats.size() - self.dim();
  std::vector<int64_t> padded_size(num_new_dimensions, 1);
  padded_size.insert(padded_size.end(), self.sizes().begin(), self.sizes().end());
  std::vector<int64_t> target_size(repeats.size());
  bool zero_tensor = false;
  for(size_t idx = 0; idx < repeats.size(); ++idx) {
    if (repeats[idx] == 0) {
      zero_tensor = true;
    }
    target_size[idx] = padded_size[idx] * repeats[idx];
  }

  Tensor xtensor = self.expand(padded_size);

  Tensor result;
  if (self.is_quantized()) {
    result = at::empty_quantized(target_size, self);
  } else {
    result = at::empty(target_size, self.options());
  }

  // return an empty tensor if one of the repeat dimensions is zero
  if (zero_tensor) {
    return result;
  }

  Tensor urtensor = at::alias(result);
  for (int64_t i = 0; i < xtensor.dim(); ++i) {
    // can't unfold with step 0, so make sure step is at least 1
    // (it doesn't matter what it is in that case, because the size is 0).
    urtensor = urtensor.unfold(i, xtensor.size(i), std::max<int64_t>(xtensor.size(i), 1));
  }

  urtensor.copy_(xtensor.expand_as(urtensor));

  return result;
}

Tensor tile(const Tensor& self, IntArrayRef reps){
  // If self.size() > len(reps), reps is promoted to self.size() by pre-pending
  // 1’s to it to keep the same behaviour as `numpy.tile`.
  // Thus for a tensor of shape (2, 3, 4, 5), a dims of (2, 2) is treated
  // as (1, 1, 2, 2).
  const int64_t size_diff = self.dim() - static_cast<int64_t>(reps.size());
  if (size_diff > 0){
    std::vector<int64_t> new_reps(size_diff, 1);
    for(auto i = decltype(reps.size()){0}; i < reps.size(); ++i){
      new_reps.emplace_back(reps[i]);
    }
    return self.repeat(IntArrayRef(new_reps));
  }
  // `torch.tile` is equivalent to the already implemented `torch.Tensor.repeat`
  return self.repeat(reps);
}

Tensor alias_with_sizes_and_strides(
    const Tensor& self,
    const c10::IntArrayRef sizes,
    const c10::IntArrayRef strides) {
  Tensor self_;
  if (self.is_quantized()) {
    auto impl = c10::make_intrusive<QTensorImpl>(
        Storage(self.storage()),
        self.key_set(),
        self.dtype(),
        get_qtensorimpl(self)->quantizer());
    impl->set_storage_offset(self.storage_offset());
    impl->set_sizes_and_strides(sizes, strides);
    self_ = Tensor(std::move(impl));
  } else {
    auto impl = c10::make_intrusive<TensorImpl>(
        Storage(self.storage()), self.key_set(), self.dtype());
    impl->set_storage_offset(self.storage_offset());
    impl->set_sizes_and_strides(sizes, strides);
    self_ = Tensor(std::move(impl));
  }
  namedinference::propagate_names(self_, self);
  return self_;
}

Tensor reshape(const Tensor& self, IntArrayRef proposed_shape) {
  if (self.is_sparse()) {
    AT_ERROR("reshape is not implemented for sparse tensors");
  }
  auto shape = infer_size(proposed_shape, self.numel());

  if (self.is_mkldnn()) {
    return at::_mkldnn_reshape(self, shape);
  }

  auto stride =
      at::detail::computeStride(self.sizes(), self.strides(), shape);
    // `computeStride` returns the proper strides to use if this
    // `reshape` can be just a view.
    //
    // NB: Even though we have viewable geometry and the target strides here,
    //     we do not just call `as_strided` on `self` because the backward
    //     for `as_strided` is not as efficient as that of `view` (since the
    //     former is meant to handle general cases).
  if (stride.has_value()) {
    return self.view(shape);
  }
  return at::_unsafe_view(self.clone(at::MemoryFormat::Contiguous), shape);
}

Tensor reshape_as(const Tensor& self, const Tensor& other) {
  return self.reshape(other.sizes());
}

static Tensor select_sparse(const Tensor& self, int64_t dim, int64_t index) {
  int64_t sparse_dim = self.sparse_dim();
  int64_t dense_dim = self.dense_dim();
  TORCH_INTERNAL_ASSERT(dim >= 0 && dim < sparse_dim + dense_dim);

  auto indices = self._indices();
  auto values = self._values();
  auto new_sizes = self.sizes().vec();
  new_sizes.erase(new_sizes.begin() + dim);

  if (dim < sparse_dim) {
    auto nzIndices = (indices[dim] == index).nonzero().view(-1);
    auto new_values = values.index_select(0, nzIndices);
    if (sparse_dim == 1) {
      // return dense part:
      if (new_values.size(0) == 1) {
        return new_values[0];
      } else {
        return new_values.sum(0);
      }
    } else {
      auto dimIndices = (arange(0, sparse_dim, self.device()) != dim).nonzero().view(-1);
      auto new_indices = indices.index_select(1, nzIndices).index_select(0, dimIndices);
      return _sparse_coo_tensor_with_dims_and_tensors(
            sparse_dim - 1, dense_dim, new_sizes, new_indices, new_values, self.options());
    }
  } else {
    auto new_values = values.select(dim - sparse_dim + 1, index);
    return _sparse_coo_tensor_with_dims_and_tensors(
         sparse_dim, dense_dim - 1, new_sizes, indices, new_values, self.options());
  }
}

Tensor select(const Tensor& self, int64_t dim, int64_t index) {
  int64_t ndim = self.dim();
  if (ndim == 0) {
    TORCH_CHECK_INDEX(false, "select() cannot be applied to a 0-dim tensor.");
  }
  dim = maybe_wrap_dim(dim, ndim);
  auto size = self.size(dim);
  if (index < -size || index >= size) {
    if (self.has_names() && self.names()[dim] != Dimname::wildcard()) {
      TORCH_CHECK_INDEX(false, "select(): index ", index, " out of range for tensor of size ",
                     self.sizes(), " at dimension ", self.names()[dim]);
    }
    TORCH_CHECK_INDEX(false, "select(): index ", index, " out of range for tensor of size ",
                   self.sizes(), " at dimension ", dim);
  }
  if (index < 0) {
    index += size;
  }
  if (self.is_sparse()) {
    return select_sparse(self, dim, index);
  }
  auto sizes = self.sizes().vec();
  auto strides = self.strides().vec();
  auto storage_offset = self.storage_offset() + index * strides[dim];
  sizes.erase(sizes.begin() + dim);
  strides.erase(strides.begin() + dim);
  auto result = self.as_strided(sizes, strides, storage_offset);
  namedinference::propagate_names_except(result, self, {dim});
  return result;
}

Tensor select(const Tensor& self, Dimname dim, int64_t index) {
  return at::select(self, dimname_to_position(self, dim), index);
}

Tensor select_backward(const Tensor& grad, IntArrayRef input_sizes, int64_t dim, int64_t index) {
  auto grad_input = at::zeros(input_sizes, grad.options());
  grad_input.select(dim, index).copy_(grad);
  return grad_input;
}

Tensor index_select_sparse(const Tensor& self, int64_t dim, const Tensor& index) {
  /*
    Algorithm:
    index - a 1-D tensor of indicies with shape (n,)
    self - sparse tensor, its shape is sizes = sparse_shape + dense_shape
      indices - 2-D tensor of indices, shape is (sparse_dims, nnz)
      values - (1+len(dense_shape))-D tensor of values, shape is (nnz,) + dense_shape
    index_select(dim, index) returns a sparse tensor with the following data
      new_sizes = sizes[:dim] + (n,) + sizes[dim+1:]
      new_indices - shape is (sparse_dims, new_nnz)
      new_values - shape is (new_nnz,) + dense_shape

      if dim < len(sparse_shape):
          for i, idx in enumerate(index):
              for j, jdx in enumerate(indices[dim]):
                  if idx == jdx:
                      icol = indices[:dim][j] + (i,) + indices[dim+1:][j]
                      new_indices.add_column(icol)
                      new_values.add_row(values[j])
      else:
          new_indices = indices
          new_values[k] = values[k].index_select(dim - len(sparse_shape), index) for k in range(nnz)
    */
  auto ndim = self.dim();
  if (ndim == 0) {
    TORCH_CHECK_INDEX(false, "index_select() cannot be applied to a 0-dim tensor.");
  }
  if (!(index.dim() == 1 && index.dtype() == at::kLong)) {
    TORCH_CHECK_INDEX(false, "index_select() argument index must be 1-D long-tensor.");
  }
  dim = maybe_wrap_dim(dim, ndim);
  auto size = self.size(dim);
  auto sparse_dim = self.sparse_dim();
  auto dense_dim = self.dense_dim();
  auto indices = self._indices();
  auto values = self._values();
  auto nnz = values.size(0);
  auto new_sizes = self.sizes().vec();
  new_sizes[dim] = index.size(0);

  if (dim < sparse_dim) {

    auto dim_indices = indices[dim];
    std::vector<int64_t> zindices;
    std::vector<int64_t> iindices;
    int64_t new_nnz = 0;
    for (int64_t i=0; i < new_sizes[dim]; i++) {
      auto idx = index[i].item<int64_t>();
      if (idx < -size || idx >= size) {
        TORCH_CHECK_INDEX(false, "index_select(): index contains ", idx, " that is out of range for tensor of size ",
                   self.sizes(), " at dimension ", dim);
      }
      if (idx < 0) {
        idx += size;
      }
      for (int64_t j=0; j < nnz; j++) {
        auto jdx = dim_indices[j].item<int64_t>();
        if (idx == jdx) {
          new_nnz++;
          iindices.push_back(i);
          zindices.push_back(j);
        }
      }
    }
    auto zIndices = at::from_blob(zindices.data(), {new_nnz}, at::kLong).to(indices.device());
    auto new_indices = indices.index_select(1, zIndices);
    new_indices[dim] = at::from_blob(iindices.data(), {new_nnz}, at::kLong).to(indices.device());
    auto new_values = values.index_select(0, zIndices);
    return _sparse_coo_tensor_with_dims_and_tensors(
        sparse_dim, dense_dim, new_sizes, new_indices, new_values, self.options());

  } else {

    auto vsize = values.sizes().vec();
    vsize[dim + 1 - sparse_dim] = index.size(0);
    auto new_values = at::empty(vsize, values.options());
    for (int64_t k=0; k < nnz; k++) {
      new_values[k] = values[k].index_select(dim - sparse_dim, index);
    }
    return _sparse_coo_tensor_with_dims_and_tensors(
        sparse_dim, dense_dim, new_sizes, indices, new_values, self.options());

  }
}

Tensor slice(const Tensor& self, int64_t dim, int64_t start, int64_t end, int64_t step) {
  int64_t ndim = self.dim();
  if (ndim == 0) {
    TORCH_CHECK_INDEX(false, "slice() cannot be applied to a 0-dim tensor.");
  }
  dim = maybe_wrap_dim(dim, ndim);
  auto sizes = self.sizes().vec();
  auto strides = self.strides().vec();
  // TODO: support negative strides
  TORCH_CHECK(step > 0, "slice step must be positive");
  if (start < 0) {
    start += sizes[dim];
  }
  if (end < 0) {
    end += sizes[dim];
  }
  if (start < 0) {
    start = 0;
  } else if (start >= sizes[dim]) {
    start = sizes[dim];
  }
  if (end < start) {
    end = start;
  } else if (end >= sizes[dim]) {
    end = sizes[dim];
  }
  auto storage_offset = self.storage_offset() + start * strides[dim];
  auto len = end - start;
  sizes[dim] = (len + step - 1) / step;  // round-up
  strides[dim] *= step;
  auto result = self.as_strided(sizes, strides, storage_offset);
  namedinference::propagate_names(result, self);
  return result;
}

Tensor slice_backward(const Tensor& grad, IntArrayRef input_sizes, int64_t dim, int64_t start, int64_t end, int64_t step) {
  auto grad_input = at::zeros(input_sizes, grad.options());
  grad_input.slice(dim, start, end, step).copy_(grad);
  return grad_input;
}

std::vector<Tensor> split(const Tensor& self, int64_t split_size, int64_t dim) {
  TORCH_CHECK(self.dim() != 0, "split expects at least a 1-dimensional tensor");
  TORCH_CHECK(split_size >= 0,  "split expects split_size be non-negative, but got split_size=", split_size);
  int64_t dim_size = self.size(dim);
  TORCH_CHECK(split_size > 0 || self.size(dim) == 0,
           "split_size can only be 0 if dimension size is 0, "
           "but got dimension size of ", dim_size);
  // if split_size is 0 and dimension size is 0, there is 1 split.
  int64_t num_splits = 1;
  if (split_size != 0) {
    // ensuring num_splits is at least 1 makes consistent the case where split_size > dim_size
    // (returns a single split).  We might want to error here, but keep it for BC.
    num_splits = std::max<int64_t>((dim_size + split_size - 1) / split_size, 1);
  }
  std::vector<Tensor> splits(num_splits);
  int64_t last_split_size = split_size - (split_size * num_splits - dim_size);

  for (int64_t i = 0; i < num_splits; ++i) {
    auto length = i < num_splits - 1 ? split_size : last_split_size;
    splits[i] = self.narrow(dim, i * split_size, length);
  }
  return splits;
}

std::vector<Tensor> unsafe_split(const Tensor& self, int64_t split_size, int64_t dim) {
  auto result = at::native::split(self, split_size, dim);
  for (auto& t : result) {
    t.unsafeGetTensorImpl()->set_version_counter(c10::VariableVersion());
  }
  return result;
}

std::vector<Tensor> split_with_sizes(const Tensor& self, IntArrayRef split_sizes, int64_t dim) {
  TORCH_CHECK(self.dim() != 0, "split expects at least a 1-dimensional tensor");
  int64_t dim_size = self.size(dim);
  int64_t num_splits = split_sizes.size();
  std::vector<Tensor> splits(num_splits);
  int64_t start_idx = 0;
  int64_t i;

  for (i = 0; i < num_splits; ++i) {
    auto length = split_sizes[i];
    TORCH_CHECK(length >= 0,
             "split_with_sizes expects split_sizes have only non-negative ",
             "entries, but got split_sizes=", split_sizes);
    splits[i] = self.narrow(dim, start_idx, length);
    start_idx += length;
  }
  TORCH_CHECK(start_idx == dim_size,
           "split_with_sizes expects split_sizes to sum exactly to ", dim_size,
           " (input tensor's size at dimension ", dim, "), ", "but got split_sizes=", split_sizes);
  return splits;
}

std::vector<Tensor> unsafe_split_with_sizes(const Tensor& self, IntArrayRef split_sizes, int64_t dim) {
  auto result = at::native::split_with_sizes(self, split_sizes, dim);
  for (auto& t : result) {
    t.unsafeGetTensorImpl()->set_version_counter(c10::VariableVersion());
  }
  return result;
}

// Precondition: tensors is non-empty
static inline std::vector<Tensor> get_stack_inputs(TensorList tensors, int64_t dim) {
  std::vector<Tensor> inputs(tensors.size());
  at::IntArrayRef entry_shape = tensors[0].sizes();
  inputs[0] = tensors[0].unsqueeze(dim);
  for (size_t i = 1; i < tensors.size(); ++i) {
    TORCH_CHECK(tensors[i].sizes() == entry_shape,
      "stack expects each tensor to be equal size, but got ", entry_shape,
      " at entry 0 and ", tensors[i].sizes(), " at entry ", i);
    inputs[i] = tensors[i].unsqueeze(dim);
  }
  return inputs;
}

Tensor stack(TensorList tensors, int64_t dim) {
  TORCH_CHECK(tensors.size() > 0,
           "stack expects a non-empty TensorList");
  dim = maybe_wrap_dim(dim, tensors[0].dim() + 1);
  return at::cat(get_stack_inputs(tensors, dim), dim);
}

Tensor& stack_out(Tensor& result, TensorList tensors, int64_t dim) {
  TORCH_CHECK(tensors.size() > 0,
           "stack expects a non-empty TensorList");
  dim = maybe_wrap_dim(dim, tensors[0].dim() + 1);
  return at::cat_out(result, get_stack_inputs(tensors, dim), dim);
}

Tensor hstack(TensorList tensors) {
  TORCH_CHECK(tensors.size() > 0,
           "hstack expects a non-empty TensorList");
  auto rep = at::atleast_1d(tensors);
  if (rep[0].dim() == 1) {
    return at::cat(rep, 0);
  }
  return at::cat(rep, 1);
}

Tensor& hstack_out(Tensor& result, TensorList tensors) {
  TORCH_CHECK(tensors.size() > 0,
           "hstack expects a non-empty TensorList");
  auto rep = at::atleast_1d(tensors);
  if (rep[0].dim() == 1) {
    return at::cat_out(result, rep, 0);
  }
  return at::cat_out(result, rep, 1);
}

Tensor vstack(TensorList tensors) {
  TORCH_CHECK(tensors.size() > 0,
           "vstack expects a non-empty TensorList");
  auto rep = at::atleast_2d(tensors);
  return at::cat(rep, 0);
}

Tensor& vstack_out(Tensor& result, TensorList tensors) {
  TORCH_CHECK(tensors.size() > 0,
           "vstack expects a non-empty TensorList");
  auto rep = at::atleast_2d(tensors);
  return at::cat_out(result, rep, 0);
}

Tensor dstack(TensorList tensors) {
  TORCH_CHECK(tensors.size() > 0,
           "dstack expects a non-empty TensorList");
  auto rep = at::atleast_3d(tensors);
  return at::cat(rep, 2);
}
Tensor& dstack_out(Tensor& result, TensorList tensors) {
  TORCH_CHECK(tensors.size() > 0,
           "dstack expects a non-empty TensorList");
  auto rep = at::atleast_3d(tensors);
  return at::cat_out(result, rep, 2);
}

static inline Tensor & sparse_transpose_(Tensor & self, int64_t dim0, int64_t dim1) {
  int64_t nsparse_dim = self.sparse_dim();
  TORCH_CHECK(dim0 < nsparse_dim && dim1 < nsparse_dim,
           "sparse transpose: transposed dimensions must be sparse ",
           "Got sparse_dim: ", nsparse_dim, ", d0: ", dim0, ", d1: ", dim1);

  if (self._indices().numel() == 0 && self._values().numel() == 0) {
    auto sizes = self.sizes().vec();
    std::swap(sizes[dim0], sizes[dim1]);

    at::sparse::get_sparse_impl(self)->raw_resize_(self.sparse_dim(), self.dense_dim(), sizes);
  } else {
    auto indices = self._indices();
    auto row0 = indices.select(0, dim0);
    auto row1 = indices.select(0, dim1);

    // swap row0 and row1
    auto tmp = at::zeros_like(row0, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    tmp.copy_(row0);
    row0.copy_(row1);
    row1.copy_(tmp);

    self._coalesced_(false);

    auto sizes = self.sizes().vec();
    std::swap(sizes[dim0], sizes[dim1]);

    at::sparse::get_sparse_impl(self)->raw_resize_(self._indices().size(0), self._values().dim() - 1, sizes);
  }
  return self;
}

// torch.row_stack, alias for torch.vstack
Tensor& row_stack_out(Tensor& result, TensorList tensors) {
  return at::vstack_out(result, tensors);
}

Tensor row_stack(TensorList tensors) {
  return at::vstack(tensors);
}

static std::vector<Tensor> reshape_input_for_column_stack(TensorList tensors) {
  std::vector<Tensor> result(tensors.size());
  auto transform_lambda = [](const Tensor& input) -> Tensor {
    // reshape 0D or 1D tensor t into (t.numel(), 1)
    if (input.dim() <= 1) {
      return input.reshape({input.numel(), 1});
    }
    return input;
  };
  std::transform(tensors.cbegin(),
                 tensors.cend(),
                 result.begin(),
                 transform_lambda);
  return result;
}

Tensor& column_stack_out(Tensor& result, TensorList tensors) {
  TORCH_CHECK(tensors.size() > 0,
              "column_stack expects a non-empty TensorList");

  auto reshaped_tensors = reshape_input_for_column_stack(tensors);
  return at::hstack_out(result, reshaped_tensors);
}

Tensor column_stack(TensorList tensors) {
  TORCH_CHECK(tensors.size() > 0,
              "column_stack expects a non-empty TensorList");

  auto reshaped_tensors = reshape_input_for_column_stack(tensors);
  return at::hstack(reshaped_tensors);
}

static Tensor& propagate_transposed_names(
    Tensor& result,
    const Tensor& other,
    int64_t dim0,
    int64_t dim1) {
  if (other.has_names()) {
    auto names = other.names().vec();
    std::swap(names[dim0], names[dim1]);
    namedinference::propagate_names_if_nonempty(result, names);
  }
  return result;
}

Tensor transpose(const Tensor& self, Dimname dim0, Dimname dim1) {
  return at::transpose(
      self, dimname_to_position(self, dim0), dimname_to_position(self, dim1));
}


Tensor & transpose_(Tensor & self, int64_t dim0, int64_t dim1) {
  auto ndims = self.dim();
  dim0 = maybe_wrap_dim(dim0, ndims);
  dim1 = maybe_wrap_dim(dim1, ndims);
  if (dim0 == dim1) {
    return self;
  }

  if (self.is_sparse()) {
    return sparse_transpose_(self, dim0, dim1);
  }

  if (self.is_mkldnn()) {
    return at::_mkldnn_transpose_(self, dim0, dim1);
  }

  auto strides = self.strides().vec();
  auto sizes = self.sizes().vec();
  std::swap(strides[dim0], strides[dim1]);
  std::swap(sizes[dim0], sizes[dim1]);
  return self.as_strided_(sizes, strides);
}

Tensor transpose(const Tensor & self, int64_t dim0, int64_t dim1) {
  auto ndims = self.dim();
  dim0 = maybe_wrap_dim(dim0, ndims);
  dim1 = maybe_wrap_dim(dim1, ndims);
  if (dim0 == dim1) {
    return self;
  }

  if (self.is_sparse()) {
    Tensor self_clone = self.clone();  // yes, this is what THS does
    return sparse_transpose_(self_clone, dim0, dim1);
  }

  if (self.is_mkldnn()) {
    return at::_mkldnn_transpose(self, dim0, dim1);
  }

  auto strides = self.strides().vec();
  auto sizes = self.sizes().vec();
  std::swap(strides[dim0], strides[dim1]);
  std::swap(sizes[dim0], sizes[dim1]);
  auto result = self.as_strided(sizes, strides);
  propagate_transposed_names(result, self, dim0, dim1);
  return result;
}

static void check_t(const Tensor& self, const char *fn) {
  if (self.is_sparse()) {
    int64_t sparse_dim = self.sparse_dim();
    int64_t dense_dim = self.dense_dim();
    TORCH_CHECK(sparse_dim <= 2 && dense_dim == 0,
             fn, " expects a tensor with <= 2 sparse and 0 dense dimensions, but got ",
             sparse_dim, " sparse and ", dense_dim, " dense dimensions");
  } else {
    TORCH_CHECK(self.dim() <= 2,
             fn, " expects a tensor with <= 2 dimensions, but self is ", self.dim(), "D");
  }
}

Tensor t(const Tensor & self) {
  check_t(self, "t()");
  return self.transpose(0, self.dim() < 2 ? 0 : 1);
}

Tensor & t_(Tensor & self) {
  check_t(self, "t_()");
  return self.transpose_(0, self.dim() < 2 ? 0 : 1);
}

std::tuple<std::vector<int64_t>, std::vector<int64_t> >
inferSqueezeGeometry(const Tensor &tensor) {
  std::vector<int64_t> sizes;
  std::vector<int64_t> strides;

  for(int64_t d = 0; d < tensor.dim(); d++) {
    if(tensor.sizes()[d] != 1) {
      sizes.push_back(tensor.sizes()[d]);
      strides.push_back(tensor.strides()[d]);
    }
  }

  return std::make_tuple(sizes, strides);
}

std::tuple<std::vector<int64_t>, std::vector<int64_t> >
inferSqueezeGeometry(const Tensor& tensor, int64_t dim) {
  std::vector<int64_t> sizes;
  std::vector<int64_t> strides;

  for(int64_t d = 0; d < tensor.dim(); d++) {
    if(d != dim || tensor.sizes()[dim] != 1) {
      sizes.push_back(tensor.sizes()[d]);
      strides.push_back(tensor.strides()[d]);
    }
  }
  return std::make_tuple(sizes, strides);
}

std::tuple<std::vector<int64_t>, std::vector<int64_t> >
inferUnsqueezeGeometry(const Tensor& tensor, int64_t dim) {
  auto sizes = tensor.sizes().vec();
  auto strides = tensor.strides().vec();
  int64_t new_stride = dim >= tensor.dim() ? 1 : sizes[dim] * strides[dim];
  sizes.insert(sizes.begin() + dim, 1);
  strides.insert(strides.begin() + dim, new_stride);

  return std::make_tuple(sizes, strides);
}

Tensor squeeze_qtensor(const Tensor& self) {
  auto quantizer = get_qtensorimpl(self)->quantizer();
  std::vector<int64_t> sizes;
  std::vector<int64_t> strides;
  std::tie(sizes, strides) = inferSqueezeGeometry(self);
  if (quantizer->qscheme() == QScheme::PER_CHANNEL_AFFINE) {
    const auto* per_channel_quantizer = static_cast<at::PerChannelAffineQuantizer*>(quantizer.get());
    auto axis = per_channel_quantizer->axis();
    int64_t shift = 0;
    for (int64_t d = 0; d < self.dim(); ++d) {
      if (self.sizes()[d] == 1) {
        TORCH_CHECK(axis != d, "Squeeze is only possible on non-axis dimension for Per-Channel Quantized Tensors.");
        if (d < axis) {
          shift += 1;
        }
      }
    }
    axis = axis - shift;
    quantizer = make_per_channel_affine_quantizer(per_channel_quantizer->scales(),
                                                  per_channel_quantizer->zero_points(),
                                                  axis,
                                                  quantizer->scalar_type());
  }
  return make_qtensor(self, sizes, strides, quantizer);
}

Tensor squeeze_qtensor(const Tensor& self, int64_t dim) {
  auto quantizer = get_qtensorimpl(self)->quantizer();
  std::vector<int64_t> sizes;
  std::vector<int64_t> strides;
  std::tie(sizes, strides) = inferSqueezeGeometry(self, dim);
  if (quantizer->qscheme() == QScheme::PER_CHANNEL_AFFINE) {
    const auto* per_channel_quantizer = static_cast<at::PerChannelAffineQuantizer*>(quantizer.get());
    auto axis = per_channel_quantizer->axis();
    TORCH_CHECK(axis != dim, "Squeeze is only possible on non-axis dimension for Per-Channel Quantized Tensors.");
    if (axis >= dim) {
      axis -= 1;
    }
    quantizer = make_per_channel_affine_quantizer(per_channel_quantizer->scales(),
                                                  per_channel_quantizer->zero_points(),
                                                  axis,
                                                  quantizer->scalar_type());
  }
  if (self.dim() == 0 || self.sizes()[dim] != 1) {
    sizes = self.sizes().vec();
    strides = self.strides().vec();
  }
  auto result = make_qtensor(self, sizes, strides, quantizer);
  namedinference::propagate_names_except(result, self, {dim});
  return result;
}

Tensor squeeze(const Tensor& self) {
  auto g = inferSqueezeGeometry(self);
  at::Tensor result;
  if (self.is_quantized()) {
    result = squeeze_qtensor(self);
  } else {
    result = self.as_strided(std::get<0>(g), std::get<1>(g));
  }
  auto maybe_outnames = namedinference::compute_squeeze_outnames(self);
  namedinference::propagate_names_if_nonempty(result, maybe_outnames);
  return result;
}

Tensor squeeze(const Tensor& self, int64_t dim) {
  int64_t dims = self.dim();
  dim = maybe_wrap_dim(dim, dims);

  if (self.is_quantized()) {
    return squeeze_qtensor(self, dim);
  }
  if (dims == 0 || self.sizes()[dim] != 1) {
    return self.as_strided(self.sizes(), self.strides());
  }
  auto g = inferSqueezeGeometry(self, dim);
  auto result = self.as_strided(std::get<0>(g), std::get<1>(g));
  namedinference::propagate_names_except(result, self, {dim});
  return result;
}

Tensor & squeeze_(Tensor& self) {
  auto g = inferSqueezeGeometry(self);
  return self.as_strided_(std::get<0>(g), std::get<1>(g));
}

Tensor & squeeze_(Tensor& self, int64_t dim) {
  int64_t dims = self.dim();
  dim = maybe_wrap_dim(dim, self.dim());

  if (dims == 0 || self.sizes()[dim] != 1) {
    return self.as_strided_(self.sizes(), self.strides());
  }
  auto g = inferSqueezeGeometry(self, dim);
  return self.as_strided_(std::get<0>(g), std::get<1>(g));
}

// _unsafe_view() differs from view() in that the returned tensor isn't treated
// as a view for the purposes of automatic differentiation. (It's not listed in
// VIEW_FUNCTIONS in gen_autograd.py).  It's only safe to use if the `self` tensor
// is temporary. For example, the viewed tensor here (a + b) is discarded immediately
// after viewing:
//
//  res = at::_unsafe_view(a + b, size);
//
// This is a hack because in-place operations on tensors treated like views
// can be much more expensive than the same operations on non-view tensors.
Tensor _unsafe_view(const Tensor& self, IntArrayRef size) {
  return self.view(size);
}

static Tensor unsqueeze_sparse(Tensor const &self, int64_t dim /* should already be wrapped */) {
  int64_t sparse_dim = self.sparse_dim();
  int64_t dense_dim = self.dense_dim();
  auto indices = self._indices();
  auto sizes = self.sizes().vec();
  sizes.insert(sizes.begin() + dim, 1);
  if (dim <= sparse_dim) {
    auto new_indices = native::cat({
      indices.narrow(0, 0, dim),
      native::zeros({1, indices.size(1)}, indices.options().dtype(kLong)),
      indices.narrow(0, dim, indices.size(0) - dim)
    });
    return _sparse_coo_tensor_with_dims_and_tensors(
        sparse_dim + 1, dense_dim, sizes, new_indices, self._values(), self.options());
  } else {
    return _sparse_coo_tensor_with_dims_and_tensors(
        sparse_dim, dense_dim + 1, sizes, indices, self._values().unsqueeze(dim - sparse_dim + 1), self.options());
  }
}

Tensor unsqueeze_qtensor(const Tensor& self, int64_t dim) {
  dim = maybe_wrap_dim(dim, self.dim() + 1);
  auto g = inferUnsqueezeGeometry(self, dim);
  auto quantizer = get_qtensorimpl(self)->quantizer();
  if (quantizer->qscheme() == QScheme::PER_CHANNEL_AFFINE) {
    const auto* per_channel_quantizer = static_cast<at::PerChannelAffineQuantizer*>(quantizer.get());
    auto axis = per_channel_quantizer->axis();
    if (axis >= dim) {
      axis += 1;
    }
    quantizer = make_per_channel_affine_quantizer(per_channel_quantizer->scales(),
                                                  per_channel_quantizer->zero_points(),
                                                  axis,
                                                  quantizer->scalar_type());
  }
  return make_qtensor(self, std::get<0>(g), std::get<1>(g), quantizer);
}

Tensor unsqueeze(const Tensor& self, int64_t dim) {
  dim = maybe_wrap_dim(dim, self.dim() + 1);

  if (self.is_sparse()) {
    return unsqueeze_sparse(self, dim);
  } else if (self.is_quantized()) {
    return unsqueeze_qtensor(self, dim);
  } else {
    auto g = inferUnsqueezeGeometry(self, dim);
    return self.as_strided(std::get<0>(g), std::get<1>(g));
  }
}

Tensor & unsqueeze_(Tensor& self, int64_t dim) {
  dim = maybe_wrap_dim(dim, self.dim() + 1);

  auto g = inferUnsqueezeGeometry(self, dim);
  return self.as_strided_(std::get<0>(g), std::get<1>(g));
}

Tensor flatten(const Tensor& self, int64_t start_dim, int64_t end_dim) {
  start_dim = maybe_wrap_dim(start_dim, self.dim());
  end_dim = maybe_wrap_dim(end_dim, self.dim());
  TORCH_CHECK(start_dim <= end_dim, "flatten() has invalid args: start_dim cannot come after end_dim");
  std::vector<int64_t> shape;

  if (self.dim() == 0) {
    return self.reshape({1});
  }

  if (start_dim == end_dim) {
    return self;
  }

  // We don't want to infer_size on the entire shape, because that can give us an extra degree
  // of freedom we don't want; for example, consider shape [0, 1, 3, 0], with start_dim=1, end_dim=2.
  // It's clear we want result shape [0, 3, 0] but passing [0, -1, 0] to infer_size means the -1
  // can take on any value and satisfy the constraints.
  auto slice_numel = prod_intlist(self.sizes().slice(start_dim, end_dim - start_dim + 1));
  shape.reserve(self.dim() - end_dim + start_dim);
  for (int64_t i = 0; i < start_dim; i++) {
    shape.push_back(self.size(i));
  }
  shape.push_back(slice_numel);
  for (int64_t i = end_dim + 1; i < self.dim(); i++) {
    shape.push_back(self.size(i));
  }

  return self.reshape(shape);
}

Tensor flatten(const Tensor& self, int64_t start_dim, int64_t end_dim, Dimname out_dim) {
  auto outnames = self.names().vec();
  outnames.erase(outnames.begin() + start_dim, outnames.begin() + end_dim + 1);
  outnames.insert(outnames.begin() + start_dim, out_dim);

  Tensor result;
  {
    NoNamesGuard guard;
    result = native::flatten(self, start_dim, end_dim);
  }
  internal_set_names_inplace(result, outnames);
  return result;
}

Tensor flatten(const Tensor& self, Dimname start_dim, Dimname end_dim, Dimname out_dim) {
  auto start_pos = dimname_to_position(self, start_dim);
  auto end_pos  = dimname_to_position(self, end_dim);
  return native::flatten(self, start_pos, end_pos, out_dim);
}

Tensor flatten(const Tensor& self, DimnameList dims, Dimname out_dim) {
  auto positions = dimnames_to_positions(self, dims);
  for (size_t i = 0; i < positions.size() - 1; i++) {
    if (positions[i] + 1 == positions[i + 1]) continue;
    TORCH_CHECK(positions[i] + 1 == positions[i + 1],
        "flatten(tensor, dims, out_dim): dims ", dims, " must be consecutive ",
        "in Tensor", self.names());
  }
  return native::flatten(self, *dims.begin(), *(dims.end() - 1), out_dim);
}

Tensor ravel(const Tensor& self) {
  return self.reshape(-1);
}

Tensor unflatten(const Tensor& self, int64_t dim, IntArrayRef sizes, c10::optional<DimnameList> names) {
  dim = maybe_wrap_dim(dim, self.dim());

  TORCH_CHECK(sizes.size() > 0, "unflatten: sizes must be non-empty");
  TORCH_INTERNAL_ASSERT(!names || names->size() == sizes.size());

  const int64_t numel = prod_intlist(sizes);
  if (self.has_names()) {
    TORCH_CHECK(numel == self.size(dim),
      "unflatten: Provided sizes ", sizes, " don't multiply up to the size of dim ",
      dim, " (", self.names()[dim], ": ", self.size(dim), ") in Tensor", self.names());
    TORCH_CHECK(names, "unflatten: input is a named tensor but no names were given for unflattened sizes");
  } else {
    TORCH_CHECK(numel == self.size(dim),
      "unflatten: Provided sizes ", sizes, " don't multiply up to the size of dim ",
      dim, " (", self.size(dim), ") in the input tensor");
  }

  DimVector shape(self.sizes().begin(), self.sizes().end());
  shape.erase(shape.begin() + dim);
  shape.insert(shape.begin() + dim, sizes.begin(), sizes.end());

  Tensor result;
  {
    NoNamesGuard guard;
    result = self.view(shape);
  }

  if (names) {
    auto outnames = self.names().vec();
    outnames.erase(outnames.begin() + dim);
    outnames.insert(outnames.begin() + dim, names->begin(), names->end());
    at::internal_set_names_inplace(result, outnames);
  }

  return result;
}

Tensor unflatten(const Tensor& self, Dimname dim, IntArrayRef sizes, DimnameList names) {
  return native::unflatten(self, dimname_to_position(self, dim), sizes, names);
}

Tensor view_as(const Tensor& self, const Tensor& other) {
  return self.view(other.sizes());
}

int64_t numel(const Tensor& self) {
  return self.unsafeGetTensorImpl()->numel();
}

std::vector<Tensor> unbind(const Tensor &self, int64_t dim) {
  dim = maybe_wrap_dim(dim, self.dim());
  int64_t size = self.size(dim);
  std::vector<Tensor> tensors(size);
  for (int i = 0; i < size; i++) {
    tensors[i] = self.select(dim, i);
  }
  return tensors;
}

std::vector<Tensor> unbind(const Tensor& self, Dimname dim) {
  return at::unbind(self, dimname_to_position(self, dim));
}

std::vector<Tensor> meshgrid(TensorList tensors) {
  int64_t size = tensors.size();
  TORCH_CHECK(size > 0, "meshgrid expects a non-empty TensorList");
  std::vector<int64_t> shape(size);
  for(int64_t i = 0; i < size; i++) {
    switch (tensors[i].dim()) {
    case 0:
      shape[i] = 1;
      break;
    case 1:
      shape[i] = tensors[i].size(0);
      break;
    default:
      AT_ERROR("Expected scalar or 1D tensor in the tensor list but got: ", tensors[i]);
    }
  }
  for(int64_t i = 0; i < size - 1; i++){
      TORCH_CHECK(tensors[i].dtype() == tensors[i+1].dtype(), "meshgrid expects all tensors to have the same dtype");
      TORCH_CHECK(tensors[i].device() == tensors[i+1].device(), "meshgrid expects all tensors to have the same device");
  }
  std::vector<Tensor> grids;
  for(int64_t i = 0; i < size; i++) {
    std::vector<int64_t> view_shape(size, 1);
    view_shape[i] = -1;
    grids.push_back(tensors[i].view(view_shape).expand(shape));
  }
  return grids;
}

// Numpy-style `a.T`: returns the tensor
// with dims reversed
Tensor numpy_T(const Tensor &self) {
  int64_t n = self.dim();
  DimVector transpose_dims;
  for (int64_t i = n - 1; i >= 0; --i) {
    transpose_dims.push_back(i);
  }
  return self.permute(transpose_dims);
}

Tensor view(const Tensor& self, IntArrayRef size) {
  auto inferred_size = at::infer_size(size, self.numel());
  auto stride = at::detail::computeStride(self.sizes(),
                                          self.strides(),
                                          inferred_size);
  TORCH_CHECK(stride.has_value(), "view size is "
    "not compatible with input tensor's size and stride (at least one dimension"
    " spans across two contiguous subspaces). Use .reshape(...) instead.");
  auto stride_value = *stride;
  return alias_with_sizes_and_strides(self, inferred_size, stride_value);
}

Tensor alias(const Tensor& self) {
    return alias_with_sizes_and_strides(self, self.sizes(), self.strides());
}

Tensor unfold(const Tensor& self, int64_t dimension, int64_t size, int64_t step) {
  // some special handling to deal with allow dimension == 0 when self.dim() == 0
  dimension = at::maybe_wrap_dim(dimension, self.dim(), /*wrap_scalar=*/true);
  int64_t max_size = self.dim() == 0 ? 1 : self.size(dimension);
  TORCH_CHECK(size <= max_size, "maximum size for tensor at dimension ", dimension,
                                " is ", max_size, " but size is ", size);
  TORCH_CHECK(step > 0, "step is ", step, " but must be > 0");

  std::vector<int64_t> new_size(self.dim() + 1);
  std::vector<int64_t> new_stride(self.dim() + 1);

  new_size[self.dim()] = size;
  new_stride[self.dim()] = self.dim() == 0 ? 1 : self.stride(dimension);
  for(int64_t d = 0; d < self.dim(); d++) {
    auto self_size = self.size(d);
    auto self_stride = self.stride(d);
    if(d == dimension) {
      new_size[d] = (self_size - size) / step + 1;
      new_stride[d] = step*self_stride;
    } else {
      new_size[d] = self_size;
      new_stride[d] = self_stride;
    }
  }

  return self.as_strided(new_size, new_stride);
}

template <typename scalar_t>
void apply_diag(Tensor& result, const Tensor& self, int64_t dimension) {
  TORCH_CHECK(self.dim() == 1 || self.dim() == 2, "matrix or a vector expected");

  int64_t i;
  auto self_data = self.data_ptr<scalar_t>();
  if (self.dim() == 1) {
    auto self_size = self.size(0);
    auto self_stride = self.stride(0);
    int64_t sz = self_size + std::abs(dimension);

    result.resize_({sz, sz});
    result.zero_();
    auto r_data = result.data_ptr<scalar_t>();
    auto r_stride_0 = result.stride(0);
    auto r_stride_1 = result.stride(1);
    r_data += (dimension >= 0 ? dimension*r_stride_1 : -dimension*r_stride_0);

    for (i = 0; i < self_size; i++) {
      r_data[i * (r_stride_0 + r_stride_1)] = self_data[i * self_stride];
    }
  } else {
    auto self_stride_0 = self.stride(0);
    auto self_stride_1 = self.stride(1);

    int64_t sz;
    if (dimension >= 0) {
      sz = std::min(self.size(0), self.size(1) - dimension);
    } else {
      sz = std::min(self.size(0) + dimension, self.size(1));
    }

    result.resize_({sz});
    result.zero_();
    auto r_data = result.data_ptr<scalar_t>();
    auto r_stride_0 = result.stride(0);
    self_data += (dimension >= 0 ? dimension * self_stride_1 : -dimension * self_stride_0);
    for (i = 0; i < sz; i++) {
      r_data[i * r_stride_0] = self_data[i * (self_stride_0 + self_stride_1)];
    }
  }
}

Tensor diag(const Tensor& self, int64_t dimension) {
  Tensor result = at::empty({0}, self.options());
  at::diag_out(result, self, dimension);
  return result;
}

Tensor& diag_cpu_out(Tensor &result, const Tensor& self, int64_t dimension) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND(at::ScalarType::Bool, self.scalar_type(), "diag", [&] {
    apply_diag<scalar_t>(result, self, dimension);
  });
  return result;
}

Tensor diag_backward(const Tensor& grad, IntArrayRef input_sizes, int64_t diagonal) {
  auto ndimension = input_sizes.size();
  AT_ASSERT(ndimension == 1 || ndimension == 2);

  if (ndimension == 1 || input_sizes[0] == input_sizes[1]) {
    return grad.diag(diagonal);
  }

  // Input was a matrix but was not square
  auto grad_input = at::zeros(input_sizes, grad.options());
  auto diag = grad_input.diagonal(diagonal);
  diag.copy_(grad);
  return grad_input;
}

Tensor diagonal_backward(const Tensor & grad, IntArrayRef input_sizes, int64_t offset, int64_t dim1, int64_t dim2) {
  auto grad_input = at::zeros(input_sizes, grad.options());
  auto diag = grad_input.diagonal(offset, dim1, dim2);
  diag.copy_(grad);
  return grad_input;
}

Tensor movedim(const Tensor& self, IntArrayRef src, IntArrayRef dst) {
  TORCH_CHECK(src.size() == dst.size(), "movedim: Invalid source or destination dims: source (",
              src, " dims ) should contain the same number of dims as destination (", dst, " dims)");

  size_t self_dim = self.dim();
  DimVector normalized_src(src.size());
  DimVector normalized_dst(dst.size());

  auto wrap_dims = [&self_dim](const IntArrayRef& vec, DimVector& normalized_vec) {
    for (int i = 0; i < vec.size(); i++) {
      normalized_vec[i] = maybe_wrap_dim(vec[i], self_dim);
    }
  };

  wrap_dims(src, normalized_src);
  wrap_dims(dst, normalized_dst);

  auto all_unique = [](const DimVector& dims) {
    DimVector copy = dims;
    std::sort(copy.begin(), copy.end());
    auto duplicate = std::adjacent_find(copy.begin(), copy.end());
    return duplicate == copy.end();
  };
  TORCH_CHECK(all_unique(normalized_src), "movedim: repeated dim in `source` (", src, ")");
  TORCH_CHECK(all_unique(normalized_dst), "movedim: repeated dim in `destination` (", dst, ")");

  // TODO: The algorithm below can probably be optimized.
  // Reference: https://github.com/pytorch/pytorch/pull/41480#discussion_r456100505

  // Algorithm Walkthrough
  // Example Input
  // Variable State:
  //     normalized_src = 0, 1
  //     normalized_dst = 2, 4
  //     self_dim = 5
  DimVector order(self_dim);
  DimVector source_dims(self_dim);
  DimVector destination_dims(self_dim);

  // We initialize two vectors to track update to the dims
  // `order` contains the final order of the dim positions.
  // Variable State:
  //     order = NA, NA, NA, NA, NA
  //     source_dims = 0, 1, 2, 3, 4
  //     destination_dims = 0, 1, 2, 3, 4
  std::iota(source_dims.begin(), source_dims.end(), 0);
  std::iota(destination_dims.begin(), destination_dims.end(), 0);

  // We mark and update position for the dim provided by user
  // i.e. `normalized_src` and `normalized_dims`
  // Variable State:
  //     order = NA, NA, 0, NA, 1
  //     source_dims = -1, -1, 2, 3, 4
  //     destination_dims = 0, 1, -1, 3, -1
  for (int64_t i = 0; i < src.size(); ++i) {
      order[normalized_dst[i]] = normalized_src[i];
      source_dims[normalized_src[i]] = -1;
      destination_dims[normalized_dst[i]] = -1;
  }

  // Remove the dims whose position we already know,
  // the ones marked with -1 in previous step
  // Variable State:
  //     source_dims = 2, 3, 4
  //     destination_dims = 0, 1, 3
  auto source_iter = std::remove(source_dims.begin(), source_dims.end(), -1);
  auto destination_iter = std::remove(destination_dims.begin(), destination_dims.end(), -1);

  int64_t rest_dim = self.dim() - src.size();
  TORCH_INTERNAL_ASSERT(std::distance(source_dims.begin(), source_iter)  == rest_dim);
  TORCH_INTERNAL_ASSERT(std::distance(destination_dims.begin(), destination_iter)  == rest_dim);

  // Update the position of the remaining dimensions.
  // `source_dims` now contains the original position
  // `destination_dims` contains the new position it will shifted to
  // after considering the user inputs.
  // Variable State:
  //     order = 2, 3, 0, 4, 1
  for (int64_t i = 0; i < rest_dim; ++i) {
      order[destination_dims[i]] = source_dims[i];
  }

  return self.permute(order);
}

Tensor movedim(const Tensor& self, int64_t src, int64_t dst) {
  return at::movedim(self, IntArrayRef{src}, IntArrayRef{dst});
}

Tensor moveaxis(const Tensor& self, IntArrayRef src, IntArrayRef dst) {
  return at::movedim(self, src, dst);
}

Tensor moveaxis(const Tensor& self, int64_t src, int64_t dst) {
  return at::movedim(self, IntArrayRef{src}, IntArrayRef{dst});
}

Tensor swapaxes(const Tensor& self, int64_t axis0, int64_t axis1) {
  return self.transpose(axis0, axis1);
}

Tensor& swapaxes_(Tensor& self, int64_t axis0, int64_t axis1) {
  return self.transpose_(axis0, axis1);
}

Tensor swapdims(const Tensor& self, int64_t dim0, int64_t dim1) {
  return self.transpose(dim0, dim1);
}

Tensor& swapdims_(Tensor& self, int64_t dim0, int64_t dim1) {
  return self.transpose_(dim0, dim1);
}

}} // at::native
