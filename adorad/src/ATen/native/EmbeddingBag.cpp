#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>
#include <ATen/TensorUtils.h>

#include <TH/THBlasUtils.h>

#ifdef USE_FBGEMM
#include <fbgemm/Fbgemm.h>
#else
#include <caffe2/perfkernels/embedding_lookup_idx.h>
#endif

#include <algorithm>
#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>
#include <tuple>
#include <vector>


namespace {
  const int MODE_SUM = 0;
  const int MODE_MEAN = 1;
  const int MODE_MAX = 2;
}

namespace at {
namespace native {

template<typename scalar_t>
scalar_t dot_impl(int64_t n, scalar_t *x, int64_t incx, scalar_t *y, int64_t incy);

static void make_offset2bag(const Tensor &offsets, Tensor& offset2bag) {
  offset2bag.index_add_(
      0, offsets, at::ones_like(offsets, LEGACY_CONTIGUOUS_MEMORY_FORMAT)); // offset2bag = [1 0 1 0 1]
  offset2bag[0] -= 1;                     // offset2bag = [0 0 1 0 1]
  offset2bag = offset2bag.cumsum(0, offset2bag.scalar_type());     // offset2bag = [0 0 1 1 2]
}

namespace {

bool isFastPathIndexSelect(const Tensor& src, Tensor& output) {
  return src.scalar_type() == kFloat && src.stride(1) == 1 && output.stride(1) == 1;
}

bool isFastPathIndexSelectScale(const Tensor& src, const Tensor& scale, Tensor& output) {
  return src.scalar_type() == kFloat && src.stride(1) == 1 && output.stride(1) == 1 && scale.stride(0) == 1;
}

// This function combines index_select (using select_indices as the index) and
// index_add (using add_indices as the index), without creating an intermediary
// tensor to hold the selected embeddings
template<typename data_t, typename index_t>
typename std::enable_if<!std::is_same<data_t, float>::value, void>::type
index_select_add(const Tensor &select_indices,
                             const Tensor &add_indices,
                             const Tensor &src,
                             Tensor &output,
                             const Tensor& /*offsets*/,
                             bool /*include_last_offset*/) {
  AT_ASSERT(select_indices.numel() == add_indices.numel());
  auto* add_indices_data = add_indices.data_ptr<index_t>();
  auto* select_indices_data = select_indices.data_ptr<index_t>();
  auto* src_data = src.data_ptr<data_t>();
  auto* output_data = output.data_ptr<data_t>();
  auto numel = add_indices.numel();
  int64_t ddim = src.size(1);
  auto src_stride0 = src.stride(0);
  auto src_stride1 = src.stride(1);
  auto output_stride0 = output.stride(0);
  auto output_stride1 = output.stride(1);

  for (int64_t i = 0; i < numel; i++) {
    THBlas_axpy<data_t>(ddim, 1,
            src_data + src_stride0 * select_indices_data[i], src_stride1,
            output_data + output_stride0 * add_indices_data[i], output_stride1);
  }
}

template<typename data_t, typename index_t>
typename std::enable_if<std::is_same<data_t, float>::value, void>::type
index_select_add(const Tensor &select_indices,
                             const Tensor &add_indices,
                             const Tensor &src,
                             Tensor &output,
                             const Tensor& offsets,
                             bool include_last_offset) {
  int64_t ddim = src.size(1);
  auto* select_indices_data = select_indices.data_ptr<index_t>();
  auto* output_data = output.data_ptr<float>();

  if (isFastPathIndexSelect(src, output)) {
    auto src_contig = src.contiguous();
    auto* src_data = src_contig.data_ptr<float>();
    int64_t output_size = offsets.numel() - 1;
    auto* offsets_data = offsets.data_ptr<index_t>();
    std::vector<index_t> offsets_include_last;

    if (include_last_offset) {
      output_size = offsets.numel() - 1;
    } else {
      output_size = offsets.numel();
      offsets_include_last.resize(offsets.numel() + 1);
      std::memcpy(
          offsets_include_last.data(),
          offsets.data_ptr<index_t>(),
          sizeof(index_t) * offsets.numel());
      offsets_include_last[offsets.numel()] = select_indices.numel();
      offsets_data = offsets_include_last.data();
    }

#ifdef USE_FBGEMM
    auto kernel_fp32_index_t =
      fbgemm::GenerateEmbeddingSpMDM<float, index_t, index_t>(
        /* block_size */ddim,
        /* has_weight */false,
        /* normalize_by_lengths */false,
        /* prefetch */16,
        /* is_weight_positional */false,
        /* use_offsets */true
      );
#endif
    at::parallel_for(
        0, output_size, 1, [&](index_t start_idx, index_t end_idx) {
#ifdef USE_FBGEMM
          kernel_fp32_index_t(
            /* output_size */end_idx - start_idx,
            /* index_size */offsets_data[end_idx] - offsets_data[start_idx],
            /* data_size */src.size(0),
            /* input */src_data,
            /* indices */select_indices_data + offsets_data[start_idx],
            /* offsets_or_lengths */offsets_data + start_idx,
            /* weights */nullptr,
            /* output */output_data + start_idx * ddim);
#else
          caffe2::EmbeddingLookupIdx(
              /*block_size=*/ddim,
              /*output_size=*/end_idx - start_idx,
              /*index_size=*/offsets_data[end_idx] - offsets_data[start_idx],
              /*data_size=*/src.size(0),
              /*input=*/src_data,
              /*indices=*/select_indices_data + offsets_data[start_idx],
              /*offsets=*/offsets_data + start_idx,
              /*weights=*/nullptr,
              /*scale_bias=*/nullptr,
              /*normalize_by_lengths=*/false,
              /*out=*/output_data + start_idx * ddim);
#endif
        });
  } else {
    AT_ASSERT(select_indices.numel() == add_indices.numel());
    auto* src_data = src.data_ptr<float>();
    auto* add_indices_data = add_indices.data_ptr<index_t>();
    auto src_stride0 = src.stride(0);
    auto src_stride1 = src.stride(1);
    auto output_stride0 = output.stride(0);
    auto output_stride1 = output.stride(1);
    auto numel = add_indices.numel();
    for (int64_t i = 0; i < numel; i++) {
      THBlas_axpy<float>(
          ddim,
          1,
          src_data + src_stride0 * select_indices_data[i],
          src_stride1,
          output_data + output_stride0 * add_indices_data[i],
          output_stride1);
    }
  }
}

// This function fuses the following three fns:
// index_select (using select_indices as the index)
// mul (scaling by per_sample_weights)
// index_add (using add_indices as the index)
template<typename data_t, typename index_t>
static typename std::enable_if<!std::is_same<data_t, float>::value, void>::type
index_select_scale_add(const Tensor &select_indices,
                                   const Tensor &add_indices,
                                   const Tensor &scale,
                                   const Tensor &src,
                                   Tensor &output,
                                   const Tensor& /*offsets*/,
                                   bool /*include_last_offset*/) {
  AT_ASSERT(select_indices.numel() == add_indices.numel());
  auto* add_indices_data = add_indices.data_ptr<index_t>();
  auto* select_indices_data = select_indices.data_ptr<index_t>();
  auto* src_data = src.data_ptr<data_t>();
  auto* output_data = output.data_ptr<data_t>();
  auto numel = add_indices.numel();
  int64_t ddim = src.size(1);
  auto src_stride0 = src.stride(0);
  auto src_stride1 = src.stride(1);
  auto output_stride0 = output.stride(0);
  auto output_stride1 = output.stride(1);

  auto* scale_data = scale.data_ptr<data_t>();
  auto scale_stride = scale.stride(0);

  for (int64_t i = 0; i < numel; i++) {
    auto* src_base = src_data + src_stride0 * select_indices_data[i];
    auto* output_base = output_data + output_stride0 * add_indices_data[i];
    auto scale = scale_data[i * scale_stride];
    for (int64_t j = 0; j < ddim; j++) {
      output_base[j * output_stride1] += src_base[j * src_stride1] * scale;
    }
  }
}

template<typename data_t, typename index_t>
typename std::enable_if<std::is_same<data_t, float>::value, void>::type
index_select_scale_add(const Tensor &select_indices,
                                          const Tensor &add_indices,
                                          const Tensor &scale,
                                          const Tensor &src,
                                          Tensor &output,
                                          const Tensor& offsets,
                                          bool include_last_offset) {
  int64_t ddim = src.size(1);
  auto* scale_data = scale.data_ptr<float>();
  auto* select_indices_data = select_indices.data_ptr<index_t>();
  auto* output_data = output.data_ptr<float>();

  if (isFastPathIndexSelectScale(src, scale, output)) {
    auto src_contig = src.contiguous();
    auto* src_data = src_contig.data_ptr<float>();
    int64_t output_size = offsets.numel() - 1;
    auto* offsets_data = offsets.data_ptr<index_t>();
    std::vector<index_t> offsets_include_last;

    if (include_last_offset) {
      output_size = offsets.numel() - 1;
    } else {
      output_size = offsets.numel();
      offsets_include_last.resize(offsets.numel() + 1);
      std::memcpy(
          offsets_include_last.data(),
          offsets.data_ptr<index_t>(),
          sizeof(index_t) * offsets.numel());
      offsets_include_last[offsets.numel()] = select_indices.numel();
      offsets_data = offsets_include_last.data();
    }

#ifdef USE_FBGEMM
    auto kernel_fp32_index_t =
      fbgemm::GenerateEmbeddingSpMDM<float, index_t, index_t>(
        /* block_size */ddim,
        /* has_weight */true,
        /* normalize_by_lengths */false,
        /* prefetch */16,
        /* is_weight_positional */false,
        /* use_offsets */true
      );
#endif
    at::parallel_for(
        0, output_size, 1, [&](index_t start_idx, index_t end_idx) {
#ifdef USE_FBGEMM
          kernel_fp32_index_t(
            /* output_size */end_idx - start_idx,
            /* index_size */offsets_data[end_idx] - offsets_data[start_idx],
            /* data_size */src.size(0),
            /* input */src_data,
            /* indices */select_indices_data + offsets_data[start_idx],
            /* offsets_or_lengths */offsets_data + start_idx,
            /* weights */scale_data + offsets_data[start_idx],
            /* output */output_data + start_idx * ddim);
#else
          caffe2::EmbeddingLookupIdx(
              /*block_size=*/ddim,
              /*output_size=*/end_idx - start_idx,
              /*index_size=*/offsets_data[end_idx] - offsets_data[start_idx],
              /*data_size=*/src.size(0),
              /*input=*/src_data,
              /*indices=*/select_indices_data + offsets_data[start_idx],
              /*offsets=*/offsets_data + start_idx,
              /*weights=*/scale_data + offsets_data[start_idx],
              /*scale_bias=*/nullptr,
              /*normalize_by_lengths=*/false,
              /*out=*/output_data + start_idx * ddim);
#endif
        });
  } else {
    AT_ASSERT(select_indices.numel() == add_indices.numel());
    auto* src_data = src.data_ptr<float>();
    auto* add_indices_data = add_indices.data_ptr<index_t>();
    auto src_stride0 = src.stride(0);
    auto src_stride1 = src.stride(1);
    auto output_stride0 = output.stride(0);
    auto output_stride1 = output.stride(1);
    auto scale_stride = scale.stride(0);
    auto numel = add_indices.numel();


    for (int64_t i = 0; i < numel; i++) {
      auto* src_base = src_data + src_stride0 * select_indices_data[i];
      auto* output_base = output_data + output_stride0 * add_indices_data[i];
      auto scale = scale_data[i * scale_stride];
      for (int64_t j = 0; j < ddim; j++) {
        output_base[j * output_stride1] += src_base[j * src_stride1] * scale;
      }
    }
  }
}

}  // namespace

static at::Tensor make_bag_size(
    const Tensor& offsets,
    const Tensor& indices,
    const int64_t mode,
    const bool requires_grad) {
  at::Tensor bag_size;
  if (mode == MODE_MEAN || mode == MODE_MAX) {
    bag_size = at::zeros(offsets.sizes(), offsets.options());
    // Compute this for MODE_MEAN and MODE_MAX (latter needed for backwards)
    if (offsets.size(0) != 1) {
      bag_size.slice(0, 0, bag_size.size(0) - 1, 1) =
          offsets.slice(0, 1, offsets.size(0), 1) -
          offsets.slice(0, 0, offsets.size(0) - 1, 1);
    }
    bag_size[-1] = indices.size(0) - offsets[-1];
  } else if (requires_grad) {
    // in MODE_SUM, only allocate bag_size if we need gradients
    bag_size = at::empty(offsets.sizes(), offsets.options());
  }
  return bag_size;
}

static Tensor apply_bag_size(const Tensor &offsets, const Tensor &indices,
                             const int64_t mode, Tensor &output,
                             const Tensor &bag_size) {
  if (mode == MODE_MEAN) {
    // Avoid dividing by 0 for empty bags.
    // Instead we want empty bags to return all 0s
    if (offsets.size(0) == 1) {
      auto bag_size_ = std::max(indices.size(0), static_cast<int64_t>(1));
      output /= bag_size_;
    } else {
      auto bag_size_ = at::max(bag_size, at::ones_like(bag_size, LEGACY_CONTIGUOUS_MEMORY_FORMAT))
                           .to(output.options())
                           .unsqueeze(1)
                           .expand_as(output);
      output /= bag_size_;
    }
  }
  return output;
}

static Tensor apply_bag_size_backward(const Tensor &offsets,
                                      const Tensor &indices, const int64_t mode,
                                      Tensor &output, const Tensor &offset2bag,
                                      const Tensor &bag_size) {
  if (mode == MODE_MEAN) {
    if (offsets.size(0) == 1) {
      auto bag_size_ = indices.size(0);
      output /= bag_size_;
    } else {
      auto inv_bag_size_ = (1 / bag_size.to(output.options()))
                             .unsqueeze(1)
                             .index_select(0, offset2bag);
      output *= inv_bag_size_;
    }
  }
  return output;
}

template <typename scalar_t>
std::tuple<Tensor, Tensor, Tensor, Tensor> embedding_bag_cpu_max(
    const Tensor& weight,
    const Tensor& indices,
    const Tensor& offset2bag,
    const Tensor& output,
    const Tensor& bag_size,
    const Tensor& offsets,
    bool include_last_offset) {
  int64_t numIndices = indices.numel();
  int64_t numBags = offsets.size(0);
  int64_t featureSize = weight.size(1);
  if (include_last_offset) {
    // Check https://github.com/pytorch/pytorch/issues/29019
    // We plan to add one more element in offsets, which is equal to the size of
    // indices. Currently for cuda devices, we still use the legacy
    // implementation even this flag is enabled.
    TORCH_CHECK(
        numBags >= 1, "include_last_offset: numBags should be at least 1");
    numBags -= 1;
  }
  auto max_indices =
      at::zeros({numBags, featureSize}, indices.options());
  AT_DISPATCH_INDEX_TYPES(indices.scalar_type(), "embedding_bag_cpu_max", [&] {
    auto* indices_data = indices.data_ptr<index_t>();
    auto* offset2bag_data = offset2bag.data_ptr<index_t>();

    auto* max_indices_data = max_indices.data_ptr<index_t>();
    auto max_indices_stride = max_indices.stride(0);

    auto* weight_data = weight.data_ptr<scalar_t>();
    auto* output_data = output.data_ptr<scalar_t>();
    auto weight_stride0 = weight.stride(0);
    auto weight_stride1 = weight.stride(1);
    auto output_stride = output.stride(0);

    for (int i = 0; i < numIndices; ++i) {
      auto bag = offset2bag_data[i];
      auto word_idx = indices_data[i];

      for (int dim = 0; dim < featureSize; dim++) {
        auto& current_item = output_data[output_stride * bag + dim];
        auto weight_item =
            weight_data[weight_stride0 * word_idx + dim * weight_stride1];
        bool is_first_for_bag = (i == 0) || offset2bag_data[i - 1] != bag;

        if (is_first_for_bag || weight_item > current_item) {
          current_item = weight_item;
          max_indices_data[max_indices_stride * bag + dim] = word_idx;
        }
      }
    }
  });

  return std::tuple<Tensor, Tensor, Tensor, Tensor>(
      output, offset2bag, bag_size, max_indices);
}

// Assumes all input tensors except for `weight` are contiguous.
// See NOTE [ embedding_bag Native Functions ] in native_functions.yaml for details
std::tuple<Tensor, Tensor, Tensor, Tensor> _embedding_bag_cpu_impl(
    const Tensor& weight,
    const Tensor& indices,
    const Tensor& offsets,
    const int64_t mode,
    const Tensor& per_sample_weights,
    bool include_last_offset,
    bool requires_grad) {
  auto indices_arg = TensorArg(indices, "indices", 1);
  checkScalarTypes("embedding_bag", indices_arg, {kLong, kInt});
  auto offsets_arg = TensorArg(offsets, "offsets", 1);
  checkScalarTypes("embedding_bag", offsets_arg, {kLong, kInt});
  checkSameType("embedding_bag", indices_arg, offsets_arg);
  auto weight_arg = TensorArg(weight, "weight", 1);
  checkScalarTypes("embedding_bag", weight_arg, {kFloat, kDouble});

  AT_DISPATCH_INDEX_TYPES(offsets.scalar_type(), "_embedding_bag_cpu_impl", [&]() {
    index_t offset_0 = offsets.data_ptr<index_t>()[0];
    index_t offset_n = offsets.data_ptr<index_t>()[offsets.size(0)-1];
    TORCH_CHECK(offset_0 == 0, "offsets[0] has to be 0, i.e., the first sequence "
                              "in the mini-batch has to start from position 0. "
                              "However, got ", offsets[0]);
    TORCH_CHECK(offset_n <= indices.size(0), "offsets[-1] can not "
                "be greater than input's length ", indices.size(0), " but got offsets[-1] of ",
                offset_n);
  });

  if (per_sample_weights.defined()) {
    TORCH_CHECK(mode == MODE_SUM,
        "embedding_bag: per_sample_weights only supported with mode='sum'");
    auto per_input_weights_arg = TensorArg(
        per_sample_weights,"per_sample_weights", 1);
    checkSameType("embedding_bag", weight_arg, per_input_weights_arg);
    TORCH_CHECK(per_sample_weights.dim() == 1);
    TORCH_CHECK(per_sample_weights.numel() == indices.numel());
  }


  at::Tensor bag_size;
  if (include_last_offset) {
    // TODO: make_bag_size can be optimized to do less temporary tensors (with
    // include_last_offset).
    bag_size = make_bag_size(offsets.slice(0, 0, offsets.size(0) - 1, 1), indices, mode, requires_grad);
  } else {
    bag_size = make_bag_size(offsets, indices, mode, requires_grad);
  }

  if (include_last_offset) {
    TORCH_CHECK(
        offsets.size(0) >= 1,
        "include_last_offset: number of offset should be at least 1");
  }

  auto output = at::empty(
      {include_last_offset ? offsets.size(0) - 1 : offsets.size(0),
       weight.size(1)},
      weight.options());

  // To save compute, if we are going to go down the fast path case for the 'sum'
  // mode, we skip calculating offset2bag, since it is not going to be used.
  auto fast_path_sum = [&weight, &per_sample_weights, &output]() {
    if (per_sample_weights.defined()) {
      return isFastPathIndexSelectScale(weight, per_sample_weights, output);
    } else {
      return isFastPathIndexSelect(weight, output);
    }
  };

  // Use an empty 0-element tensor as a sentinel that we have skipped the
  // creation of offset2bag because autograd chokes when trying to use an
  // undefined tensor as an input to a backward op.
  Tensor offset2bag = at::empty({0}, offsets.options());
  if (mode == MODE_MEAN || mode == MODE_MAX || !fast_path_sum()) {
    // If the last entries are empty, that the last offsets are irrelevant as they
    // won't change anything in the assignment of ID -> bag, but index_add would
    // throw out of bounds error. So to keep it simple we just add one more
    // entry to the end then get rid of it after make_offset2bag.
    offset2bag = at::zeros(
       {indices.sizes()[0] + 1}, offsets.options()); // offset2bag = [0 0 0 0 0]

    make_offset2bag(offsets, offset2bag);

    offset2bag.resize_({indices.sizes()[0]});

    // only initialize output in slow path
    output.zero_();
  }

  if (mode == MODE_MEAN || mode == MODE_SUM) {
    // explicitly capture all required variables to work around windows build
    // TODO: fix this when windows can correctly capture variables in nested lambda
    AT_DISPATCH_FLOATING_TYPES(weight.scalar_type(), "embedding_bag_cpu",
      [&indices, &offset2bag, &per_sample_weights, &weight, &output, &offsets, &include_last_offset, &mode]() {
      AT_DISPATCH_INDEX_TYPES(indices.scalar_type(), "embedding_bag_cpu",
        [&indices, &offset2bag, &per_sample_weights, &weight, &output, &offsets, &include_last_offset, &mode]() {
        if (per_sample_weights.defined()) {
          AT_ASSERT(mode == MODE_SUM);
          index_select_scale_add<scalar_t, index_t>(
              indices, offset2bag, per_sample_weights, weight, output, offsets, include_last_offset);
        } else {
          index_select_add<scalar_t, index_t>(indices, offset2bag, weight, output, offsets, include_last_offset);
        }
      });
    });
    auto ret = apply_bag_size(offsets, indices, mode, output, bag_size);
    return std::tuple<Tensor, Tensor, Tensor, Tensor>(ret, offset2bag, bag_size, bag_size);
  } else { // MODE_MAX
    at::optional<Tensor> maybe_per_sample_weights;
    if (per_sample_weights.defined()) {
      maybe_per_sample_weights = per_sample_weights;
    }
    return AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      weight.scalar_type(), "embedding_bag_cpu_max", [&]() {
        return embedding_bag_cpu_max<scalar_t>(
            weight, indices, offset2bag, output, bag_size, offsets, include_last_offset);
      }
    );
  }
}

// embedding_bag wrapper to enforce contiguity in tensors other than `weight`.
// This is created to save extra `.contiguous()` call in backward.
// See NOTE [ embedding_bag Native Functions ] in native_functions.yaml for details
std::tuple<Tensor, Tensor, Tensor, Tensor>
embedding_bag(const Tensor &weight, const Tensor &indices,
              const Tensor &offsets, const bool scale_grad_by_freq,
              const int64_t mode, bool sparse,
              const Tensor &per_sample_weights,
              bool include_last_offset) {
  if (!weight.requires_grad()) {
    return at::_embedding_bag_forward_only(weight, indices.contiguous(), offsets.contiguous(),
                              scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset);
  }

  return at::_embedding_bag(weight, indices.contiguous(), offsets.contiguous(),
                            scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset);
};

// Assumes all input tensors except for `weight` are contiguous.
// See NOTE [ embedding_bag Native Functions ] in native_functions.yaml for details
std::tuple<Tensor, Tensor, Tensor, Tensor>
_embedding_bag_forward_only_cpu(const Tensor &weight, const Tensor &indices,
                  const Tensor &offsets, const bool scale_grad_by_freq,
                  const int64_t mode, bool sparse,
                  const Tensor &per_sample_weights, bool include_last_offset) {
  std::ignore = scale_grad_by_freq;
  std::ignore = sparse;
  return _embedding_bag_cpu_impl(
      weight,
      indices,
      offsets,
      mode,
      per_sample_weights,
      include_last_offset,
      /*requires_grad=*/false);
}

// Assumes all input tensors except for `weight` are contiguous.
// See NOTE [ embedding_bag Native Functions ] in native_functions.yaml for details
std::tuple<Tensor, Tensor, Tensor, Tensor>
_embedding_bag_cpu(const Tensor &weight, const Tensor &indices,
                  const Tensor &offsets, const bool scale_grad_by_freq,
                  const int64_t mode, bool sparse,
                  const Tensor &per_sample_weights, bool include_last_offset) {
  std::ignore = scale_grad_by_freq;
  std::ignore = sparse;
  return _embedding_bag_cpu_impl(
      weight,
      indices,
      offsets,
      mode,
      per_sample_weights,
      include_last_offset,
      /*requires_grad=*/true);
}

// Assumes all input tensors are contiguous.
// See NOTE [ embedding_bag Native Functions ] in native_functions.yaml for details
Tensor _embedding_bag_backward(const Tensor &grad, const Tensor &indices,
                              const Tensor &offsets,
                              const Tensor &offset2bag,
                              const Tensor &bag_size_,
                              const Tensor &max_indices_,
                              int64_t num_weights,
                              bool scale_grad_by_freq, int64_t mode,
                              bool sparse,
                              const Tensor& per_sample_weights) {
  auto indices_arg = TensorArg(indices, "indices", 1);
  checkScalarTypes("embedding_bag", indices_arg, {kLong, kInt});
  checkContiguous("embedding_bag", indices_arg);
  auto offsets_arg = TensorArg(offsets, "offsets", 1);
  checkScalarTypes("embedding_bag", offsets_arg, {kLong, kInt});
  checkSameType("embedding_bag", indices_arg, offsets_arg);
  checkContiguous("embedding_bag", offsets_arg);

  Tensor offset2bag_;
  if (indices.numel() != 0 && offset2bag.numel() == 0) {
    offset2bag_ = at::zeros(
       {indices.sizes()[0] + 1}, offsets.options()); // offset2bag = [0 0 0 0 0]

    make_offset2bag(offsets, offset2bag_);

    offset2bag_.resize_({indices.sizes()[0]});
  } else {
    auto offset2bag_arg = TensorArg(offset2bag, "offset2bag", 1);
    checkScalarTypes("embedding_bag", offset2bag_arg, {kLong, kInt});
    checkContiguous("embedding_bag", offset2bag_arg);
    offset2bag_ = offset2bag;
  }

  if (sparse) {
    return at::_embedding_bag_sparse_backward(
        grad, indices, offsets, offset2bag_, bag_size_, num_weights,
        scale_grad_by_freq, mode, per_sample_weights);
  } else {
    return at::_embedding_bag_dense_backward(
        grad, indices, offsets, offset2bag_, bag_size_, max_indices_, num_weights,
        scale_grad_by_freq, mode, per_sample_weights);
  }
}

static Tensor _embedding_bag_dense_backward_cpu_max(
    const Tensor& grad,
    const Tensor& bag_size,
    const Tensor& max_indices,
    int64_t num_weights) {
  AT_ASSERT(max_indices.defined());
  auto index_grad_weight =
      at::zeros({num_weights, grad.size(1)}, grad.options());
  auto nonempty_max_indices = max_indices.index_select(0, bag_size.nonzero().view(-1));
  auto nonempty_grad = grad.index_select(0, bag_size.nonzero().view(-1));

  for (int64_t dim = 0; dim < grad.size(1); dim++) {
    index_grad_weight.select(1, dim).index_add_(
      0, nonempty_max_indices.select(1, dim), nonempty_grad.select(1, dim));
  }
  return index_grad_weight;
}

template<typename index_t>
static std::vector<index_t> compute_counts(
    int64_t num_weights,
    index_t* indices_data,
    int64_t indices_length) {
  std::vector<index_t> counts(num_weights, 0);
  for (int i = 0; i < indices_length; i++) {
    counts[indices_data[i]]++;
  }
  return counts;
}

// counts_uniq stores the index of the NEXT unique element
// of the (sorted) indices vector.
//
// For example:
// indices: [0, 0, 0, 1, 3, 3, 4]
// counts: [3, 1, 0, 2, 1, 0]
// counts_uniq: [3, 4, 6, 7]
//
// The unique indices can be found at index 0, 3, 4, 6.
template<typename index_t>
static std::vector<index_t> compute_counts_uniq(
    int64_t num_weights,
    index_t* indices_data,
    int64_t indices_length,
    const std::vector<index_t>& counts) {
  std::vector<index_t> counts_uniq;
  counts_uniq.reserve(num_weights);
  int64_t o = 0;
  for (int64_t i = 0; i < indices_length; i += counts[indices_data[i]]) {
    counts_uniq.push_back(counts[indices_data[i]]);
    if (o > 0) {
      counts_uniq[o] += counts_uniq[o - 1];
    }
    o++;
  }
  return counts_uniq;
}

template <typename scalar_t>
void _embedding_bag_dense_backward_cpu_sum_mean(
    const Tensor& grad,
    const Tensor& indices_,
    const Tensor& offsets_,
    const Tensor& offset2bag__,
    int64_t num_weights,
    bool scale_grad_by_freq,
    int64_t mode,
    const Tensor& per_sample_weights_,
    Tensor& index_grad_weight) {

  Tensor &offset2bag_ = const_cast<Tensor &>(offset2bag__);

  auto ind_sort_ = indices_.sort();
  auto indices = std::get<0>(ind_sort_);
  auto ind_sort = std::get<1>(ind_sort_);
  auto offset2bag = offset2bag_.index_select(0, ind_sort);

  optional<Tensor> per_sample_weights;
  scalar_t* per_sample_weights_data;
  optional<int64_t> per_sample_weights_stride;
  if (per_sample_weights_.defined()) {
    per_sample_weights = per_sample_weights_.index_select(0, ind_sort);
    per_sample_weights_data = per_sample_weights->data_ptr<scalar_t>();
    per_sample_weights_stride = per_sample_weights->stride(0);
  }

  int64_t numel = indices.numel();

  // explicitly capture all required variables to work around windows build
  // TODO: fix this when windows can correctly capture variables in nested lambda
  AT_DISPATCH_INDEX_TYPES(indices.scalar_type(), "_embedding_bag_dense_backward_cpu_sum_mean",
    [&indices, &offsets_, &offset2bag, &num_weights, &numel, &per_sample_weights,
      &per_sample_weights_data, &per_sample_weights_stride, &mode, &scale_grad_by_freq,
      &grad, &index_grad_weight] {
    auto* indices_data = indices.data_ptr<index_t>();
    auto* offsets_data = offsets_.data_ptr<index_t>();
    auto* offset2bag_data = offset2bag.data_ptr<index_t>();

    auto counts = compute_counts(num_weights, indices_data, numel);
    auto next_unique_index_idx =
        compute_counts_uniq(num_weights, indices_data, numel, counts);

    auto loop =
      [&next_unique_index_idx, &indices_data, &offset2bag_data, &per_sample_weights,
        &mode, &per_sample_weights_data, &per_sample_weights_stride, &scale_grad_by_freq,
        &counts, &offsets_, &indices, &offsets_data, &grad, &index_grad_weight](index_t start, index_t end) {
      for (index_t i = start; i < end; i++) {
        index_t start = i == 0 ? 0 : next_unique_index_idx[i - 1];
        index_t index = indices_data[start];
        for (index_t j = start; j < next_unique_index_idx[i]; j++) {
          index_t source = offset2bag_data[j];
          double scale = 1.0;
          if (per_sample_weights) {
            AT_ASSERT(mode == MODE_SUM);
            scale = per_sample_weights_data[*per_sample_weights_stride * j];
          }
          if (scale_grad_by_freq) {
            scale /= counts[indices_data[i]];
          }
          if (mode == 1) { // MODE_MEAN
            if (offsets_.size(0) == 1) {
              auto bag_size = indices.size(0);
              scale /= bag_size;
            } else {
              if (source == offsets_.size(0) - 1) {
                scale /= indices.size(0) - offsets_data[offsets_.size(0) - 1];
              } else {
                scale /= offsets_data[source + 1] - offsets_data[source];
              }
            }
          }
          int64_t ddim = grad.size(1);
          auto igwd = index_grad_weight.data_ptr<scalar_t>();
          auto gd = grad.data_ptr<scalar_t>();
          THBlas_axpy<scalar_t>(ddim, (scalar_t)scale, gd + ddim * source, 1,
                      igwd + ddim * index, 1);
        }
      }
    };

    if (numel > 1000) {
      at::parallel_for(0, (int64_t)next_unique_index_idx.size(), 0, loop);
    } else {
      loop(0, (int64_t)next_unique_index_idx.size());
    }
  });
}

Tensor _embedding_bag_dense_backward_cpu(const Tensor &grad_, const Tensor &indices_,
                                  const Tensor &offsets_,
                                  const Tensor &offset2bag__,
                                  const Tensor &bag_size_,
                                  const Tensor& max_indices_, int64_t num_weights,
                                  bool scale_grad_by_freq, int64_t mode,
                                  const Tensor& per_sample_weights_) {
  // indices_, offsets_ and offset2bag__ are assumed having correct dtypes and
  // contiguous here due to the checks in _embedding_bag_backward above.
  // Also see NOTE [ embedding_bag Native Functions ] in native_functions.yaml
  // for more details.
  auto grad = grad_.contiguous();
  auto grad_arg = TensorArg(grad, "grad_", 1);
  checkScalarTypes("embedding_bag", grad_arg, {kFloat, kDouble});

  if (mode == MODE_MAX) {
    return _embedding_bag_dense_backward_cpu_max(
        grad_, bag_size_, max_indices_, num_weights);
  }
  AT_ASSERT(mode == MODE_MEAN || mode == MODE_SUM);

  auto index_grad_weight =
      at::zeros({num_weights, grad.size(1)}, grad.options());

  AT_DISPATCH_FLOATING_TYPES(grad.scalar_type(), "embedding_bag_backward", [&] {
      _embedding_bag_dense_backward_cpu_sum_mean<scalar_t>(
          grad, indices_, offsets_, offset2bag__, num_weights,
          scale_grad_by_freq, mode, per_sample_weights_, index_grad_weight);
  });
  return index_grad_weight;
}

template<typename scalar_t>
Tensor _embedding_bag_per_sample_weights_backward_cpu_template(
    const Tensor& grad,
    const Tensor& weight,  // NB: embedding table, not per_sample_weights
    const Tensor& indices,
    const Tensor& offsets,
    const Tensor& offset2bag,
    int64_t mode) {
  TORCH_CHECK(
      mode == MODE_SUM,
      "embedding_bag_backward: per_sample_weights only supported for mode='sum'");

  AT_ASSERT(grad.dim() == 2);
  auto embedding_features = grad.size(1);

  AT_ASSERT(indices.dim() == 1);
  auto num_samples = indices.size(0);

  AT_ASSERT(weight.dim() == 2);
  AT_ASSERT(weight.size(1) == embedding_features);

  auto output = at::zeros({num_samples}, grad.options());

  auto indices_arg = TensorArg(indices, "indices", 1);
  checkScalarTypes("embedding_bag", indices_arg, {kLong, kInt});
  checkContiguous("embedding_bag", indices_arg);

  Tensor offset2bag_;
  if (indices.numel() != 0 && offset2bag.numel() == 0) {
    offset2bag_ = at::zeros(
       {indices.sizes()[0] + 1}, offset2bag.options()); // offset2bag = [0 0 0 0 0]

    make_offset2bag(offsets, offset2bag_);

    offset2bag_.resize_({indices.sizes()[0]});
  } else {
    auto offset2bag_arg = TensorArg(offset2bag, "offset2bag", 1);
    checkScalarTypes("embedding_bag", offset2bag_arg, {kLong, kInt});
    checkContiguous("embedding_bag", offset2bag_arg);
    offset2bag_ = offset2bag;
  }

  auto* grad_data = grad.data_ptr<scalar_t>();
  auto grad_stride0 = grad.stride(0);
  auto grad_stride1 = grad.stride(1);

  auto* weight_data = weight.data_ptr<scalar_t>();
  auto weight_stride0 = weight.stride(0);
  auto weight_stride1 = weight.stride(1);

  // explicitly capture all required variables to work around windows build
  // TODO: fix this when windows can correctly capture variables in nested lambda
  AT_DISPATCH_INDEX_TYPES(indices.scalar_type(), "_embedding_bag_per_sample_weights_backward_cpu_template",
    [&indices, &output, &offset2bag_, &num_samples, &embedding_features,
      &grad_data, &grad_stride0, &grad_stride1, &weight_data, &weight_stride0, &weight_stride1] () {
    auto* indices_data = indices.data_ptr<index_t>();

    // The following are contiguous
    auto* output_data = output.data_ptr<scalar_t>();
    auto* offset2bag_data = offset2bag_.data_ptr<index_t>();

    // XXX: 64 was arbitrarily chosen. There is probably a sweet spot for this number.
    parallel_for(0, num_samples, 64,
      [&embedding_features, &grad_data, &grad_stride0, &grad_stride1, &weight_data, &weight_stride0,
        &weight_stride1, &offset2bag_data, &indices_data, &output_data](index_t begin, index_t end) {
      for (index_t sample_idx = begin; sample_idx < end; sample_idx++) {
        auto bag_idx = offset2bag_data[sample_idx];
        auto embedding_idx = indices_data[sample_idx];

        output_data[sample_idx] = dot_impl<scalar_t>(
            embedding_features,
            grad_data + grad_stride0 * bag_idx, grad_stride1,
            weight_data + weight_stride0 * embedding_idx, weight_stride1);
      }
    });
  });
  return output;
}

Tensor _embedding_bag_per_sample_weights_backward_cpu(
    const Tensor& grad,
    const Tensor& weight,  // NB: embedding table, not per_sample_weights
    const Tensor& indices,
    const Tensor& offsets,
    const Tensor& offset2bag,
    int64_t mode) {
  return AT_DISPATCH_FLOATING_TYPES(
    grad.scalar_type(), "_embedding_bag_per_sample_weights_backward_cpu", [&]() {
      return _embedding_bag_per_sample_weights_backward_cpu_template<scalar_t>(
          grad, weight, indices, offsets, offset2bag, mode);
    }
  );
}

Tensor _embedding_bag_sparse_backward(
    const Tensor &grad_, const Tensor &indices, const Tensor &offsets,
    const Tensor &offset2bag, const Tensor &bag_size_, int64_t num_weights,
    bool scale_grad_by_freq, int64_t mode, const Tensor& per_sample_weights) {
  // indices, offsets and offset2bag are assumed having correct dtypes and
  // contiguous here due to the checks in _embedding_bag_backward above.
  // Also see NOTE [ embedding_bag Native Functions ] in native_functions.yaml
  // for more details.

  Tensor grad = grad_;
  Tensor index_grad = grad_.index_select(0, offset2bag);
  index_grad = apply_bag_size_backward(offsets, indices, mode, index_grad,
                                       offset2bag, bag_size_);
  if (per_sample_weights.defined()) {
    AT_ASSERT(mode == MODE_SUM);
    index_grad.mul_(per_sample_weights.unsqueeze(1));
  }
  return native::embedding_backward(index_grad, indices, num_weights, -1,
                                    scale_grad_by_freq, true);
}
}
} // namespace at::native
