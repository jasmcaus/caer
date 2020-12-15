#include <torch/nn/modules/embedding.h>

#include <torch/types.h>
#include <torch/utils.h>
#include <torch/nn/init.h>

#include <cstddef>
#include <ostream>
#include <utility>
#include <vector>

namespace F = torch::nn::functional;

namespace torch {
namespace nn {
EmbeddingImpl::EmbeddingImpl(const EmbeddingOptions& options_) : options(options_) { // NOLINT(modernize-pass-by-value)
  reset();
}

void EmbeddingImpl::reset() {
  if (options.padding_idx() != c10::nullopt) {
    if (*options.padding_idx() > 0) {
      TORCH_CHECK(*options.padding_idx() < options.num_embeddings(), "Padding_idx must be within num_embeddings");
    }
    else if (*options.padding_idx() < 0) {
      TORCH_CHECK(*options.padding_idx() >= -options.num_embeddings(), "Padding_idx must be within num_embedding");
      options.padding_idx(options.num_embeddings() + *options.padding_idx());
    }
  }

  if (!options._weight().defined()) {
    weight = register_parameter(
        "weight", torch::empty({options.num_embeddings(), options.embedding_dim()}));
    reset_parameters();
  } else {
    TORCH_CHECK(options._weight().sizes() == torch::IntArrayRef({options.num_embeddings(), options.embedding_dim()}), "Shape of _weight does not match num_embeddings and embedding_dim");
    weight = register_parameter("weight", options._weight());
  }
}

void EmbeddingImpl::reset_parameters() {
  torch::nn::init::normal_(weight);
  if (options.padding_idx() != c10::nullopt) {
    torch::NoGradGuard no_grad;
    weight[*options.padding_idx()].fill_(0);
  }
}

void EmbeddingImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::Embedding(num_embeddings=" << options.num_embeddings()
         << ", embedding_dim=" << options.embedding_dim();
  if (options.padding_idx() != c10::nullopt) {
    stream << ", padding_idx=" << *options.padding_idx();
  }
  if (options.max_norm() != c10::nullopt) {
    stream << ", max_norm=" << *options.max_norm();
  }
  if (options.norm_type() != 2) {
    stream << ", norm_type=" << options.norm_type();
  }
  if (options.scale_grad_by_freq()) {
    stream << ", scale_grad_by_freq=" << std::boolalpha << options.scale_grad_by_freq();
  }
  if (options.sparse()) {
    stream << ", sparse=" << std::boolalpha << options.sparse();
  }
  stream << ")";
}

torch::Tensor EmbeddingImpl::forward(const Tensor& input) {
  return F::detail::embedding(
    input,
    weight,
    options.padding_idx(),
    options.max_norm(),
    options.norm_type(),
    options.scale_grad_by_freq(),
    options.sparse());
}

EmbeddingBagImpl::EmbeddingBagImpl(const EmbeddingBagOptions& options_) : options(options_) { // NOLINT(modernize-pass-by-value)
  reset();
}

void EmbeddingBagImpl::reset() {
  if (!options._weight().defined()) {
    weight = register_parameter(
        "weight", torch::empty({options.num_embeddings(), options.embedding_dim()}));
    reset_parameters();
  } else {
    TORCH_CHECK(
      options._weight().sizes() == torch::IntArrayRef({options.num_embeddings(), options.embedding_dim()}),
      "Shape of weight does not match num_embeddings and embedding_dim");
    weight = register_parameter("weight", options._weight());
  }
}

void EmbeddingBagImpl::reset_parameters() {
  torch::nn::init::normal_(weight);
}

torch::Tensor EmbeddingBagImpl::forward(const Tensor& input, const Tensor& offsets, const Tensor& per_sample_weights) {
  return F::detail::embedding_bag(
    input,
    weight,
    offsets,
    options.max_norm(),
    options.norm_type(),
    options.scale_grad_by_freq(),
    options.mode(),
    options.sparse(),
    per_sample_weights,
    options.include_last_offset());
}

void EmbeddingBagImpl::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::EmbeddingBag(num_embeddings=" << options.num_embeddings()
        << ", embedding_dim=" << options.embedding_dim();
  if (options.max_norm() != c10::nullopt) {
    stream << ", max_norm=" << *options.max_norm();
  }
  if (options.norm_type() != 2) {
    stream << ", norm_type=" << options.norm_type();
  }
  if (options.scale_grad_by_freq()) {
    stream << ", scale_grad_by_freq=" << std::boolalpha << options.scale_grad_by_freq();
  }
  if (options.sparse()) {
    stream << ", sparse=" << std::boolalpha << options.sparse();
  }
  if (!c10::get_if<enumtype::kMean>(&options.mode())) {
      stream << ", mode=" << torch::enumtype::get_enum_name(options.mode());
  }
  if (options.include_last_offset()) {
    stream << ", include_last_offset=" << std::boolalpha << options.include_last_offset();
  }
  stream << ")";
}
} // namespace nn
} // namespace torch
