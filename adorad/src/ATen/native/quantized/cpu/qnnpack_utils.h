#pragma once

#ifdef USE_PYTORCH_QNNPACK
#include <ATen/ATen.h>
#include <pytorch_qnnpack.h>
#include <qnnpack_func.h>

#include <ATen/native/quantized/cpu/conv_packed_params.h>
#include <ATen/native/quantized/cpu/packed_params.h>
#include <ATen/native/utils/Factory.h>

#include <utility>

struct QnnpackOperatorDeleter {
  void operator()(pytorch_qnnp_operator_t op) {
    pytorch_qnnp_delete_operator(op);
  }
};

// PackedWeight struct for QNNPACK stores the original Weight and Bias as
// QNNPACK currently does not support an unpack function.
// For PyTorch Mobile, once the model is scripted and serialized we don't need
// to call unpack, so we can save some memory by checking for this case and free
// the original weights after packing.
// Input scale is set to null in pre-pack step. QNNPACK needs bias quantized
// with input scale which is available at runtime in pytorch. During runtime if
// input scale value changes then we requantize bias with the updated scale. For
// inference we expect the graph to be static so the input scale should not
// change across consecutive inference calls.
struct PackedLinearWeightsQnnp : public LinearPackedParamsBase {
  PackedLinearWeightsQnnp(
      std::unique_ptr<qnnpack::PackBMatrix> w,
      at::Tensor orig_weight,
      at::Tensor bias,
      c10::optional<double> input_scale,
      at::Tensor w_scales,
      std::vector<uint8_t>&& w_zps)
      : w(std::move(w)),
        orig_weight(std::move(orig_weight)),
        bias_(at::native::mobile::allocate_padded_contiguous_if_needed(
            bias, bias.suggest_memory_format())),
        input_scale(std::move(input_scale)),
        w_scales(w_scales),
        w_zero_points(std::move(w_zps)) {}

  std::unique_ptr<qnnpack::PackBMatrix> w;
  at::Tensor orig_weight;
  at::Tensor bias_;
  c10::optional<double> input_scale;
  at::Tensor w_scales;
  std::vector<uint8_t> w_zero_points;
  std::vector<float> requantization_scales;

  at::Tensor apply(
      at::Tensor input,
      double output_scale,
      int64_t output_zero_point) override;
  at::Tensor apply_relu(
      at::Tensor input,
      double output_scale,
      int64_t output_zero_point) override;

  at::Tensor apply_dynamic(at::Tensor input, bool reduce_range=false) override;
  at::Tensor apply_dynamic_relu(at::Tensor input, bool reduce_range=false) override;

  std::tuple<at::Tensor, c10::optional<at::Tensor>> unpack() override;

  c10::optional<at::Tensor> bias() override {
    return bias_;
  }

  static c10::intrusive_ptr<LinearPackedParamsBase> prepack(
      at::Tensor weight,
      c10::optional<at::Tensor> bias);

 private:
  template <bool ReluFused>
  at::Tensor apply_impl(
      at::Tensor input,
      double output_scale,
      int64_t output_zero_point);

  template <bool ReluFused>
  at::Tensor apply_dynamic_impl(at::Tensor input);
};

template <int kSpatialDim = 2>
struct PackedConvWeightsQnnp : public ConvPackedParamsBase<kSpatialDim> {
  PackedConvWeightsQnnp(
      std::unique_ptr<qnnpack::PrePackConvWeights> w,
      at::Tensor orig_weight,
      at::Tensor bias,
      torch::List<int64_t> stride,
      torch::List<int64_t> padding,
      torch::List<int64_t> output_padding,
      torch::List<int64_t> dilation,
      int64_t groups,
      bool transpose,
      c10::optional<double> input_scale,
      std::vector<int64_t> kernel,
      at::Tensor w_scale,
      std::vector<uint8_t>&& w_zps,
      bool is_per_channel)
      : w(std::move(w)),
        orig_weight(std::move(orig_weight)),
        bias(std::move(bias)),
        stride_(std::move(stride)),
        padding_(std::move(padding)),
        output_padding_(std::move(output_padding)),
        dilation_(std::move(dilation)),
        groups_(groups),
        transpose_(transpose),
        input_scale(input_scale),
        kernel_(std::move(kernel)),
        w_scales(w_scale),
        w_zero_points(std::move(w_zps)),
        conv_p(
            {(uint32_t)kernel_[1], (uint32_t)kernel_[0]},
            {(uint32_t)stride_[1], (uint32_t)stride_[0]},
            {(uint32_t)dilation_[1], (uint32_t)dilation_[0]},
            {(uint32_t)padding_[0], (uint32_t)padding_[1],
             (uint32_t)padding_[0], (uint32_t)padding_[1]},
            {(uint32_t)output_padding_[1], (uint32_t)output_padding_[0]},
            groups_,
            transpose ? this->orig_weight.size(0)
                      : this->orig_weight.size(1) * groups_,
            transpose ? this->orig_weight.size(1) * groups_
                      : this->orig_weight.size(0),
            transpose_,
            is_per_channel) {

          if (conv_p.per_channel && conv_p.ukernel_type == pytorch_qnnp_ukernel_type_xzp_gemm) {
            TORCH_INTERNAL_ASSERT(
              "Per channel quantized weights are not supported for XZP kernels");
          }

          pytorch_qnnp_operator_t convolution{nullptr};
          // Initially all the params are set to zero.
          convolution =
              static_cast<pytorch_qnnp_operator_t>(calloc(1, sizeof(struct pytorch_qnnp_operator)));
          if (convolution == nullptr) {
            TORCH_INTERNAL_ASSERT(
                "failed to allocate %zu bytes for pytorch_qnnp_operator structure",
                sizeof(struct pytorch_qnnp_operator));
          }

          convolution_op =
            std::unique_ptr<pytorch_qnnp_operator, QnnpackOperatorDeleter>(convolution);

          convolution->ukernel_type = conv_p.ukernel_type;
          convolution->groups = groups;
          convolution->group_input_channels = conv_p.group_input_channels;
          convolution->kernel_height = conv_p.kernel_dims[1];
          convolution->kernel_width = conv_p.kernel_dims[0];
          convolution->stride_height = conv_p.stride_dims[1];
          convolution->stride_width = conv_p.stride_dims[0];
          convolution->dilation_height = conv_p.dilation[1];
          convolution->dilation_width = conv_p.dilation[0];
          convolution->input_padding_top = conv_p.padding[0];
          convolution->input_padding_left = conv_p.padding[1];
          convolution->input_padding_bottom = conv_p.padding[2];
          convolution->input_padding_right = conv_p.padding[3];

          // const size_t group_input_channels = conv_p.group_input_channels;
          const uint32_t kr = pytorch_qnnp_params.q8conv.kr;
          const size_t k_stride = (conv_p.group_input_channels + (kr - 1)) & -kr;

          size_t zero_size = sizeof(uint8_t) * k_stride;
          size_t zero_offset = 0;

          if (transpose_) {
            convolution->adjustment_width = conv_p.adjustment_dims[0];
            convolution->adjustment_height = conv_p.adjustment_dims[1];

            // const uint32_t kr = pytorch_qnnp_params.q8conv.kr;
            // const size_t k_stride = (conv_p.group_input_channels + (kr - 1)) & -kr;

            if (conv_p.group_input_channels < 8) {
              zero_size += 8;
              zero_offset = 8;
            }
          } else {
            const bool any_padding = (conv_p.padding[0]| conv_p.padding[1]
                |conv_p.padding[2] | conv_p.padding[3]) != 0;

            zero_buffer_size = 0;
            if (any_padding) {
              zero_size = 0;
              zero_offset = 0;
              if (conv_p.ukernel_type == pytorch_qnnp_ukernel_type_dwconv) {
                const uint32_t cr = pytorch_qnnp_params.q8dw9.cr;
                const size_t group_stride = (groups + (cr - 1)) & -cr;
                if (groups >= 8) {
                  zero_size = sizeof(uint8_t) * group_stride;
                  zero_offset = 0;
                } else {
                  zero_size = sizeof(uint8_t) * group_stride + 8;
                  zero_offset = sizeof(uint8_t) * 8;
                }
              } else if (conv_p.ukernel_type == pytorch_qnnp_ukernel_type_conv ||
                  conv_p.ukernel_type == pytorch_qnnp_ukernel_type_gemm) {
                if (conv_p.group_input_channels >= 8) {
                  zero_size = sizeof(uint8_t) * k_stride;
                  zero_offset = 0;
                } else {
                  zero_size = sizeof(uint8_t) * k_stride + 8;
                  zero_offset = 8;
                }
              }
            }
          }

          void* zero_buffer = malloc(zero_size);
          if (zero_buffer == NULL) {
            pytorch_qnnp_delete_operator(convolution);
            pytorch_qnnp_log_error(
                "failed to allocate %zu bytes for zero padding", zero_size);
          }
          // Need to set to input zero point
          // memset(zero_buffer, input_zero_point, zero_size);
          zero_buffer_size = zero_size;
          convolution->zero_buffer = zero_buffer;
          convolution->zero_pointer =
            (void*)((uintptr_t)zero_buffer + zero_offset);
        }

  std::unique_ptr<pytorch_qnnp_operator, QnnpackOperatorDeleter> convolution_op;
  std::unique_ptr<qnnpack::PrePackConvWeights> w;
  at::Tensor orig_weight;
  at::Tensor bias;
  torch::List<int64_t> stride_;
  torch::List<int64_t> padding_;
  torch::List<int64_t> output_padding_;
  torch::List<int64_t> dilation_;
  int64_t groups_;
  bool transpose_;
  c10::optional<double> input_scale;
  std::vector<int64_t> kernel_;
  at::Tensor w_scales;
  std::vector<uint8_t> w_zero_points;
  std::vector<float> requantization_scales;
  qnnpack::conv_param_t conv_p;
  size_t zero_buffer_size;

  at::Tensor apply(
      const at::Tensor& input,
      double output_scale,
      int64_t output_zero_point) override;

  at::Tensor apply_relu(
      const at::Tensor& input,
      double output_scale,
      int64_t output_zero_point) override;

  std::tuple<at::Tensor, c10::optional<at::Tensor>> unpack() override;

  static c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>> prepack(
      at::Tensor weight,
      c10::optional<at::Tensor> bias,
      torch::List<int64_t> stride,
      torch::List<int64_t> padding,
      torch::List<int64_t> output_padding,
      torch::List<int64_t> dilation,
      int64_t groups,
      bool transpose);

  torch::List<int64_t> stride() const override {
    return stride_;
  }

  torch::List<int64_t> padding() const override {
    return padding_;
  }

  torch::List<int64_t> output_padding() const override {
    return output_padding_;
  }

  torch::List<int64_t> dilation() const override {
    return dilation_;
  }

  int64_t groups() const override {
    return groups_;
  }

  bool transpose() const override {
    return transpose_;
  }

 private:
  template <bool ReluFused>
  at::Tensor apply_impl(
      const at::Tensor& input,
      double output_scale,
      int64_t output_zero_point);
};

enum class Activation : uint8_t { NONE = 0, RELU = 1 };

#if defined(__ANDROID__) && !defined(__NDK_MAJOR__)
template <class T>
inline float Round(const float x) {
  return ::nearbyintf(x);
}
inline double Round(const double x) {
  return ::nearbyint(x);
}
#else
template <class T>
inline T Round(const T x) {
  return std::nearbyint(x);
}
#endif

inline uint8_t QuantizeUint8(float scale, int32_t zero_point, float value) {
  const int32_t qmin = std::numeric_limits<uint8_t>::min();
  const int32_t qmax = std::numeric_limits<uint8_t>::max();
  auto r = zero_point + static_cast<int32_t>(Round(value / scale));
  r = std::max(r, qmin);
  r = std::min(r, qmax);
  return static_cast<uint8_t>(r);
}

inline std::pair<uint8_t, uint8_t> activationLimits(
    float scale,
    int32_t zero_point,
    Activation Ac) {
  switch (Ac) {
    case Activation::NONE:
      return {std::numeric_limits<uint8_t>::min(),
              std::numeric_limits<uint8_t>::max()};
    case Activation::RELU:
      return {QuantizeUint8(scale, zero_point, 0.0),
              std::numeric_limits<uint8_t>::max()};
    default:
#ifdef _MSC_VER
      __assume(0);
#else
      __builtin_unreachable();
#endif
  }
}

namespace at {
namespace native {
namespace qnnp_avgpool_helper {
Tensor qnnpack_avg_pool2d(
    Tensor input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override);
} // qnnp_avgpool_helper
} // namespace native
} // namespace at

namespace {
std::vector<float> generate_requantization_scales(
    const at::Tensor& weight_scales,
    const float input_scale,
    const float output_scale,
    std::vector<float>& requant_scales) {
  // Since weight scale is allocated with padding
  // weight_scales.numel() gives us padded num elements.
  auto num_output_channels_padded = weight_scales.numel();
  float* weight_scales_data = weight_scales.data_ptr<float>();
  if (requant_scales.size() < num_output_channels_padded) {
    requant_scales.resize(num_output_channels_padded);
  }
  for (int i = 0; i < num_output_channels_padded; ++i) {
    auto inverse_output_scale = 1.f /output_scale;
    requant_scales[i] = (weight_scales_data[i] * input_scale) * inverse_output_scale;
    TORCH_CHECK(
        (requant_scales[i] > 0.0f && std::isnormal(requant_scales[i])),
        "failed to create op with requantization scale: ",
        requant_scales[i],
        ": requantization scale must be finite and positive");
  }
  return requant_scales;
}

std::pair<std::vector<uint8_t>, at::Tensor> make_zero_points_and_scales_tensor(
    const at::Tensor& weight_contig,
    bool transpose = false,
    uint32_t groups = 1
  ) {
  const int out_ch_idx = transpose ? 1 : 0;
  auto num_output_channels = weight_contig.size(out_ch_idx) * (transpose ? groups : 1);
  // Add 8 to account for bufferring needed by QNNPACK.
  auto num_output_channels_padded = num_output_channels + 8;
  const auto qtype = weight_contig.qscheme();
  std::vector<uint8_t> weight_zp(num_output_channels_padded, 0);
  // Adjust weight zero point, similar to weight data.
  if (qtype == at::kPerTensorAffine) {
    for (int i = 0; i < num_output_channels; ++i) {
      weight_zp[i] = (uint8_t)(weight_contig.q_zero_point() + 128);
    }
  } else if (qtype == at::kPerChannelAffine) {
    for (int i = 0; i < num_output_channels; ++i) {
      weight_zp[i] =
          (uint8_t)(
              weight_contig.q_per_channel_zero_points()[i].item<int32_t>() +
              128);
    }
  } else {
    TORCH_INTERNAL_ASSERT("Unsupported quantization scheme.");
  }
  at:: Tensor weight_scales =
    at::empty(
        {num_output_channels_padded},
        at::device(at::kCPU).dtype(at::kFloat));
  float* weight_scales_data = weight_scales.data_ptr<float>();
  if (qtype == at::kPerTensorAffine) {
    for (int i = 0; i < num_output_channels; ++i) {
      weight_scales_data[i] = weight_contig.q_scale();
    }
  } else if (qtype == at::kPerChannelAffine) {
    for (int i = 0; i < num_output_channels; ++i) {
      weight_scales_data[i] =
        weight_contig.q_per_channel_scales()[i].item<float>();
    }
  } else {
    TORCH_INTERNAL_ASSERT("Unsupported quantization scheme.");
  }
  for (int i = num_output_channels; i <  num_output_channels_padded; ++i) {
    weight_scales_data[i] = 1.f;
  }
  return {weight_zp, weight_scales};
}
} // namespace

#endif
