#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>

// TODO: Remove the condition on AT_ROCM_ENABLED entirely,
// don't build this file as part of CPU build.
#include <ATen/cuda/CUDAConfig.h>

#if !AT_ROCM_ENABLED()

namespace at { namespace native {

// See Note [ATen preprocessor philosophy]

std::tuple<Tensor, Tensor, Tensor> miopen_batch_norm(
    const Tensor& input, const Tensor& weight,
    const Tensor& bias, const Tensor& running_mean, const Tensor& running_var,
    bool training, double exponential_average_factor, double epsilon) {
  AT_ERROR("miopen_batch_norm: ATen not compiled with MIOpen support");
}

std::tuple<Tensor, Tensor, Tensor> miopen_batch_norm_backward(
    const Tensor& input, const Tensor& grad_output, const Tensor& weight,
    const Tensor& running_mean, const Tensor& running_var,
    const Tensor& save_mean, const Tensor& save_var,
    double epsilon) {
  AT_ERROR("miopen_batch_norm_backward: ATen not compiled with MIOpen support");
}

}}  // namespace at::native

#else // AT_ROCM_ENABLED

#include <ATen/miopen/Descriptors.h>
#include <ATen/miopen/Types.h>
#include <ATen/miopen/Utils.h>

#include <ATen/TensorUtils.h>

namespace at { namespace native {

namespace {

Tensor expandScale(const Tensor& t, int64_t dim) {
  std::vector<int64_t> size{ 1, t.numel() };
  while (static_cast<int64_t>(size.size()) < dim) {
    size.emplace_back(1);
  }
  return t.view(size);
}

}  // namespace

std::tuple<Tensor, Tensor, Tensor> miopen_batch_norm(
    const Tensor& input_t, const Tensor& weight_t,
    const Tensor& bias_t, const Tensor& running_mean_t, const Tensor& running_var_t,
    bool training, double exponential_average_factor, double epsilon)
{
  TensorArg input{ input_t, "input", 1 },
            weight{ weight_t, "weight", 2 },
            bias{ bias_t, "bias", 3 },
            running_mean{ running_mean_t, "running_mean", 4 },
            running_var{ running_var_t, "running_var", 5 };
  CheckedFrom c = "miopen_batch_norm";

  checkAllDefined(c, {input, weight, bias});
  if (!training) {
    checkAllDefined(c, {running_mean, running_var});
  }
  checkAllSameGPU(c, {input, weight, bias, running_mean, running_var});
  if (input->scalar_type() != ScalarType::Half) {
    checkAllSameType(c, {input, weight});
  }
  checkAllSameType(c, {weight, bias, running_mean, running_var});
  checkAllContiguous(c, {input, weight, bias, running_mean, running_var});
  checkDimRange(c, input, 2, 6 /* exclusive */);
  auto num_features = input->size(1);
  for (auto t : {weight, bias, running_mean, running_var}) {
    if (t->defined()) {
      checkNumel(c, t, num_features);
    }
  }

  miopenBatchNormMode_t mode;
  if (input->dim() == 2) {
    mode = miopenBNPerActivation;
  } else {
    mode = miopenBNSpatial;
  }

  auto output_t = at::empty(input->sizes(), input->options());
  TensorArg output{ output_t, "output", 0 };

  auto handle = getMiopenHandle();
  auto dataType = getMiopenDataType(*input);
  TensorDescriptor idesc{ *input, 4 };  // input descriptor
  TensorDescriptor wdesc{ expandScale(*weight, input->dim()), 4 };  // descriptor for weight, bias, running_mean, etc.

  Constant one(dataType, 1);
  Constant zero(dataType, 0);
  Tensor save_mean, save_var;

  if (training) {
    int64_t num_features = input_t.size(1);
    save_mean = at::empty({ num_features }, weight_t.options());
    save_var = at::empty({ num_features }, weight_t.options());
    MIOPEN_CHECK(miopenBatchNormalizationForwardTraining(
      handle, mode, &one, &zero,
      idesc.desc(), input->data_ptr(),
      idesc.desc(), output->data_ptr(),
      wdesc.desc(),
      weight->data_ptr(),
      bias->data_ptr(),
      exponential_average_factor,
      at::maybe_data_ptr(running_mean),
      at::maybe_data_ptr(running_var),
      epsilon,
      save_mean.data_ptr(),
      save_var.data_ptr()));
  } else {
    MIOPEN_CHECK(miopenBatchNormalizationForwardInference(
      handle, mode, &one, &zero,
      idesc.desc(), input->data_ptr(),
      idesc.desc(), output->data_ptr(),
      wdesc.desc(),
      weight->data_ptr(),
      bias->data_ptr(),
      running_mean->data_ptr(),
      running_var->data_ptr(),
      epsilon));
  }

  // save_mean and save_var can be undefined
  // If this causes problems, we can initialize them to empty tensors
  // of the correct type
  return std::tuple<Tensor, Tensor, Tensor>{output_t, save_mean, save_var};
}

std::tuple<Tensor, Tensor, Tensor> miopen_batch_norm_backward(
    const Tensor& input_t, const Tensor& grad_output_t, const Tensor& weight_t,
    // Unused: but we require them to be passed so that double backwards
    // has access
    const Tensor& running_mean, const Tensor& running_var,
    const Tensor& save_mean_t, const Tensor& save_var_t,
    double epsilon)
{
  TensorArg input{ input_t, "input", 1 },
            grad_output{ grad_output_t, "grad_output", 2 },
            weight{ weight_t, "weight", 3 },
            save_mean{ save_mean_t, "save_mean", 4 },
            save_var{ save_var_t, "save_var", 5 };
  CheckedFrom c = "miopen_batch_norm_backward";

  checkAllDefined(c, {input, grad_output, weight, save_mean, save_var});
  checkAllSameGPU(c, {input, grad_output, weight, save_mean, save_var});
  if (input->scalar_type() == ScalarType::Half) {
    checkScalarType(c, weight, ScalarType::Float);
  } else {
    checkAllSameType(c, {input, weight});
  }
  checkAllSameType(c, {input, grad_output});
  checkAllSameType(c, {weight, save_mean, save_var});
  checkAllContiguous(c, {input, grad_output, save_mean, save_var});
  checkDimRange(c, input, 2, 6 /* exclusive */);
  checkSameSize(c, input, grad_output);
  auto num_features = input->size(1);
  for (auto t : {weight, save_mean, save_var}) {
    checkNumel(c, t, num_features);
  }

  miopenBatchNormMode_t mode;
  if (input->dim() == 2) {
    mode = miopenBNPerActivation;
  } else {
    mode = miopenBNSpatial;
  }

  auto grad_input_t  = at::empty(input->sizes(), input->options());
  auto grad_weight_t = at::empty(weight->sizes(), weight->options());
  auto grad_bias_t   = at::empty(weight->sizes(), weight->options());

  auto handle = getMiopenHandle();
  auto dataType = getMiopenDataType(*input);

  TensorDescriptor idesc{ *input, 4 };  // input, output, grad_output descriptor
  TensorDescriptor wdesc{ expandScale(*weight, input->dim()), 4 };  // descriptor for weight, bias, save_mean, etc.

  Constant one(dataType, 1);
  Constant zero(dataType, 0);

  MIOPEN_CHECK(miopenBatchNormalizationBackward(
    handle, mode, &one, &zero, &one, &zero,
    idesc.desc(), input->data_ptr(),
    idesc.desc(), grad_output->data_ptr(),
    idesc.desc(), grad_input_t.data_ptr(),
    wdesc.desc(), weight->data_ptr(),
    grad_weight_t.data_ptr(),
    grad_bias_t.data_ptr(),
    epsilon,
    save_mean->data_ptr(),
    save_var->data_ptr()));

  return std::tuple<Tensor,Tensor,Tensor>{grad_input_t, grad_weight_t, grad_bias_t};
}

}}  // namespace native

#endif
