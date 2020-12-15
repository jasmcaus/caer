#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>
#include <ATen/cuda/CUDAConfig.h>

#if !AT_CUDNN_ENABLED()

namespace at { namespace native {

// See Note [ATen preprocessor philosophy]

std::tuple<Tensor, Tensor, Tensor, Tensor> cudnn_batch_norm(
    const Tensor& input, const Tensor& weight,
    const Tensor& bias, const Tensor& running_mean, const Tensor& running_var,
    bool training, double exponential_average_factor, double epsilon) {
  AT_ERROR("cudnn_batch_norm: ATen not compiled with cuDNN support");
}

std::tuple<Tensor, Tensor, Tensor> cudnn_batch_norm_backward(
    const Tensor& input, const Tensor& grad_output, const Tensor& weight,
    const Tensor& running_mean, const Tensor& running_var,
    const Tensor& save_mean, const Tensor& save_var,
    double epsilon, const Tensor& reservedSpace) {
  AT_ERROR("cudnn_batch_norm_backward: ATen not compiled with cuDNN support");
}

}}  // namespace at::native

#else // AT_CUDNN_ENABLED

#include <ATen/cudnn/Descriptors.h>
#include <ATen/cudnn/Types.h>
#include <ATen/cudnn/Utils.h>
#include <ATen/cuda/Exceptions.h>

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

std::tuple<Tensor, Tensor, Tensor, Tensor> cudnn_batch_norm(
    const Tensor& input_t, const Tensor& weight_t,
    const Tensor& bias_t, const Tensor& running_mean_t, const Tensor& running_var_t,
    bool training, double exponential_average_factor, double epsilon)
{
  TensorArg input{ input_t, "input", 1 },
            weight{ weight_t, "weight", 2 },
            bias{ bias_t, "bias", 3 },
            running_mean{ running_mean_t, "running_mean", 4 },
            running_var{ running_var_t, "running_var", 5 };
  CheckedFrom c = "cudnn_batch_norm";

  checkAllDefined(c, {input, weight, bias});
  if (!training) {
    checkAllDefined(c, {running_mean, running_var});
  }
  checkAllSameGPU(c, {input, weight, bias, running_mean, running_var});
  if (input->scalar_type() == ScalarType::Half) {
    checkScalarType(c, weight, ScalarType::Float);
  } else {
    checkAllSameType(c, {input, weight});
  }
  checkAllSameType(c, {weight, bias, running_mean, running_var});
  // TODO: is weight required to be contiguous?
  checkAllContiguous(c, {weight, bias, running_mean, running_var});
  // TODO: TensorArg check should start handle memory format
  TORCH_CHECK(input->is_contiguous(input->suggest_memory_format()));

  checkDimRange(c, input, 2, 6 /* exclusive */);
  auto num_features = input->size(1);
  for (auto t : {weight, bias, running_mean, running_var}) {
    if (t->defined()) {
      checkNumel(c, t, num_features);
    }
  }

  cudnnBatchNormMode_t mode;
  if (input->dim() == 2) {
    mode = CUDNN_BATCHNORM_PER_ACTIVATION;
  } else if (training && input->suggest_memory_format() == at::MemoryFormat::ChannelsLast) {
#if CUDNN_VERSION >= 7400
    mode = CUDNN_BATCHNORM_SPATIAL_PERSISTENT;
#else
    mode = CUDNN_BATCHNORM_SPATIAL;
#endif // CUDNN_VERSION >= 7400
  } else {
    // TODO: The new CUDNN_BATCHNORM_SPATIAL_PERSISTENT mode was
    // introduced in CuDNN 7 for performance optimization, but it results in
    // accuracy losses in convolution models such as ResNeXt-101 and
    // video R(2+1)D. We will fall back to the normal CUDNN_BATCHNORM_SPATIAL
    mode = CUDNN_BATCHNORM_SPATIAL;
  }

  auto output_t = at::empty_like(*input, input->options(), input->suggest_memory_format());

  TensorArg output{ output_t, "output", 0 };

  auto handle = getCudnnHandle();
  auto dataType = getCudnnDataType(*input);
  TensorDescriptor idesc{ *input, 4 };  // input descriptor
  TensorDescriptor wdesc{ expandScale(*weight, input->dim()), 4 };  // descriptor for weight, bias, running_mean, etc.

  Constant one(dataType, 1);
  Constant zero(dataType, 0);
  Tensor save_mean, save_var;

  Tensor reserve;

  if (training) {

    int64_t num_features = input_t.size(1);
    save_mean = at::empty({ num_features }, weight_t.options());
    save_var = at::empty({ num_features }, weight_t.options());
    
#if CUDNN_VERSION >= 7400
    auto op = CUDNN_BATCHNORM_OPS_BN;
    size_t workspace_size;
    AT_CUDNN_CHECK(cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(
        handle,
        mode,
        op,
        idesc.desc(),
        idesc.desc(),
        idesc.desc(),
        wdesc.desc(),
        nullptr,
        &workspace_size));
    Tensor workspace = at::empty(workspace_size, input->options().dtype(kByte));

    // get the reserved size and allocate as tensor
    size_t reserve_size;
    AT_CUDNN_CHECK(cudnnGetBatchNormalizationTrainingExReserveSpaceSize(
        handle,
        mode,
        op,
        nullptr,
        idesc.desc(),
        &reserve_size));
    reserve = at::empty(reserve_size, input->options().dtype(kByte));

    AT_CUDNN_CHECK(cudnnBatchNormalizationForwardTrainingEx(
        handle,
        mode,
        op,
        &one,
        &zero,
        idesc.desc(),
        input->data_ptr(),
        nullptr,  // z descriptor for BN-Add-Relu
        nullptr,  // z for BN-Add-ReLU
        idesc.desc(),
        output->data_ptr(),
        wdesc.desc(),
        weight->data_ptr(),
        bias->data_ptr(),
        exponential_average_factor,
        at::maybe_data_ptr(running_mean),
        at::maybe_data_ptr(running_var),
        epsilon,
        save_mean.data_ptr(),
        save_var.data_ptr(),
        nullptr,
        workspace.data_ptr(),
        workspace_size,
        reserve.data_ptr(),
        reserve_size));
#else
    reserve = at::empty({0}, input->options().dtype(kByte));
    AT_CUDNN_CHECK(cudnnBatchNormalizationForwardTraining(
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
#endif // CUDNN_VERSION >= 7400
  } else {
    reserve = at::empty({0}, input->options().dtype(kByte));
    AT_CUDNN_CHECK(cudnnBatchNormalizationForwardInference(
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
  return std::tuple<Tensor, Tensor, Tensor, Tensor>{output_t, save_mean, save_var, reserve};
}

// NB: CuDNN only implements the backward algorithm for batchnorm
// in training mode (evaluation mode batchnorm has a different algorithm),
// which is why this doesn't accept a 'training' parameter.
std::tuple<Tensor, Tensor, Tensor> cudnn_batch_norm_backward(
    const Tensor& input_t, const Tensor& grad_output_t,
    const Tensor& weight_t,
    // Unused: but we require them to be passed so that double backwards
    // has access
    const Tensor& running_mean, const Tensor& running_var,
    const Tensor& save_mean_t, const Tensor& save_var_t,
    double epsilon, const Tensor& reserveSpace)
{
  // TODO: Is it worth it to have a contiguous call or maybe we should go with
  // whatever format is given here.
  TensorArg input{ input_t, "input", 1 },
            grad_output{ grad_output_t.contiguous(input_t.suggest_memory_format()), "grad_output", 2 },
            weight{ weight_t, "weight", 3 },
            save_mean{ save_mean_t, "save_mean", 4 },
            save_var{ save_var_t, "save_var", 5 },
            reserve{ reserveSpace, "reserve_space", 6 };
  CheckedFrom c = "cudnn_batch_norm_backward";

  checkAllDefined(c, {input, grad_output, weight, save_mean, save_var});
  checkAllSameGPU(c, {input, grad_output, weight, save_mean, save_var});
  if (input->scalar_type() == ScalarType::Half) {
    checkScalarType(c, weight, ScalarType::Float);
  } else {
    checkAllSameType(c, {input, weight});
  }
  checkAllSameType(c, {input, grad_output});
  checkAllSameType(c, {weight, save_mean, save_var});
  // TODO: is weight required to be contiguous?
  checkAllContiguous(c, {save_mean, save_var});
  // TODO: TensorArg check should start handle memory format
  TORCH_CHECK(input->is_contiguous(input->suggest_memory_format()));
  TORCH_CHECK(grad_output->is_contiguous(grad_output->suggest_memory_format()));
  checkDimRange(c, input, 2, 6 /* exclusive */);
  checkSameSize(c, input, grad_output);
  auto num_features = input->size(1);
  for (auto t : {weight, save_mean, save_var}) {
    checkNumel(c, t, num_features);
  }

  cudnnBatchNormMode_t mode;
  if (input->dim() == 2) {
    mode = CUDNN_BATCHNORM_PER_ACTIVATION;
  } else if (input->suggest_memory_format() == at::MemoryFormat::ChannelsLast) {
#if CUDNN_VERSION >= 7400
    mode = CUDNN_BATCHNORM_SPATIAL_PERSISTENT;
#else
    mode = CUDNN_BATCHNORM_SPATIAL;
#endif // CUDNN_VERSION >= 7400
  } else {
    // TODO: The new CUDNN_BATCHNORM_SPATIAL_PERSISTENT mode was
    // introduced in CuDNN 7 for performance optimization, but it results in
    // accuracy losses in convolution models such as ResNeXt-101 and
    // video R(2+1)D. We will fall back to the normal CUDNN_BATCHNORM_SPATIAL
    mode = CUDNN_BATCHNORM_SPATIAL;
  }

  auto grad_input_t  = at::empty(input->sizes(), input->options(), input->suggest_memory_format());
  auto grad_weight_t = at::empty(weight->sizes(), weight->options());
  auto grad_bias_t   = at::empty(weight->sizes(), weight->options());

  auto handle = getCudnnHandle();
  auto dataType = getCudnnDataType(*input);

  TensorDescriptor idesc{ *input, 4 };  // input, grad_output descriptor
  TensorDescriptor odesc{ *grad_output, 4 };  // input, grad_output descriptor
  TensorDescriptor wdesc{ expandScale(*weight, input->dim()), 4 };  // descriptor for weight, save_mean, etc.

  Constant one(dataType, 1);
  Constant zero(dataType, 0);

#if CUDNN_VERSION >= 7400
  auto op = CUDNN_BATCHNORM_OPS_BN;

  size_t workspace_size;
  AT_CUDNN_CHECK(cudnnGetBatchNormalizationBackwardExWorkspaceSize(
      handle,
      mode,
      op,
      idesc.desc(),
      idesc.desc(),
      idesc.desc(),
      nullptr,
      odesc.desc(),
      wdesc.desc(),
      nullptr,
      &workspace_size));
  Tensor workspace = at::empty(workspace_size, input->options().dtype(kByte));

  AT_CUDNN_CHECK(cudnnBatchNormalizationBackwardEx(
    handle, mode, op, &one, &zero, &one, &zero,
    idesc.desc(), input->data_ptr(),
    nullptr, nullptr,
    odesc.desc(), grad_output->data_ptr(),
    nullptr, nullptr,
    idesc.desc(), grad_input_t.data_ptr(),
    wdesc.desc(), weight->data_ptr(), 
    nullptr,
    grad_weight_t.data_ptr(),
    grad_bias_t.data_ptr(),
    epsilon,
    save_mean->data_ptr(),
    save_var->data_ptr(),
    nullptr,
    workspace.data_ptr(),
    workspace_size,
    reserve->data_ptr(),
    reserve->numel()));
#else
  AT_CUDNN_CHECK(cudnnBatchNormalizationBackward(
    handle, mode, &one, &zero, &one, &zero,
    idesc.desc(), input->data_ptr(),
    odesc.desc(), grad_output->data_ptr(),
    idesc.desc(), grad_input_t.data_ptr(),
    wdesc.desc(), weight->data_ptr(),
    grad_weight_t.data_ptr(),
    grad_bias_t.data_ptr(),
    epsilon,
    save_mean->data_ptr(),
    save_var->data_ptr()));
#endif // CUDNN_VERSION >= 7400

  return std::tuple<Tensor,Tensor,Tensor>{grad_input_t, grad_weight_t, grad_bias_t};
}

}}  // namespace native

#endif
