#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THCUNN/generic/RReLU.cu"
#else

#include <THCUNN/common.h>
#include <ATen/CUDAGeneratorImpl.h>

void THNN_(RReLU_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           THCTensor *noise,
           double lower,
           double upper,
           bool train,
           bool inplace,
           c10::optional<at::Generator> generator)
{
  THCUNN_assertSameGPU(state, 3, input, output, noise);
  auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(generator, at::cuda::detail::getDefaultCUDAGenerator());
  if (train)
  {
    auto inputTensor = THTensor_wrap(input).contiguous();
    input = inputTensor.unsafeGetTensorImpl();
    THCTensor_(resizeAs)(state, noise, input);
    scalar_t *input_data = THCTensor_(data)(state, input);
    scalar_t *noise_data = THCTensor_(data)(state, noise);
    ptrdiff_t n = THCTensor_(nElement)(state, input);

    // philox offset calculation for grid-stride loop utilizing curand4
    const uint32_t curand4_engine_calls = 4;
    dim3 grid = NUM_BLOCKS(n);
    uint64_t counter_offset = ((n - 1) / (BLOCK_SIZE * grid.x) + 1) * curand4_engine_calls;
    at::PhiloxCudaState rng_engine_inputs;
    {
      // See Note [Acquire lock when using random generators]
      std::lock_guard<std::mutex> lock(gen->mutex_);
      rng_engine_inputs = gen->philox_cuda_state(counter_offset);
    }
    if (inplace)
    {
      rreluUpdateOutputTrain<<<grid, BLOCK_SIZE, 0, c10::cuda::getCurrentCUDAStream()>>>(
        n, rng_engine_inputs, input_data, noise_data, input_data, lower, upper);
      THCTensor_(set)(state, output, input);
    }
    else
    {
      THCTensor_(resizeAs)(state, output, input);
      scalar_t *output_data = THCTensor_(data)(state, output);
      rreluUpdateOutputTrain<<<grid, BLOCK_SIZE, 0, c10::cuda::getCurrentCUDAStream()>>>(
        n, rng_engine_inputs, input_data, noise_data, output_data, lower, upper);
    }
    THCudaCheck(cudaGetLastError());
  }
  else
  {
    const scalar_t negSlope = ScalarConvert<double, scalar_t>::to((lower + upper) / 2);
    if (inplace)
    {
      THC_pointwiseApply1<scalar_t>(state, input, RReLUUpdateOutputEvalIP_functor<scalar_t>(negSlope));
      THCTensor_(set)(state, output, input);
    }
    else
    {
      THCTensor_(resizeAs)(state, output, input);
      THC_pointwiseApply2<scalar_t, scalar_t>(state, output, input, RReLUUpdateOutputEval_functor<scalar_t>(negSlope));
    }
  }
}

void THNN_(RReLU_updateGradInput)(
           THCState *state,
           THCTensor *input,
           THCTensor *gradOutput,
           THCTensor *gradInput,
           THCTensor *noise,
           double lower,
           double upper,
           bool train,
           bool inplace)
{
  THCUNN_check_nElement(state, input, gradOutput);
  THCUNN_assertSameGPU(state, 4, input, gradOutput, gradInput, noise);

  auto gradOutputTensor = THTensor_wrap(gradOutput).contiguous();
  gradOutput = gradOutputTensor.unsafeGetTensorImpl();

  if (train && upper - lower > 1E-6)    // e.g. if upper == lower, RReLU behaves like LeakyReLU
  {
    // multiply the gradient by the noise tensor
    if (inplace)
    {
      THCTensor_(cmul)(state, gradOutput, gradOutput, noise);
      THCTensor_(set)(state, gradInput, gradOutput);
    }
    else
    {
      THCTensor_(resizeAs)(state, gradInput, input);
      THCTensor_(cmul)(state, gradInput, gradOutput, noise);
    }
  }
  else
  {
    // use constant factor for negative input values
    const scalar_t negSlope = ScalarConvert<double, scalar_t>::to((lower + upper) / 2);
    if (inplace)
    {
      THC_pointwiseApply2<scalar_t, scalar_t>(state, gradOutput, input, RReLUupdateGradInputEvalIP_functor<scalar_t>(negSlope));
      THCTensor_(set)(state, gradInput, gradOutput);
    }
    else
    {
      THCTensor_(resizeAs)(state, gradInput, input);
      THC_pointwiseApply3<scalar_t, scalar_t, scalar_t>(state, gradInput, gradOutput, input, RReLUupdateGradInputEval_functor<scalar_t>(negSlope));
    }
  }
}

#endif
