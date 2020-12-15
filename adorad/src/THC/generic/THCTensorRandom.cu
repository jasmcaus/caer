#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THC/generic/THCTensorRandom.cu"
#else

#include <ATen/cuda/CUDAContext.h>
#include <ATen/CUDAGeneratorImpl.h>
#include <ATen/Utils.h>
#include <c10/cuda/CUDAException.h>
#include <utility>

#if defined(THC_REAL_IS_FLOAT) || defined(THC_REAL_IS_DOUBLE) || defined(THC_REAL_IS_HALF)

void THCTensor_(multinomialAliasSetup)(THCState *state, THCTensor *_probs, THCudaLongTensor *_J, THCTensor *_q){
  THArgCheck(_probs->dim() == 1, 1,
             "expected 1-D probability tensor, got %d-D probability tensor instead",
             _probs->dim());
  THAssert(THCTensor_(isContiguous)(state, _q));
  THAssert(THCudaLongTensor_isContiguous(state, _J));
  THCTensor *probs = THCTensor_(newContiguous)(state, _probs);
  THAssert(THCTensor_(isContiguous)(state, probs));
  int64_t inputsize = THCTensor_(nElement)(state, probs);
  THCudaLongTensor *smaller = THCudaLongTensor_newWithSize1d(state, inputsize);
  THCudaLongTensor *larger = THCudaLongTensor_newWithSize1d(state, inputsize);
  THCudaLongTensor *smaller_short = THCudaLongTensor_newWithSize1d(state, inputsize);
  THCudaLongTensor *larger_short = THCudaLongTensor_newWithSize1d(state, inputsize);

  THCudaLongTensor_resize1d(state, _J, inputsize);
  THCTensor_(resize1d)(state, _q, inputsize);

  scalar_t one = ScalarConvert<int64_t, scalar_t>::to(1);
  int inputBlockDim = THCCeilDiv((int)inputsize + BLOCK_SIZE - 1, BLOCK_SIZE);
  aliasMultinomialFilter
    <<<inputBlockDim, BLOCK_SIZE, 0, c10::cuda::getCurrentCUDAStream() >>>(
                     THCTensor_(data)(state, _q),
                     THCTensor_(data)(state, probs),
                     THCudaLongTensor_data(state, smaller),
                     THCudaLongTensor_data(state, larger),
                     THCudaLongTensor_data(state, _J),
                     THCudaLongTensor_data(state, smaller_short),
                     THCudaLongTensor_data(state, larger_short),
                     one, inputsize
                     );
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  at::Tensor smaller_short_wrapped = THTensor_wrap(smaller_short);
  at::Tensor smaller_wrapped = THTensor_wrap(smaller);
  at::Tensor larger_short_wrapped = THTensor_wrap(larger_short);
  at::Tensor larger_wrapped = THTensor_wrap(larger);
  at::nonzero_out(smaller_short_wrapped, smaller_wrapped);
  at::nonzero_out(larger_short_wrapped, larger_wrapped);
  int h_large_c = THCudaLongTensor_nElement(state, larger_short);
  THCudaLongTensor_resize1d(state, smaller_short, inputsize);
  THCudaLongTensor_resize1d(state, larger_short, inputsize);
  aliasMultinomialSetup
    <<<1, 1, 0, c10::cuda::getCurrentCUDAStream()>>>(
                THCudaLongTensor_data(state, _J),
                THCTensor_(data)(state, _q),
                inputsize,
                THCudaLongTensor_data(state, smaller_short),
                THCudaLongTensor_data(state, larger_short),
                inputsize - h_large_c, h_large_c
                );
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  scalar_t q_max = at::max(THTensor_wrap(_q)).item<scalar_t>();
  condDiv<<<
    inputBlockDim, BLOCK_SIZE, 0, c10::cuda::getCurrentCUDAStream()>>>(
                      THCTensor_(data)(state, _q),
                      THCudaLongTensor_data(state, _J),
                      inputsize, q_max
                      );
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  THCudaLongTensor_free(state, smaller);
  THCudaLongTensor_free(state, larger);
  THCudaLongTensor_free(state, smaller_short);
  THCudaLongTensor_free(state, larger_short);
  THCTensor_free(state, probs);
}

void THCTensor_(multinomialAliasDraw)(THCState *state, THCudaLongTensor *self, THCTensor *_q, THCudaLongTensor *_J, int n_sample, c10::optional<at::Generator> gen_){
  THArgCheck(_q->dim() == 1, 1,
             "expected 1-D probability table, got %d-D probability table instead",
             _q->dim());
  THArgCheck(_J->dim() == 1, 2,
             "expected 1-D alias table, got %d-D alias table instead",
             _J->dim());
  THArgCheck(n_sample > 0, 3, "cannot sample <= 0 samples");
  THAssert(THCTensor_(isContiguous)(state, _q));
  THAssert(THCudaLongTensor_isContiguous(state, _J));
  int64_t K = THCudaLongTensor_nElement(state, _J);
  THCudaLongTensor_resize1d(state, self, n_sample);
  ptrdiff_t size = THCudaLongTensor_nElement(state, self);

  THCTensor *uniform = THCTensor_(newWithSize1d)(state, n_sample);
  THCTensor *bernoulli = THCTensor_(newWithSize1d)(state, n_sample);

  auto out_uniform = THTensor_wrap(uniform);
  auto out_bernoulli = THTensor_wrap(bernoulli);
  at::native::uniform_(out_uniform, 0, K, gen_);
  at::native::uniform_(out_bernoulli, 0, 1, gen_);

  multinomialAliasDrawKernel
    <<<THCCeilDiv((int)n_sample+BLOCK_SIZE-1, BLOCK_SIZE), BLOCK_SIZE, 0, c10::cuda::getCurrentCUDAStream()>>>(
          size,
          THCudaLongTensor_data(state, self),
          THCudaLongTensor_data(state, _J),
          THCTensor_(data)(state, _q),
          K,
          THCTensor_(data)(state, uniform),
          THCTensor_(data)(state, bernoulli)
          );
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  THCTensor_(free)(state, uniform);
  THCTensor_(free)(state, bernoulli);
}

#endif
#endif
