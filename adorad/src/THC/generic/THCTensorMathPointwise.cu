#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THC/generic/THCTensorMathPointwise.cu"
#else

#include <ATen/MemoryOverlap.h>
#include <ATen/NamedTensorUtils.h>

#if !defined(THC_REAL_IS_BOOL)

static void propagate_names_if_named_tensor_enabled(THCTensor* result, THCTensor* src) {
  at::namedinference::propagate_names(result, src);
}

#define IMPLEMENT_CUDA_TENSOR_BASIC_FUNC_(NAME, CFUNC, REAL)             \
  struct Tensor_##NAME##_##REAL##_Op {                                  \
    __device__ __forceinline__ void operator()(scalar_t* out, scalar_t* in) const { \
      *out = CFUNC(*in);                                                \
    }                                                                   \
                                                                        \
    __device__ __forceinline__ void operator()(scalar_t* v) const {         \
      *v = CFUNC(*v);                                                   \
    }                                                                   \
  };                                                                    \
                                                                        \
  void THCTensor_(NAME)(THCState* state, THCTensor* self_, THCTensor* src) { \
    THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self_, src));       \
    at::assert_no_internal_overlap(self_);                              \
    if (self_ == src) {                                                 \
      if (!THC_pointwiseApply1<scalar_t>(state, self_, Tensor_##NAME##_##REAL##_Op())) { \
        THArgCheck(false, 2, CUTORCH_DIM_WARNING);                      \
      }                                                                 \
    } else {                                                            \
      THCTensor_(resizeAs)(state, self_, src);                          \
                                                                        \
      if (!THC_pointwiseApply2<scalar_t, scalar_t>(state, self_, src, Tensor_##NAME##_##REAL##_Op())) { \
        THArgCheck(false, 2, CUTORCH_DIM_WARNING);                      \
      }                                                                 \
    }                                                                   \
                                                                        \
    THCudaCheck(cudaGetLastError());                                    \
    propagate_names_if_named_tensor_enabled(self_, src);                \
  }

#define IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(NAME, CFUNC, REAL) \
  IMPLEMENT_CUDA_TENSOR_BASIC_FUNC_(NAME, CFUNC, REAL)

#if defined(THC_REAL_IS_FLOAT) || defined(THC_REAL_IS_DOUBLE) || defined(THC_REAL_IS_HALF)

IMPLEMENT_CUDA_TENSOR_BASIC_FUNC( sqrt, THCNumerics<scalar_t>::sqrt,  Real)

#endif
#undef IMPLEMENT_CUDA_TENSOR_BASIC_FUNC_
#undef IMPLEMENT_CUDA_TENSOR_BASIC_FUNC

void THCTensor_(crossKernel)(THCState *state, THCTensor *self, THCTensor *x, THCTensor *y, int dimension)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 3, self, x, y));

  int64_t sx = THCTensor_(stride)(state, x, dimension);
  int64_t sy = THCTensor_(stride)(state, y, dimension);
  int64_t so = THCTensor_(stride)(state, self, dimension);
  THCTensor *nx = THCTensor_(newNarrow)(state, x, dimension, 0, 1);
  THCTensor *ny = THCTensor_(newNarrow)(state, y, dimension, 0, 1);
  THCTensor *nself = THCTensor_(newNarrow)(state, self, dimension, 0, 1);
  if (!THC_pointwiseApply3<scalar_t, scalar_t, scalar_t>(state, nself, nx, ny, TensorCrossOp<scalar_t>(sx, sy, so))) {
    THArgCheck(false, 2, CUTORCH_DIM_WARNING);
  }
  THCTensor_(free)(state, nx);
  THCTensor_(free)(state, ny);
  THCTensor_(free)(state, nself);
}

namespace {
c10::intrusive_ptr<at::TensorImpl, at::UndefinedTensorImpl> retainTensorImpl(THCTensor* self) {
  c10::raw::intrusive_ptr::incref(self);
  return c10::intrusive_ptr<at::TensorImpl, at::UndefinedTensorImpl>::reclaim(self);
}
}

void THCTensor_(cmul)(THCState *state, THCTensor *self_, THCTensor *src1, THCTensor *src2)
{
  auto out = at::Tensor(retainTensorImpl(self_));
  at::mul_out(out, at::Tensor(retainTensorImpl(src1)), at::Tensor(retainTensorImpl(src2)));
}

#endif
#endif
