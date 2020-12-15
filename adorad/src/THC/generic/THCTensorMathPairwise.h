#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THC/generic/THCTensorMathPairwise.h"
#else

THC_API int THCTensor_(equal)(THCState *state, THCTensor *self, THCTensor *src);

#if !defined(THC_REAL_IS_BOOL)

THC_API void THCTensor_(mul)(THCState *state, THCTensor *self, THCTensor *src, scalar_t value);

#endif

#endif
