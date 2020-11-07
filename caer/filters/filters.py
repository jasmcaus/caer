#### WIP #########

# import numpy as np 
# from collections.abc import Iterable


# def convolve(inp, weights, output=None, mode='reflect', cval=0.0,
#              origin=0):
#     return _correlate_or_convolve(inp, weights, output, mode, cval, origin, True)


# def _invalid_origin(origin, lenw):
#     return (origin < -(lenw // 2)) or (origin > (lenw - 1) // 2)


# def _correlate_or_convolve(inp, weights, output, mode, cval, origin,
#                            convolution):
#     inp = np.asarray(inp)

#     if np.iscomplexobj(inp):
#         raise TypeError('Complex type not supported')

#     origins = _normalize_sequence(origin, inp.ndim)
#     weights = np.asarray(weights, dtype=np.float64)
#     wshape = [ii for ii in weights.shape if ii > 0]

#     if len(wshape) != inp.ndim:
#         raise RuntimeError('filter weights array has incorrect shape.')

#     if convolution:
#         weights = weights[tuple([slice(None, None, -1)] * weights.ndim)]
#         for ii in range(len(origins)):
#             origins[ii] = -origins[ii]
#             if not weights.shape[ii] & 1:
#                 origins[ii] -= 1

#     for origin, lenw in zip(origins, wshape):
#         if _invalid_origin(origin, lenw):
#             raise ValueError('Invalid origin; origin must satisfy '
#                              '-(weights.shape[k] // 2) <= origin[k] <= '
#                              '(weights.shape[k]-1) // 2')

#     if not weights.flags.contiguous:
#         weights = weights.copy()

#     output = _get_output(output, inp)
#     temp_needed = np.may_share_memory(inp, output)

#     if temp_needed:
#         # inp and output arrays cannot share memory
#         temp = output
#         output = _get_output(output.dtype, inp)

#     if not isinstance(mode, str) and isinstance(mode, Iterable):
#         raise RuntimeError('A sequence of modes is not supported')

#     mode = _extend_mode_to_code(mode)

#     _nd_image.correlate(inp, weights, output, mode, cval, origins)

#     if temp_needed:
#         temp[...] = output
#         output = temp
#     return output


# def _extend_mode_to_code(mode):
#     """Convert an extension mode to the corresponding integer code.
#     """
#     if mode == 'nearest':
#         return 0
#     elif mode == 'wrap':
#         return 1
#     elif mode == 'reflect':
#         return 2
#     elif mode == 'mirror':
#         return 3
#     elif mode == 'constant':
#         return 4
#     else:
#         raise RuntimeError('boundary mode not supported')


# def _normalize_sequence(inp, rank):
#     """If inp is a scalar, create a sequence of length equal to the
#     rank by duplicating the inp. If inp is a sequence,
#     check if its length is equal to the length of array.
#     """
#     is_str = isinstance(inp, str)
#     if not is_str and isinstance(inp, Iterable):
#         normalized = list(inp)
#         if len(normalized) != rank:
#             err = "sequence argument must have length equal to inp rank"
#             raise RuntimeError(err)
#     else:
#         normalized = [inp] * rank
#     return normalized


# def _get_output(output, inp, shape=None):
#     if shape is None:
#         shape = inp.shape
#     if output is None:
#         output = np.zeros(shape, dtype=inp.dtype.name)
#     elif isinstance(output, (type, np.dtype)):
#         # Classes (like `np.float32`) and dtypes are interpreted as dtype
#         output = np.zeros(shape, dtype=output)
#     elif isinstance(output, str):
#         output = np.typeDict[output]
#         output = np.zeros(shape, dtype=output)
#     elif output.shape != shape:
#         raise RuntimeError("output shape not correct")
#     return output