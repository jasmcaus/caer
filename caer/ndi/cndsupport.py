#    _____           ______  _____ 
#  / ____/    /\    |  ____ |  __ \
# | |        /  \   | |__   | |__) | Caer - Modern Computer Vision
# | |       / /\ \  |  __|  |  _  /  Languages: Python, C, C++
# | |___   / ____ \ | |____ | | \ \  http://github.com/jasmcaus/caer
#  \_____\/_/    \_ \______ |_|  \_\

# Licensed under the MIT License <http://opensource.org/licenses/MIT>
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-21 The Caer Authors <http://github.com/jasmcaus>


from collections.abc import Iterable
import numpy as np


def _extend_mode_to_code(mode):
    """Convert an extension mode to the corresponding integer code.
    """
    if mode == 'nearest':
        return 0
    elif mode == 'wrap':
        return 1
    elif mode == 'reflect':
        return 2
    elif mode == 'mirror':
        return 3
    elif mode == 'constant':
        return 4
    else:
        raise RuntimeError('boundary mode not supported')


def _normalize_sequence(inp, rank):
    """If inp is a scalar, create a sequence of length equal to the
    rank by duplicating the inp. If inp is a sequence,
    check if its length is equal to the length of array.
    """
    is_str = isinstance(inp, str)

    if not is_str and isinstance(inp, Iterable):
        normalized = list(inp)
        if len(normalized) != rank:
            err = "sequence argument must have length equal to inp rank"
            raise RuntimeError(err)

    else:
        normalized = [inp] * rank
        
    return normalized


def _get_output(output, inp, shape=None, complex_output=False):
    if shape is None:
        shape = inp.shape

    if output is None:
        if not complex_output:
            output = np.zeros(shape, dtype=input.dtype.name)
        else:
            complex_type = np.promote_types(input.dtype, np.complex64)
            output = np.zeros(shape, dtype=complex_type)

    elif isinstance(output, (type, np.dtype)):
        # Classes (like `np.float32`) and dtypes are interpreted as dtype
        if complex_output and np.dtype(output).kind != 'c':
            raise RuntimeError("output must have complex dtype")

        output = np.zeros(shape, dtype=output)

    elif isinstance(output, str):
        output = np.typeDict[output]
        if complex_output and np.dtype(output).kind != 'c':
            raise RuntimeError("output must have complex dtype")

        output = np.zeros(shape, dtype=output)

    elif output.shape != shape:
        raise RuntimeError("output shape not correct")
    
    elif complex_output and output.dtype.kind != 'c':
        raise RuntimeError("output must have complex dtype")

    return output
