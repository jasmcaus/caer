#
#  _____ _____ _____ _____
# |     |     | ___  | __|  Caer - Modern Computer Vision
# |     | ___ |      | \    Languages: Python, C, C++
# |_____|     | ____ |  \   http://github.com/jasmcaus/caer

# Licensed under the MIT License <http://opensource.org/licenses/MIT>
# SPDX-License-Identifier: MIT
# Copyright (c) 2020 The Caer Authors <http://github.com/jasmcaus>


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


def _get_output(output, inp, shape=None):
    if shape is None:
        shape = inp.shape
    if output is None:
        output = np.zeros(shape, dtype=inp.dtype.name)
    elif isinstance(output, (type, np.dtype)):
        # Classes (like `np.float32`) and dtypes are interpreted as dtype
        output = np.zeros(shape, dtype=output)
    elif isinstance(output, str):
        output = np.typeDict[output]
        output = np.zeros(shape, dtype=output)
    elif output.shape != shape:
        raise RuntimeError("output shape not correct")
    return output
