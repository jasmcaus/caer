#    _____           ______  _____ 
#  / ____/    /\    |  ____ |  __ \
# | |        /  \   | |__   | |__) | Caer - Modern Computer Vision
# | |       / /\ \  |  __|  |  _  /  Languages: Python, C, C++
# | |___   / ____ \ | |____ | | \ \  http://github.com/jasmcaus/caer
#  \_____\/_/    \_ \______ |_|  \_\

# Licensed under the MIT License <http://opensource.org/licenses/MIT>
# SPDX-License-Identifier: MIT
# Copyright (c) 2020 The Caer Authors <http://github.com/jasmcaus>


import math
import torch
import numpy as np
from torch._six import inf


class __PrinterOptions(object):
    precision = 4
    threshold = 1000
    edgeitems = 3
    linewidth = 80
    sci_mode = None
    prefix = 'tensor('


PRINT_OPTS = __PrinterOptions()


# We could use **kwargs, but this will give better docs
def set_printoptions(precision=None, threshold=None,edgeitems=None, linewidth=None,profile=None,sci_mode=None):
    r"""Set options for printing. Items shamelessly taken from NumPy

    Args:
        precision: Number of digits of precision for floating point output
            (default = 4).
        threshold: Total number of array elements which trigger summarization
            rather than full `repr` (default = 1000).
        edgeitems: Number of array items in summary at beginning and end of
            each dimension (default = 3).
        linewidth: The number of characters per line for the purpose of
            inserting line breaks (default = 80). Thresholded matrices will
            ignore this parameter.
        profile: Sane defaults for pretty printing. Can override with any of
            the above options. (any one of `default`, `short`, `full`)
        sci_mode: Enable (True) or disable (False) scientific notation. If
            None (default) is specified, the value is defined by
            `torch._tensor_str._Formatter`. This value is automatically chosen
            by the framework.
    """
    if profile is not None:
        if profile == "default":
            PRINT_OPTS.precision = 4
            PRINT_OPTS.threshold = 1000
            PRINT_OPTS.edgeitems = 3
            PRINT_OPTS.linewidth = 80
        elif profile == "short":
            PRINT_OPTS.precision = 2
            PRINT_OPTS.threshold = 1000
            PRINT_OPTS.edgeitems = 2
            PRINT_OPTS.linewidth = 80
        elif profile == "full":
            PRINT_OPTS.precision = 4
            PRINT_OPTS.threshold = inf
            PRINT_OPTS.edgeitems = 3
            PRINT_OPTS.linewidth = 80

    if precision is not None:
        PRINT_OPTS.precision = precision
    if threshold is not None:
        PRINT_OPTS.threshold = threshold
    if edgeitems is not None:
        PRINT_OPTS.edgeitems = edgeitems
    if linewidth is not None:
        PRINT_OPTS.linewidth = linewidth
    PRINT_OPTS.sci_mode = sci_mode
 

class _Formatter(object):
    def __init__(self, tensor):
        self.floating_dtype = tensor.dtype.is_floating_point
        self.int_mode = True
        self.sci_mode = False
        self.max_width = 1

        tensor_view = tensor.reshape(-1)

        if not self.floating_dtype:
            for value in tensor_view:
                value_str = '{}'.format(value)
                # d = max(self.max_width, len(value_str))
                d = max(2, len(value_str))
                self.max_width = d

        else:
            # FLOATING POINT

            for value in tensor_view:
                if value != torch.ceil(value):
                    print('YE')
                    self.int_mode = False
                    break

            if self.int_mode:
                print('ee')
                for value in tensor_view:
                    value_str = ('{:.0f}').format(value)
                    self.max_width = max(self.max_width, len(value_str) + 1)


        if PRINT_OPTS.sci_mode is not None:
            self.sci_mode = PRINT_OPTS.sci_mode


    def width(self):
        return self.max_width


    def format(self, value):
        if self.floating_dtype:
            if self.sci_mode:
                ret = ('{{:{}.{}e}}').format(self.max_width, PRINT_OPTS.precision).format(value)

            elif self.int_mode:
                ret = '{:.0f}'.format(value)
                if not (math.isinf(value) or math.isnan(value)):
                    ret += '.'

            else:
                ret = ('{{:.{}f}}').format(PRINT_OPTS.precision).format(value)

        else:
            ret = '{}'.format(value)

        # return (self.max_width - len(ret)) * ' ' + ret
        return (2 - len(ret)) * ' ' + ret

def _scalar_str(self, formatter):
    # Usually, we must never come here. 
    # This is only for when the 'Adorad' library is built.
    # Changes may be made to this.
    return formatter.format(self.item())


def _vector_str(self, indent, summarize, formatter):
    # length includes spaces and comma between elements
    element_length = formatter.width() + 2

    elements_per_line = max(1, int(math.floor((PRINT_OPTS.linewidth - indent) / (element_length))))
    char_per_line = element_length * elements_per_line

    def _val_formatter(val, formatter=formatter):
        return formatter.format(val)

    # Preventing the entire tensor from being displayed to the terminal. 
    # We (figuratively) "prune" the tensor for output
    if summarize and self.size(0) > 2 * PRINT_OPTS.edgeitems:
        data = ([_val_formatter(val) for val in self[:PRINT_OPTS.edgeitems].tolist()] +
                [' ...'] +
                [_val_formatter(val) for val in self[-PRINT_OPTS.edgeitems:].tolist()])
    else:
        data = [_val_formatter(val) for val in self.tolist()]

    data_lines = [data[i:i + elements_per_line] for i in range(0, len(data), elements_per_line)]
    lines = [', '.join(line) for line in data_lines]

    return '[' + (',' + '\n' + ' ' * (indent + 1)).join(lines) + ']'


def _tensor_str_with_formatter(self, indent, summarize, formatter):
    dim = self.dim()
    # dim = self.ndim()

    # if dim == 0:
    #     return _scalar_str(self, formatter)

    if dim == 1:
        return _vector_str(self, indent, summarize, formatter)

    # Preventing the entire tensor from being displayed to the terminal. 
    # We (figuratively) "prune" the tensor for output
    if summarize and self.size(0) > 2 * PRINT_OPTS.edgeitems:
        slices = ([_tensor_str_with_formatter(self[i], indent + 1, summarize, formatter)
                   for i in range(0, PRINT_OPTS.edgeitems)] +
                  ['...'] +
                  [_tensor_str_with_formatter(self[i], indent + 1, summarize, formatter)
                   for i in range(len(self) - PRINT_OPTS.edgeitems, len(self))])

    # If tensor is small enough to display to terminal
    else:
        slices = [_tensor_str_with_formatter(self[i], indent + 1, summarize, formatter)
                  for i in range(0, self.size(0))]

    tensor_str = (',' + '\n' * (dim - 1) + ' ' * (indent + 1)).join(slices)
    return '[' + tensor_str + ']'


def _tensor_str(self, indent):
    if self.numel() == 0:
        return '[]'

    summarize = self.numel() > PRINT_OPTS.threshold
    # summarize = self.size > PRINT_OPTS.threshold

    if self.dtype is torch.float16 or self.dtype is torch.bfloat16:
        self = self.float()


    formatter = _Formatter(get_summarized_data(self) if summarize else self)

    x = _tensor_str_with_formatter(self, indent, summarize, formatter)

    return x


def _add_suffixes(tensor_str, suffixes, indent, force_newline):
    tensor_strs = [tensor_str]
    last_line_len = len(tensor_str) - tensor_str.rfind('\n') + 1

    for suffix in suffixes:
        suffix_len = len(suffix)

        if force_newline or last_line_len + suffix_len + 2 > PRINT_OPTS.linewidth:
            tensor_strs.append(',\n' + ' ' * indent + suffix)
            last_line_len = indent + suffix_len
            force_newline = False
        else:
            tensor_strs.append(', ' + suffix)
            last_line_len += suffix_len + 2

    tensor_strs.append(')')

    return ''.join(tensor_strs)


def get_summarized_data(self):
    dim = self.dim()
    # dim = self.ndim()
    
    # if dim == 0:
    #     return self

    if dim == 1:
        if self.size(0) > 2 * PRINT_OPTS.edgeitems:
            return np.cat((self[:PRINT_OPTS.edgeitems], self[-PRINT_OPTS.edgeitems:]))
        else:
            return self

    if self.size(0) > 2 * PRINT_OPTS.edgeitems:
        start = [self[i] for i in range(0, PRINT_OPTS.edgeitems)]
        end = ([self[i]
               for i in range(len(self) - PRINT_OPTS.edgeitems, len(self))])
        return torch.stack([get_summarized_data(x) for x in (start + end)])
    else:
        return torch.stack([get_summarized_data(x) for x in self])


def _str_intern(self):
    prefix = PRINT_OPTS.prefix
    indent = len(prefix)
    suffixes = []


    if self.numel() == 0:
        # Explicitly print the shape if it is not (0,), to match NumPy behavior
        if self.dim() != 1:
            suffixes.append('size=' + str(tuple(self.shape)))

        tensor_str = '[]'

    else:
        suffixes.append('dtype=' + str(self.dtype))

        tensor_str = _tensor_str(self, indent)

    return _add_suffixes(prefix + tensor_str, suffixes, indent, force_newline=self.is_sparse)


def _str(self):
    return _str_intern(self)
