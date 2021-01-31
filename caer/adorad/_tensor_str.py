#    _____           ______  _____ 
#  / ____/    /\    |  ____ |  __ \
# | |        /  \   | |__   | |__) | Caer - Modern Computer Vision
# | |       / /\ \  |  __|  |  _  /  Languages: Python, C, C++, Cuda
# | |___   / ____ \ | |____ | | \ \  http://github.com/jasmcaus/caer
#  \_____\/_/    \_ \______ |_|  \_\

# Licensed under the MIT License <http://opensource.org/licenses/MIT>
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021 The Caer Authors <http://github.com/jasmcaus>


import math
import numpy as np

class __PrinterOptions(object):
    precision = 4
    threshold = 1000
    edgeitems = 3
    linewidth = 80
    sci_mode = None
    prefix = 'tensor('


PRINT_OPTS = __PrinterOptions()


class _Formatter(object):
    def __init__(self, tensor):
        self.floating_dtype = 'float' in str(repr(tensor.dtype))
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
                if value != np.ceil(value):
                    self.int_mode = False
                    break

            if self.int_mode:
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
    if summarize and self.size_dim(0) > 2 * PRINT_OPTS.edgeitems:
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
    if summarize and self.size_dim(0) > 2 * PRINT_OPTS.edgeitems:
        slices = ([_tensor_str_with_formatter(self[i], indent + 1, summarize, formatter)
                   for i in range(0, PRINT_OPTS.edgeitems)] +
                  ['...'] +
                  [_tensor_str_with_formatter(self[i], indent + 1, summarize, formatter)
                   for i in range(len(self) - PRINT_OPTS.edgeitems, len(self))])

    # If tensor is small enough to display to terminal
    else:
        slices = [_tensor_str_with_formatter(self[i], indent + 1, summarize, formatter)
                  for i in range(0, self.size_dim(0))]

    tensor_str = (',' + '\n' * (dim - 1) + ' ' * (indent + 1)).join(slices)
    return '[' + tensor_str + ']'


def _tensor_str(self, indent):
    if self.numel() == 0:
        return '[]'
    summarize = self.numel() > PRINT_OPTS.threshold
    # summarize = self.size_dim > PRINT_OPTS.threshold

    # if self.dtype is torch.float16 or self.dtype is torch.bfloat16:
    #     self = self.float()


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
    
    if dim == 0:
        return self

    if dim == 1:
        if self.size_dim(0) > 2 * PRINT_OPTS.edgeitems:
            return np.concatenate((self[:PRINT_OPTS.edgeitems], self[-PRINT_OPTS.edgeitems:]))
        else:
            return self

    if self.size_dim(0) > 2 * PRINT_OPTS.edgeitems:
        start = [self[i] for i in range(0, PRINT_OPTS.edgeitems)]
        end = ([self[i]
               for i in range(len(self) - PRINT_OPTS.edgeitems, len(self))])
        return np.stack([get_summarized_data(x) for x in (start + end)])
    else:
        return np.stack([get_summarized_data(x) for x in self])


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

    return _add_suffixes(prefix + tensor_str, suffixes, indent, force_newline=False)


def _str(self):
    return _str_intern(self)
