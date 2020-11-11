# Copyright 2020 The Caer Authors. All Rights Reserved.
#
# Licensed under the MIT License (see LICENSE);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at <https://opensource.org/licenses/MIT>
#
# ==============================================================================

mode2int = {
    'nearest' : 0,
    'wrap' : 1,
    'reflect' : 2,
    'mirror' : 3,
    'constant' : 4,
    'ignore' : 5,
}

modes = frozenset(mode2int.keys())

def _checked_mode2int(mode, cval, fname):
    if mode not in modes:
        raise ValueError('caer.%s: `mode` not in %s' % (fname, modes))
    if mode == 'constant' and cval != 0.:
        raise NotImplementedError('Please email caer developers to get this implemented.')
    return mode2int[mode]

_check_mode = _checked_mode2int