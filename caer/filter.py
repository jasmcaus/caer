#
#  _____ _____ _____ ____
# |     | ___ | ___  | __|  Caer - Modern Computer Vision
# |     |     |      | \    version 3.9.1
# |_____|     | ____ |  \   http://github.com/jasmcaus/caer

# Licensed under the MIT License <http://opensource.org/licenses/MIT>.
# SPDX-License-Identifier: MIT
# Copyright (c) 2020 The Caer Authors <http://github.com/jasmcaus>.
# 

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