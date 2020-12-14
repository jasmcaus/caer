#    _____           ______  _____ 
#  / ____/    /\    |  ____ |  __ \
# | |        /  \   | |__   | |__) | Caer - Modern Computer Vision
# | |       / /\ \  |  __|  |  _  /  Languages: Python, C, C++
# | |___   / ____ \ | |____ | | \ \  http://github.com/jasmcaus/caer
#  \_____\/_/    \_ \______ |_|  \_\

# Licensed under the MIT License <http://opensource.org/licenses/MIT>
# SPDX-License-Identifier: MIT
# Copyright (c) 2020 The Caer Authors <http://github.com/jasmcaus>

r"""
    caer.transforms consists of 1) Color and 2) Position transforms for use.
"""

from .position import (
    hflip,
    vflip,
    hvflip,
    rand_flip,
    scale,
    translate,
    transpose,
    rotate,
    crop,
    center_crop,
    rand_crop,
    posterize,
    solarize,
    equalize,
    clip,
    __all__ as __all_trans__
)

from .color import *

__all__ = __all_trans__ 

# Stop polluting the namespace
del __all_trans__