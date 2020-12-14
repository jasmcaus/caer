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
    caer.transforms consists of position transforms for use.
    For color-based transforms, see caer.filters.
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
    __all__ as __all_pos__
)

from .color import (
    change_light,
    darken,
    brighten,
    random_brightness,
    add_snow,
    add_rain,
    add_fog,
    add_gravel,
    add_sun_flare,
    add_motion_blur,
    add_autumn,
    add_shadow,
    correct_exposure,
    augment_random,
    __all__ as __all_color__
)

__all__ = __all_color__ + __all_pos__


# Stop polluting the namespace
del __all_pos__
del __all_color__