#    _____           ______  _____ 
#  / ____/    /\    |  ____ |  __ \
# | |        /  \   | |__   | |__) | Caer - Modern Computer Vision
# | |       / /\ \  |  __|  |  _  /  Languages: Python, C, C++, Cuda
# | |___   / ____ \ | |____ | | \ \  http://github.com/jasmcaus/caer
#  \_____\/_/    \_ \______ |_|  \_\

# Licensed under the MIT License <http://opensource.org/licenses/MIT>
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021 The Caer Authors <http://github.com/jasmcaus>

r"""
    caer.transforms consists of position and color-based transforms for use.
"""

# Don't import from functional.py

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
    pad,
    __all__ as __all_pos__
)

from .color import (
    adjust_brightness,
    adjust_contrast,
    adjust_hue,
    adjust_saturation,
    adjust_gamma,
    affine,
    darken,
    brighten,
    random_brightness,
    correct_exposure,
    augment_random,
    __all__ as __all_color__
)

from .simulate import (
    sim_snow,
    sim_rain,
    sim_fog,
    sim_gravel,
    sim_sun_flare,
    sim_motion_blur,
    sim_autumn,
    sim_shadow,
    __all__ as __all_sim__
)

__all__ = __all_color__ + __all_pos__ + __all_sim__


# Stop polluting the namespace
del __all_pos__
del __all_color__
del __all_sim__