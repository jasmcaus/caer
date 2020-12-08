#    _____           ______  _____ 
#  / ____/    /\    |  ____ |  __ \
# | |        /  \   | |__   | |__) | Caer - Modern Computer Vision
# | |       / /\ \  |  __|  |  _  /  Languages: Python, C, C++
# | |___   / ____ \ | |____ | | \ \  http://github.com/jasmcaus/caer
#  \_____\/_/    \_ \______ |_|  \_\

# Licensed under the MIT License <http://opensource.org/licenses/MIT>
# SPDX-License-Identifier: MIT
# Copyright (c) 2020 The Caer Authors <http://github.com/jasmcaus>


from .bgr import (
    bgr_to_gray,
    bgr_to_hsv,
    bgr_to_lab,
    bgr_to_rgb,
    is_bgr_image,
    __all__ as __all_bgr__
)

from .rgb import (
    rgb_to_gray,
    rgb_to_hsv,
    rgb_to_lab,
    rgb_to_bgr,
    is_rgb_image,
    __all__ as __all_rgb__
)

from .gray import (
    gray_to_lab,
    gray_to_rgb,
    gray_to_hsv,
    gray_to_bgr,
    is_gray_image,
    __all__ as __all_gray__
)

from .hsv import (
    hsv_to_gray,
    hsv_to_rgb,
    hsv_to_lab,
    hsv_to_bgr,
    is_hsv_image,
    __all__ as __all_hsv__
)

from .lab import (
    lab_to_gray,
    lab_to_rgb,
    lab_to_hsv,
    lab_to_bgr,
    is_lab_image,
    __all__ as __all_lab__
)

from .constants import (
    IMREAD_COLOR,
    BGR2RGB,
    BGR2GRAY,
    BGR2HSV,
    RGB2GRAY,
    RGB2BGR,
    RGB2HSV,
    BGR2LAB,
    RGB2LAB,
    HSV2BGR,
    HSV2RGB,
    LAB2BGR,
    LAB2RGB,
    GRAY2BGR,
    GRAY2RGB,
    __all__ as __all_const__
)

__all__ = __all_const__ + __all_rgb__ + __all_gray__ + __all_bgr__ + __all_hsv__ + __all_lab__

# Don't pollute namespace
del __all_const__
del __all_bgr__
del __all_rgb__
del __all_gray__
del __all_hsv__
del __all_lab__