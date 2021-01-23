#    _____           ______  _____ 
#  / ____/    /\    |  ____ |  __ \
# | |        /  \   | |__   | |__) | Caer - Modern Computer Vision
# | |       / /\ \  |  __|  |  _  /  Languages: Python, C, C++
# | |___   / ____ \ | |____ | | \ \  http://github.com/jasmcaus/caer
#  \_____\/_/    \_ \______ |_|  \_\

# Licensed under the MIT License <http://opensource.org/licenses/MIT>
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-21 The Caer Authors <http://github.com/jasmcaus>

from .core import (
    to_bgr,
    to_rgb,
    to_gray,
    to_hsv,
    to_hls,
    to_lab,
    __all__ as __all_core__
)

from ._bgr import (
    bgr2rgb,
    bgr2gray,
    bgr2hsv,    
    bgr2hls,
    bgr2lab,
    __all__ as __all_bgr__
)

from ._rgb import (
    rgb2bgr,
    rgb2gray,
    rgb2hsv,
    rgb2hls,
    rgb2lab,
    __all__ as __all_rgb__
)

from ._gray import (
    gray2rgb,
    gray2bgr,    
    gray2hsv,
    gray2hls,
    gray2lab,    
    __all__ as __all_gray__
)

from ._hsv import (
    hsv2rgb,
    hsv2bgr,
    hsv2gray,
    hsv2hls,
    hsv2lab,
    __all__ as __all_hsv__
)

from ._hls import (
    hls2rgb,
    hls2bgr,
    hls2gray,
    hls2hsv,
    hls2lab,
    __all__ as __all_hls__
)

from ._lab import (
    lab2rgb,
    lab2bgr,
    lab2gray,
    lab2hsv,
    lab2hls,
    __all__ as __all_lab__
)

# from .constants import (
#     IMREAD_COLOR,
#     BGR2RGB,
#     BGR2GRAY,
#     BGR2HSV,
#     RGB2GRAY,
#     RGB2BGR,
#     RGB2HSV,
#     BGR2LAB,
#     RGB2LAB,
#     HSV2BGR,
#     HSV2RGB,
#     LAB2BGR,
#     LAB2RGB,
#     GRAY2BGR,
#     GRAY2RGB,
#     HLS2BGR,
#     HLS2RGB,
#     __all__ as __all_const__
# )


# __all__ = __all_const__ + __all_core__ + __all_rgb__ + __all_hls__+ __all_gray__ + __all_bgr__ + __all_hsv__ + __all_lab__
__all__ = __all_core__ + __all_rgb__ + __all_hls__+ __all_gray__ + __all_bgr__ + __all_hsv__ + __all_lab__

# # Don't pollute namespace
# del __all_const__
del __all_bgr__
del __all_rgb__
del __all_gray__
del __all_hsv__
del __all_lab__
del __all_hls__