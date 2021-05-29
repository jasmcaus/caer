#    _____           ______  _____
#  / ____/    /\    |  ____ |  __ \
# | |        /  \   | |__   | |__) | Caer - Modern Computer Vision
# | |       / /\ \  |  __|  |  _  /  Languages: Python, C, C++, Cuda
# | |___   / ____ \ | |____ | | \ \  http://github.com/jasmcaus/caer
#  \_____\/_/    \_ \______ |_|  \_\

# Licensed under the MIT License <http://opensource.org/licenses/MIT>
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021 The Caer Authors <http://github.com/jasmcaus>

from .core import (
    to_bgr,
    to_rgb,
    to_gray,
    to_hsv,
    to_hls,
    to_lab,
    to_yuv,
    to_luv,
    __all__ as __all_core__,
)

from ._bgr import (
    bgr2rgb,
    bgr2gray,
    bgr2hsv,
    bgr2hls,
    bgr2lab,
    bgr2yuv,
    bgr2luv,
    __all__ as __all_bgr__,
)

from ._rgb import (
    rgb2bgr,
    rgb2gray,
    rgb2hsv,
    rgb2hls,
    rgb2lab,
    rgb2yuv,
    rgb2luv,
    __all__ as __all_rgb__,
)

from ._gray import (
    gray2rgb,
    gray2bgr,
    gray2hsv,
    gray2hls,
    gray2lab,
    gray2yuv,
    gray2luv,
    __all__ as __all_gray__,
)

from ._hsv import (
    hsv2rgb,
    hsv2bgr,
    hsv2gray,
    hsv2hls,
    hsv2lab,
    hsv2yuv,
    hsv2luv,
    __all__ as __all_hsv__,
)

from ._hls import (
    hls2rgb,
    hls2bgr,
    hls2gray,
    hls2hsv,
    hls2lab,
    hls2yuv,
    hls2luv,
    __all__ as __all_hls__,
)

from ._lab import (
    lab2rgb,
    lab2bgr,
    lab2gray,
    lab2hsv,
    lab2hls,
    lab2yuv,
    lab2luv,
    __all__ as __all_lab__,
)

from ._yuv import (
    yuv2rgb,
    yuv2bgr,
    yuv2gray,
    yuv2hsv,
    yuv2hls,
    yuv2lab,
    yuv2luv,
    __all__ as __all_yuv__,
)

from ._luv import (
    luv2rgb,
    luv2bgr,
    luv2gray,
    luv2hsv,
    luv2hls,
    luv2lab,
    luv2yuv,
    __all__ as __all_luv__,
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
__all__ = (
    __all_core__
    + __all_rgb__
    + __all_hls__
    + __all_gray__
    + __all_bgr__
    + __all_hsv__
    + __all_lab__
    + __all_yuv__
    + __all_luv__
)

# # Don't pollute namespace
# del __all_const__
del __all_bgr__
del __all_rgb__
del __all_gray__
del __all_hsv__
del __all_lab__
del __all_hls__
del __all_yuv__
del __all_luv__
