#    _____           ______  _____
#  / ____/    /\    |  ____ |  __ \
# | |        /  \   | |__   | |__) | Caer - Modern Computer Vision
# | |       / /\ \  |  __|  |  _  /  Languages: Python, C, C++, Cuda
# | |___   / ____ \ | |____ | | \ \  http://github.com/jasmcaus/caer
#  \_____\/_/    \_ \______ |_|  \_\

# Licensed under the MIT License <http://opensource.org/licenses/MIT>
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021 The Caer Authors <http://github.com/jasmcaus>

from ..coreten import Tensor, to_tensor
from ..annotations import Union, Optional

from ._bgr import bgr2gray, bgr2hsv, bgr2lab, bgr2rgb, bgr2hls, bgr2yuv, bgr2luv
from ._rgb import rgb2gray, rgb2hsv, rgb2lab, rgb2bgr, rgb2hls, rgb2yuv, rgb2luv
from ._gray import gray2lab, gray2rgb, gray2hsv, gray2bgr, gray2hls, gray2yuv, gray2luv
from ._hsv import hsv2gray, hsv2rgb, hsv2lab, hsv2bgr, hsv2hls, hsv2yuv, hsv2luv
from ._hls import hls2gray, hls2rgb, hls2lab, hls2bgr, hls2hsv, hls2yuv, hls2luv
from ._lab import lab2gray, lab2rgb, lab2hsv, lab2bgr, lab2hls, lab2yuv, lab2luv
from ._yuv import yuv2bgr, yuv2rgb, yuv2hsv, yuv2hls, yuv2gray, yuv2lab, yuv2luv
from ._luv import luv2bgr, luv2rgb, luv2hsv, luv2hls, luv2gray, luv2lab, luv2yuv

__all__ = [
    "to_rgb",
    "to_bgr",
    "to_gray",
    "to_hsv",
    "to_hls",
    "to_lab",
    "to_yuv",
    "to_luv",
]


def to_rgb(tens: Tensor) -> Tensor: # type: ignore[return]
    r"""
        Converts any supported colorspace to RGB

    Args:
        tens (Tensor)

    Returns:
        Tensor
    """
    if not isinstance(tens, Tensor):
        raise TypeError("`tens` must be a caer.Tensor")

    # Convert to tensor
    tens._nullprt()  # raises a ValueError if we're dealing with a Foreign Tensor with illegal `.cspace` value
    cspace = tens.cspace

    # If "null", we assume we have a brand new Tensor
    if cspace == "null":
        print(
            "Warning: Caer was unable to assign a colorspace for a foreign tensor. Sticking with `rgb`."
            "This issue will be fixed in a future update."
        )

        # We assume that the tens is a BGR image
        img = bgr2rgb(tens)
        img = to_tensor(img, cspace="rgb")
        return img

    elif cspace == "rgb":
        return tens

    elif cspace == "bgr":
        img = bgr2rgb(tens)
        return to_tensor(img, cspace="rgb")

    elif cspace == "gray":
        img = gray2rgb(tens)
        return to_tensor(img, cspace="rgb")

    elif cspace == "hls":
        img = hls2rgb(tens)
        return to_tensor(img, cspace="rgb")

    elif cspace == "hsv":
        img = hsv2rgb(tens)
        return to_tensor(img, cspace="rgb")

    elif cspace == "lab":
        img = lab2rgb(tens)
        return to_tensor(img, cspace="rgb")

    elif cspace == "yuv":
        img = yuv2rgb(tens)
        return to_tensor(img, cspace="rgb")

    elif cspace == "luv":
        img = luv2rgb(tens)
        return to_tensor(img, cspace="rgb")


def to_bgr(tens: Tensor) -> Tensor: # type: ignore[return]
    r"""
        Converts any supported colorspace to BGR

    Args:
        tens (Tensor)

    Returns:
        Tensor
    """
    if not isinstance(tens, Tensor):
        raise TypeError("`tens` must be a caer.Tensor")

    # Convert to tensor
    tens._nullprt() # raises a ValueError if we're dealing with a Foreign Tensor with illegal `.cspace` value
    cspace = tens.cspace

    if cspace == "bgr":
        return tens

    elif cspace == "gray":
        img = gray2bgr(tens)
        return to_tensor(img, cspace="bgr")

    elif cspace == "rgb":
        img = rgb2bgr(tens)
        return to_tensor(img, cspace="bgr")

    elif cspace == "hls":
        img = hls2bgr(tens)
        return to_tensor(img, cspace="bgr")

    elif cspace == "hsv":
        img = hsv2bgr(tens)
        return to_tensor(img, cspace="bgr")

    elif cspace == "lab":
        img = lab2bgr(tens)
        return to_tensor(img, cspace="bgr")

    elif cspace == "yuv":
        img = yuv2bgr(tens)
        return to_tensor(img, cspace="bgr")

    elif cspace == "luv":
        img = luv2bgr(tens)
        return to_tensor(img, cspace="bgr")


def to_gray(tens: Tensor) -> Tensor: # type: ignore[return]
    r"""
        Converts any supported colorspace to grayscale

    Args:
        tens (Tensor)

    Returns:
        Tensor
    """
    if not isinstance(tens, Tensor):
        raise TypeError("`tens` must be a caer.Tensor")

    # Convert to tensor
    tens._nullprt() # raises a ValueError if we're dealing with a Foreign Tensor with illegal `.cspace` value
    cspace = tens.cspace

    if cspace == "gray":
        return tens

    elif cspace == "bgr":
        img = bgr2gray(tens)
        return to_tensor(img, cspace="gray")

    elif cspace == "rgb":
        img = rgb2gray(tens)
        return to_tensor(img, cspace="gray")

    elif cspace == "hls":
        img = hls2gray(tens)
        return to_tensor(img, cspace="gray")

    elif cspace == "hsv":
        img = hsv2gray(tens)
        return to_tensor(img, cspace="gray")

    elif cspace == "lab":
        img = lab2gray(tens)
        return to_tensor(img, cspace="gray")

    elif cspace == "yuv":
        img = yuv2gray(tens)
        return to_tensor(img, cspace="gray")

    elif cspace == "luv":
        img = luv2gray(tens)
        return to_tensor(img, cspace="gray")


def to_hsv(tens: Tensor) -> Tensor: # type: ignore[return]
    r"""
        Converts any supported colorspace to hsv

    Args:
        tens (Tensor)

    Returns:
        Tensor
    """
    if not isinstance(tens, Tensor):
        raise TypeError("`tens` must be a caer.Tensor")

    # Convert to tensor
    tens._nullprt() # raises a ValueError if we're dealing with a Foreign Tensor with illegal `.cspace` value
    cspace = tens.cspace

    if cspace == "hsv":
        return tens

    elif cspace == "bgr":
        img = bgr2hsv(tens)
        return to_tensor(img, cspace="hsv")

    elif cspace == "rgb":
        img = rgb2hsv(tens)
        return to_tensor(img, cspace="hsv")

    elif cspace == "hls":
        img = hls2hsv(tens)
        return to_tensor(img, cspace="hsv")

    elif cspace == "gray":
        img = gray2hsv(tens)
        return to_tensor(img, cspace="hsv")

    elif cspace == "lab":
        img = lab2hsv(tens)
        return to_tensor(img, cspace="hsv")

    elif cspace == "yuv":
        img = yuv2hsv(tens)
        return to_tensor(img, cspace="hsv")

    elif cspace == "luv":
        img = luv2hsv(tens)
        return to_tensor(img, cspace="hsv")


def to_hls(tens: Tensor) -> Tensor: # type: ignore[return]
    r"""
        Converts any supported colorspace to HLS

    Args:
        tens (Tensor)

    Returns:
        Tensor
    """
    if not isinstance(tens, Tensor):
        raise TypeError("`tens` must be a caer.Tensor")

    # Convert to tensor
    tens._nullprt() # raises a ValueError if we're dealing with a Foreign Tensor with illegal `.cspace` value
    cspace = tens.cspace

    if cspace == "hls":
        return tens

    elif cspace == "bgr":
        img = bgr2hls(tens)
        return to_tensor(img, cspace="hls")

    elif cspace == "rgb":
        img = rgb2hls(tens)
        return to_tensor(img, cspace="hls")

    elif cspace == "hsv":
        img = hsv2hls(tens)
        return to_tensor(img, cspace="hls")

    elif cspace == "gray":
        img = gray2hls(tens)
        return to_tensor(img, cspace="hls")

    elif cspace == "lab":
        img = lab2hls(tens)
        return to_tensor(img, cspace="hls")

    elif cspace == "yuv":
        img = yuv2hls(tens)
        return to_tensor(img, cspace="hls")

    elif cspace == "luv":
        img = luv2hls(tens)
        return to_tensor(img, cspace="hls")


def to_lab(tens: Tensor) -> Tensor: # type: ignore[return]
    r"""
        Converts any supported colorspace to LAB

    Args:
        tens (Tensor)

    Returns:
        Tensor
    """
    if not isinstance(tens, Tensor):
        raise TypeError("`tens` must be a caer.Tensor")

    # Convert to tensor
    tens._nullprt() # raises a ValueError if we're dealing with a Foreign Tensor with illegal `.cspace` value
    cspace = tens.cspace

    if cspace == "lab":
        return tens

    elif cspace == "bgr":
        img = bgr2lab(tens)
        return to_tensor(img, cspace="lab")

    elif cspace == "rgb":
        img = rgb2lab(tens)
        return to_tensor(img, cspace="lab")

    elif cspace == "hsv":
        img = hsv2lab(tens)
        return to_tensor(img, cspace="lab")

    elif cspace == "gray":
        img = gray2lab(tens)
        return to_tensor(img, cspace="lab")

    elif cspace == "hls":
        img = hls2lab(tens)
        return to_tensor(img, cspace="lab")

    elif cspace == "yuv":
        img = yuv2lab(tens)
        return to_tensor(img, cspace="lab")

    elif cspace == "luv":
        img = luv2lab(tens)
        return to_tensor(img, cspace="lab")



def to_yuv(tens: Tensor) -> Tensor: # type: ignore[return]
    r"""
        Converts any supported colorspace to YUV

    Args:
        tens (Tensor)

    Returns:
        Tensor
    """
    if not isinstance(tens, Tensor):
        raise TypeError("`tens` must be a caer.Tensor")

    # Convert to tensor
    tens._nullprt() # raises a ValueError if we're dealing with a Foreign Tensor with illegal `.cspace` value
    cspace = tens.cspace

    if cspace == "yuv":
        return tens

    elif cspace == "bgr":
        img = bgr2yuv(tens)
        return to_tensor(img, cspace="yuv")

    elif cspace == "rgb":
        img = rgb2yuv(tens)
        return to_tensor(img, cspace="yuv")

    elif cspace == "hsv":
        img = hsv2yuv(tens)
        return to_tensor(img, cspace="yuv")

    elif cspace == "gray":
        img = gray2yuv(tens)
        return to_tensor(img, cspace="yuv")

    elif cspace == "hls":
        img = hls2yuv(tens)
        return to_tensor(img, cspace="yuv")

    elif cspace == "lab":
        img = lab2yuv(tens)
        return to_tensor(img, cspace="yuv")

    elif cspace == "luv":
        img = luv2yuv(tens)
        return to_tensor(img, cspace="yuv")


def to_luv(tens: Tensor) -> Tensor: # type: ignore[return]
    r"""
        Converts any supported colorspace to LUV

    Args:
        tens (Tensor)

    Returns:
        Tensor
    """
    if not isinstance(tens, Tensor):
        raise TypeError("`tens` must be a caer.Tensor")

    # Convert to tensor
    tens._nullprt() # raises a ValueError if we're dealing with a Foreign Tensor with illegal `.cspace` value
    cspace = tens.cspace

    if cspace == "luv":
        return tens

    elif cspace == "bgr":
        img = bgr2luv(tens)
        return to_tensor(img, cspace="luv")

    elif cspace == "rgb":
        img = rgb2luv(tens)
        return to_tensor(img, cspace="luv")

    elif cspace == "hsv":
        img = hsv2luv(tens)
        return to_tensor(img, cspace="luv")

    elif cspace == "gray":
        img = gray2luv(tens)
        return to_tensor(img, cspace="luv")

    elif cspace == "hls":
        img = hls2luv(tens)
        return to_tensor(img, cspace="luv")

    elif cspace == "lab":
        img = lab2luv(tens)
        return to_tensor(img, cspace="luv")

    elif cspace == "yuv":
        img = yuv2luv(tens)
        return to_tensor(img, cspace="luv")