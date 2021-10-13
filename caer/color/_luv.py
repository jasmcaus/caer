#    _____           ______  _____
#  / ____/    /\    |  ____ |  __ \
# | |        /  \   | |__   | |__) | Caer - Modern Computer Vision
# | |       / /\ \  |  __|  |  _  /  Languages: Python, C, C++, Cuda
# | |___   / ____ \ | |____ | | \ \  http://github.com/jasmcaus/caer
#  \_____\/_/    \_ \______ |_|  \_\

# Licensed under the MIT License <http://opensource.org/licenses/MIT>
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021 The Caer Authors <http://github.com/jasmcaus>

import cv2 as cv

from ..coreten import Tensor, to_tensor
from ._constants import LUV2BGR, LUV2RGB
from ._bgr import bgr2gray, bgr2hls, bgr2hsv, bgr2lab, bgr2yuv

__all__ = [
    "luv2bgr",
    "luv2gray",
    "luv2hls",
    "luv2hsv",
    "luv2lab",
    "luv2rgb",
    "luv2yuv",
]


def _is_luv_image(tens):
    return len(tens.shape) == 3 and tens.shape[-1] == 3


def luv2bgr(tens: Tensor) -> Tensor:
    r"""
        Converts an LUV Tensor to its BGR version.

    Args:
        tens (Tensor): Valid LUV Tensor

    Returns:
        BGR Tensor of shape ``(height, width, channels)``

    Raises:
        ValueError: If `tens` is not of shape 3

    """
    if not _is_luv_image(tens):
        raise ValueError(
            f"Tensor of shape 3 expected. Found shape {len(tens.shape)}. "
            "This function converts an LUV Tensor to its BGR counterpart"
        )

    img = cv.cvtColor(tens, LUV2BGR)
    return to_tensor(img, cspace="bgr")


def luv2rgb(tens: Tensor) -> Tensor:
    r"""
        Converts a LUV Tensor to its RGB version.

    Args:
        tens (Tensor): Valid LUV Tensor

    Returns:
        RGB Tensor of shape ``(height, width, channels)``

    Raises:
        ValueError: If `tens` is not of shape 3

    """
    if not _is_luv_image(tens):
        raise ValueError(
            f"Tensor of shape 3 expected. Found shape {len(tens.shape)}." 
            "This function converts a LUV Tensor to its RGB counterpart"
        )

    img = cv.cvtColor(tens, LUV2RGB)
    return to_tensor(img, cspace="rgb")


def luv2gray(tens: Tensor) -> Tensor:
    r"""
        Converts an LUV Tensor to its GRAY version.

    Args:
        tens (Tensor): Valid LUV Tensor

    Returns:
        GRAY Tensor of shape ``(height, width, channels)``

    Raises:
        ValueError: If `tens` is not of shape 3

    """
    if not _is_luv_image(tens):
        raise ValueError(
            f"Tensor of shape 3 expected. Found shape {len(tens.shape)}. "
            "This function converts an LUV Tensor to its GRAY counterpart"
        )

    img = luv2bgr(tens)
    img = bgr2gray(tens)
    return to_tensor(img, cspace="gray")


def luv2hls(tens: Tensor) -> Tensor:
    r"""
        Converts an LUV Tensor to its HLS version.

    Args:
        tens (Tensor): Valid LUV Tensor

    Returns:
        HLS Tensor of shape ``(height, width, channels)``

    Raises:
        ValueError: If `tens` is not of shape 3

    """
    if not _is_luv_image(tens):
        raise ValueError(
            f"Tensor of shape 3 expected. Found shape {len(tens.shape)}. "
            "This function converts an LUV Tensor to its HLS counterpart"
        )

    img = luv2bgr(tens)
    img = bgr2hls(tens)
    return to_tensor(img, cspace="hls")


def luv2hsv(tens: Tensor) -> Tensor:
    r"""
        Converts an LUV Tensor to its HSV version.

    Args:
        tens (Tensor): Valid LUV Tensor

    Returns:
        HSV Tensor of shape ``(height, width, channels)``

    Raises:
        ValueError: If `tens` is not of shape 3

    """
    if not _is_luv_image(tens):
        raise ValueError(
            f"Tensor of shape 3 expected. Found shape {len(tens.shape)}. "
            "This function converts an LUV Tensor to its HSV counterpart"
        )

    img = luv2bgr(tens)
    img = bgr2hsv(tens)
    return to_tensor(img, cspace="hsv")


def luv2lab(tens: Tensor) -> Tensor:
    r"""
        Converts an LUV Tensor to its LAB version.

    Args:
        tens (Tensor): Valid LUV Tensor

    Returns:
        LAB Tensor of shape ``(height, width, channels)``

    Raises:
        ValueError: If `tens` is not of shape 3

    """
    if not _is_luv_image(tens):
        raise ValueError(
            f"Tensor of shape 3 expected. Found shape {len(tens.shape)}. "
            "This function converts an LUV Tensor to its LAB counterpart"
        )

    img = luv2bgr(tens)
    img = bgr2lab(tens)
    return to_tensor(img, cspace="lab")


def luv2yuv(tens: Tensor) -> Tensor:
    r"""
        Converts an LUV Tensor to its YUV version.

    Args:
        tens (Tensor): Valid LUV Tensor

    Returns:
        YUV Tensor of shape ``(height, width, channels)``

    Raises:
        ValueError: If `tens` is not of shape 3

    """
    if not _is_luv_image(tens):
        raise ValueError(
            f"Tensor of shape 3 expected. Found shape {len(tens.shape)}. "
            "This function converts an LUV Tensor to its YUV counterpart"
        )

    img = luv2bgr(tens)
    img = bgr2yuv(tens)
    return to_tensor(img, cspace="yuv")
