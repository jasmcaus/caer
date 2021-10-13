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
from ._constants import GRAY2BGR, GRAY2RGB
from ._bgr import bgr2lab, bgr2hsv, bgr2hls, bgr2yuv, bgr2luv

__all__ = [
    "gray2rgb",
    "gray2bgr",
    "gray2lab",
    "gray2hsv",
    "gray2hls",
    "gray2yuv",
    "gray2luv",
]


def _is_gray_image(tens: Tensor):
    # tens = to_tensor(tens)
    # return tens.is_gray()
    return (len(tens.shape) == 2) or (len(tens.shape) == 3 and tens.shape[-1] == 1)


def gray2rgb(tens: Tensor) -> Tensor:
    r"""
        Converts a Grayscale Tensor to its RGB version.

    Args:
        tens (Tensor): Valid Grayscale Tensor

    Returns:
        RGB Tensor of shape ``(height, width, channels)``

    Raises:
        ValueError: If `tens` is not of shape 2

    """
    if not _is_gray_image(tens):
        raise ValueError(
            f"Tensor of shape 2 expected. Found shape {len(tens.shape)}. "
            "This function converts a Grayscale Tensor to its RGB counterpart"
        )

    img = cv.cvtColor(tens, GRAY2RGB)
    return to_tensor(img, cspace="rgb")


def gray2bgr(tens: Tensor) -> Tensor:
    r"""
        Converts a Grayscale Tensor to its BGR version.

    Args:
        tens (Tensor): Valid Grayscale Tensor

    Returns:
        BGR Tensor of shape ``(height, width, channels)``

    Raises:
        ValueError: If `tens` is not of shape 2

    """
    if not _is_gray_image(tens):
        raise ValueError(
            f"Tensor of shape 2 expected. Found shape {len(tens.shape)}. "
            "This function converts a Grayscale Tensor to its BGR counterpart"
        )

    img = cv.cvtColor(tens, GRAY2BGR)
    return to_tensor(img, cspace="bgr")


def gray2hsv(tens: Tensor) -> Tensor:
    r"""
        Converts a Grayscale Tensor to its HSV version.

    Args:
        tens (Tensor): Valid Grayscale Tensor

    Returns:
        HSV Tensor of shape ``(height, width, channels)``

    Raises:
        ValueError: If `tens` is not of shape 2

    """
    if not _is_gray_image(tens):
        raise ValueError(
            f"Tensor of shape 2 expected. Found shape {len(tens.shape)}. "
            "This function converts a LAB Tensor to its HSV counterpart"
        )

    bgr = gray2bgr(tens)

    img = bgr2hsv(bgr)
    return to_tensor(img, cspace="hsv")


def gray2hls(tens: Tensor) -> Tensor:
    r"""
        Converts a Grayscale Tensor to its HLS version.

    Args:
        tens (Tensor): Valid Grayscale Tensor

    Returns:
        HLS Tensor of shape ``(height, width, channels)``

    Raises:
        ValueError: If `tens` is not of shape 2

    """
    if not _is_gray_image(tens):
        raise ValueError(
            f"Tensor of shape 2 expected. Found shape {len(tens.shape)}. "
            "This function converts a LAB Tensor to its HLS counterpart"
        )

    bgr = gray2bgr(tens)

    img = bgr2hls(bgr)
    return to_tensor(img, cspace="hls")


def gray2lab(tens: Tensor) -> Tensor:
    r"""
        Converts a Grayscale Tensor to its LAB version.

    Args:
        tens (Tensor): Valid Grayscale Tensor

    Returns:
        LAB Tensor of shape ``(height, width, channels)``

    Raises:
        ValueError: If `tens` is not of shape 2

    """
    if not _is_gray_image(tens):
        raise ValueError(
            f"Tensor of shape 2 expected. Found shape {len(tens.shape)}. "
            "This function converts a Grayscale Tensor to its LAB counterpart"
        )

    bgr = gray2bgr(tens)

    img = bgr2lab(bgr)
    return to_tensor(img, cspace="lab")


def gray2yuv(tens: Tensor) -> Tensor:
    r"""
        Converts a Grayscale Tensor to its YUV version.

    Args:
        tens (Tensor): Valid Grayscale Tensor

    Returns:
        YUV Tensor of shape ``(height, width, channels)``

    Raises:
        ValueError: If `tens` is not of shape 2

    """
    if not _is_gray_image(tens):
        raise ValueError(
            f"Tensor of shape 2 expected. Found shape {len(tens.shape)}. "
            "This function converts a Grayscale Tensor to its YUV counterpart"
        )

    bgr = gray2bgr(tens)

    img = bgr2yuv(bgr)
    return to_tensor(img, cspace="yuv")


def gray2luv(tens: Tensor) -> Tensor:
    r"""
        Converts a Grayscale Tensor to its LUV version.

    Args:
        tens (Tensor): Valid Grayscale Tensor

    Returns:
        LUV Tensor of shape ``(height, width, channels)``

    Raises:
        ValueError: If `tens` is not of shape 2

    """
    if not _is_gray_image(tens):
        raise ValueError(
            f"Tensor of shape 2 expected. Found shape {len(tens.shape)}. "
            "This function converts a Grayscale Tensor to its LUV counterpart"
        )

    bgr = gray2bgr(tens)

    img = bgr2luv(bgr)
    return to_tensor(img, cspace="luv")
