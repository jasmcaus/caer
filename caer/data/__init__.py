#    _____           ______  _____ 
#  / ____/    /\    |  ____ |  __ \
# | |        /  \   | |__   | |__) | Caer - Modern Computer Vision
# | |       / /\ \  |  __|  |  _  /  Languages: Python, C, C++, Cuda
# | |___   / ____ \ | |____ | | \ \  http://github.com/jasmcaus/caer
#  \_____\/_/    \_ \______ |_|  \_\

# Licensed under the MIT License <http://opensource.org/licenses/MIT>
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2025 The Caer Authors <http://github.com/jasmcaus>

from ..io import imread 
from ..path import abspath, join
from .._base import __curr__ 
from ..annotations import Optional, Tuple
import numpy as np 

from ..coreten import Tensor

HERE = join(__curr__, 'data').replace('\\', "/") + "/"

def _get_path_to_data(name) -> str:
    return join(HERE, name)


def black_cat(target_size: Optional[Tuple[int, int]] = None, rgb: bool = True) -> Tensor:
    r"""
        Returns a standard 640x427 image Tensor (RGB, by default) of a black cat.

    Args:
        target_size (Optional[Tuple[int, int]]): Intended target size (follows the ``(width, height)`` format).
            If None, the unaltered tensor will be returned.
        rgb (bool): Boolean whether to return an RGB Tensor (default is ``True``).
    
    Returns:
        Tensor of shape ``(height, width, channels)``.

    Examples::

        >> tens = caer.data.black_cat()
        >> tens.shape
        (427, 640, 3)
        
    """
    return imread(HERE+'black_cat.jpg', target_size=target_size, rgb=rgb)


def camera(target_size: Optional[Tuple[int, int]] = None, rgb: bool = True) -> Tensor:
    r"""
        Returns a standard 640x427 image Tensor (RGB, by default) of a camera.

    Args:
        target_size (Optional[Tuple[int, int]]): Intended target size (follows the ``(width, height)`` format).
            If None, the unaltered tensor will be returned.
        rgb (bool): Boolean whether to return an RGB Tensor (default is ``True``).
    
    Returns:
        Tensor of shape ``(height, width, channels)``.

    Examples::

        >> tens = caer.data.camera()
        >> tens.shape
        (427, 640, 3)
        
    """
    return imread(HERE+'camera.jpg', target_size=target_size, rgb=rgb)


def fighter_fish(target_size: Optional[Tuple[int, int]] = None, rgb: bool = True) -> Tensor:
    r"""
        Returns a standard 640x640 image Tensor (RGB, by default) of a fighter fish.

    Args:
        target_size (Optional[Tuple[int, int]]): Intended target size (follows the ``(width, height)`` format).
            If None, the unaltered tensor will be returned.
        rgb (bool): Boolean whether to return an RGB Tensor (default is ``True``).
    
    Returns:
        Tensor of shape ``(height, width, channels)``.

    Examples::

        >> tens = caer.data.fighter_fish()
        >> tens.shape
        (640, 640, 3)
        
    """
    return imread(HERE+'fighter_fish.jpg', target_size=target_size, rgb=rgb)


def gold_fish(target_size: Optional[Tuple[int, int]] = None, rgb: bool = True) -> Tensor:
    r"""
        Returns a standard 640x901 image Tensor (RGB, by default) of a gold fish.

    Args:
        target_size (Optional[Tuple[int, int]]): Intended target size (follows the ``(width, height)`` format).
            If None, the unaltered tensor will be returned.
        rgb (bool): Boolean whether to return an RGB Tensor (default is ``True``).
    
    Returns:
        Tensor of shape ``(height, width, channels)``.

    Examples::

        >> tens = caer.data.gold_fish()
        >> tens.shape
        (901, 640, 3)
        
    """
    return imread(HERE+'gold_fish.jpg', target_size=target_size, rgb=rgb)


def guitar(target_size: Optional[Tuple[int, int]] = None, rgb: bool = True) -> Tensor:
    r"""
        Returns a standard 640x427 image Tensor (RGB, by default) of a guitar.

    Args:
        target_size (Optional[Tuple[int, int]]): Intended target size (follows the ``(width, height)`` format).
            If None, the unaltered tensor will be returned.
        rgb (bool): Boolean whether to return an RGB Tensor (default is ``True``).
    
    Returns:
        Tensor of shape ``(height, width, channels)``.

    Examples::

        >> tens = caer.data.guitar()
        >> tens.shape
        (427, 640, 3)
        
    """
    return imread(HERE+'guitar.jpg', target_size=target_size, rgb=rgb)


def puppy(target_size: Optional[Tuple[int, int]] = None, rgb: bool = True) -> Tensor:
    r"""
        Returns a standard 640x512 image Tensor (RGB, by default) of a puppy.

    Args:
        target_size (Optional[Tuple[int, int]]): Intended target size (follows the ``(width, height)`` format).
            If None, the unaltered tensor will be returned.
        rgb (bool): Boolean whether to return an RGB Tensor (default is ``True``).
    
    Returns:
        Tensor of shape ``(height, width, channels)``.

    Examples::

        >> tens = caer.data.puppy()
        >> tens.shape
        (512, 640, 3)
        
    """
    return imread(HERE+'puppy.jpg', target_size=target_size, rgb=rgb)


def snow(target_size: Optional[Tuple[int, int]] = None, rgb: bool = True) -> Tensor:
    r"""
        Returns a standard 640x360 image Tensor (RGB, by default) of snow.

    Args:
        target_size (Optional[Tuple[int, int]]): Intended target size (follows the ``(width, height)`` format).
            If None, the unaltered tensor will be returned.
        rgb (bool): Boolean whether to return an RGB Tensor (default is ``True``).
    
    Returns:
        Tensor of shape ``(height, width, channels)``.

    Examples::

        >> tens = caer.data.snow()
        >> tens.shape
        (360, 640, 3)
        
    """
    return imread(HERE+'snow.jpg', target_size=target_size, rgb=rgb)


def sunrise(target_size: Optional[Tuple[int, int]] = None, rgb: bool = True) -> Tensor:
    r"""
        Returns a standard 640x427 image Tensor (RGB, by default) of a sunrise landscape.

    Args:
        target_size (Optional[Tuple[int, int]]): Intended target size (follows the ``(width, height)`` format).
            If None, the unaltered tensor will be returned.
        rgb (bool): Boolean whether to return an RGB Tensor (default is ``True``).
    
    Returns:
        Tensor of shape ``(height, width, channels)``.

    Examples::

        >> tens = caer.data.sunrise()
        >> tens.shape
        (427, 640, 3)
        
    """
    return imread(HERE+'sunrise.jpg', target_size=target_size, rgb=rgb)


def tent(target_size: Optional[Tuple[int, int]] = None, rgb: bool = True) -> Tensor:
    r"""
        Returns a standard 640x427 image Tensor (RGB, by default) of a tent.

    Args:
        target_size (Optional[Tuple[int, int]]): Intended target size (follows the ``(width, height)`` format).
            If None, the unaltered tensor will be returned.
        rgb (bool): Boolean whether to return an RGB Tensor (default is ``True``).
    
    Returns:
        Tensor of shape ``(height, width, channels)``.

    Examples::

        >> tens = caer.data.tent()
        >> tens.shape
        (427, 640, 3)
        
    """
    return imread(HERE+'tent.jpg', target_size=target_size, rgb=rgb)


__all__ = [d for d in dir() if not d.startswith('_')]