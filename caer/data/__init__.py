#    _____           ______  _____ 
#  / ____/    /\    |  ____ |  __ \
# | |        /  \   | |__   | |__) | Caer - Modern Computer Vision
# | |       / /\ \  |  __|  |  _  /  Languages: Python, C, C++, Cuda
# | |___   / ____ \ | |____ | | \ \  http://github.com/jasmcaus/caer
#  \_____\/_/    \_ \______ |_|  \_\

# Licensed under the MIT License <http://opensource.org/licenses/MIT>
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021 The Caer Authors <http://github.com/jasmcaus>

from ..io import imread 
from ..path import abspath, minijoin
from .._base import __curr__ 
from ..jit.annotations import Optional
import numpy as np 

from ..adorad import Tensor

here = minijoin(__curr__, 'data').replace('\\', "/") + "/"

def _get_path_to_data(name) -> str:
    return minijoin(here, name)


def audio_mixer(target_size: Optional[tuple] = None, rgb: bool = True) -> Tensor:
    r"""
        Returns a standard 640x427 image Tensor (RGB, by default) of an audio mixer.

    Args:
        target_size (Optional[tuple]): Intended target size (follows the ``(width, height)`` format).
            If None, the unaltered tensor will be returned.
        rgb (bool): Boolean whether to return an RGB Tensor (default is ``True``).
    
    Returns:
        Tensor of shape ``(height, width, channels)``.

    Examples::

        >> tens = caer.data.audio_mixer()
        >> tens.shape
        (427, 640, 3)

    """
    return imread(here+'audio_mixer.jpg', target_size=target_size, rgb=rgb)


def bear(target_size: Optional[tuple] = None, rgb: bool = True) -> Tensor:
    r"""
        Returns a standard 640x427 image Tensor (RGB, by default) of a bear.

    Args:
        target_size (Optional[tuple]): Intended target size (follows the ``(width, height)`` format).
            If None, the unaltered tensor will be returned.
        rgb (bool): Boolean whether to return an RGB Tensor (default is ``True``).
    
    Returns:
        Tensor of shape ``(height, width, channels)``.

    Examples::

        >> tens = caer.data.bear()
        >> tens.shape
        (427, 640, 3)

    """
    return imread(here+'bear.jpg', target_size=target_size, rgb=rgb)


def beverages(target_size: Optional[tuple] = None, rgb: bool = True) -> Tensor:
    r"""
        Returns a standard 640x427 image Tensor (RGB, by default) of beverages.

    Args:
        target_size (Optional[tuple]): Intended target size (follows the ``(width, height)`` format).
            If None, the unaltered tensor will be returned.
        rgb (bool): Boolean whether to return an RGB Tensor (default is ``True``).
    
    Returns:
        Tensor of shape ``(height, width, channels)``.

    Examples::

        >> tens = caer.data.beverages()
        >> tens.shape
        (427, 640, 3)
        
    """
    return imread(here+'beverages.jpg', target_size=target_size, rgb=rgb)


def black_cat(target_size: Optional[tuple] = None, rgb: bool = True) -> Tensor:
    r"""
        Returns a standard 640x427 image Tensor (RGB, by default) of a black cat.

    Args:
        target_size (Optional[tuple]): Intended target size (follows the ``(width, height)`` format).
            If None, the unaltered tensor will be returned.
        rgb (bool): Boolean whether to return an RGB Tensor (default is ``True``).
    
    Returns:
        Tensor of shape ``(height, width, channels)``.

    Examples::

        >> tens = caer.data.black_cat()
        >> tens.shape
        (427, 640, 3)
        
    """
    return imread(here+'black_cat.jpg', target_size=target_size, rgb=rgb)


def blue_tang(target_size: Optional[tuple] = None, rgb: bool = True) -> Tensor:
    r"""
        Returns a standard 640x414 image Tensor (RGB, by default) of a blue tang (a type of fish).

    Args:
        target_size (Optional[tuple]): Intended target size (follows the ``(width, height)`` format).
            If None, the unaltered tensor will be returned.
        rgb (bool): Boolean whether to return an RGB Tensor (default is ``True``).
    
    Returns:
        Tensor of shape ``(height, width, channels)``.

    Examples::

        >> tens = caer.data.audio_mixer()
        >> tens.shape
        (414, 640, 3)
        
    """
    return imread(here+'blue_tang.jpg', target_size=target_size, rgb=rgb)


def camera(target_size: Optional[tuple] = None, rgb: bool = True) -> Tensor:
    r"""
        Returns a standard 640x427 image Tensor (RGB, by default) of a camera.

    Args:
        target_size (Optional[tuple]): Intended target size (follows the ``(width, height)`` format).
            If None, the unaltered tensor will be returned.
        rgb (bool): Boolean whether to return an RGB Tensor (default is ``True``).
    
    Returns:
        Tensor of shape ``(height, width, channels)``.

    Examples::

        >> tens = caer.data.camera()
        >> tens.shape
        (427, 640, 3)
        
    """
    return imread(here+'camera.jpg', target_size=target_size, rgb=rgb)


def controller(target_size: Optional[tuple] = None, rgb: bool = True) -> Tensor:
    r"""
        Returns a standard 640x427 image Tensor (RGB, by default) of a game controller.

    Args:
        target_size (Optional[tuple]): Intended target size (follows the ``(width, height)`` format).
            If None, the unaltered tensor will be returned.
        rgb (bool): Boolean whether to return an RGB Tensor (default is ``True``).
    
    Returns:
        Tensor of shape ``(height, width, channels)``.

    Examples::

        >> tens = caer.data.controller()
        >> tens.shape
        (427, 640, 3)
        
    """
    return imread(here+'controller.jpg', target_size=target_size, rgb=rgb)


def drone(target_size: Optional[tuple] = None, rgb: bool = True) -> Tensor:
    r"""
        Returns a standard 640x358 image Tensor (RGB, by default) of a robotic drone.

    Args:
        target_size (Optional[tuple]): Intended target size (follows the ``(width, height)`` format).
            If None, the unaltered tensor will be returned.
        rgb (bool): Boolean whether to return an RGB Tensor (default is ``True``).
    
    Returns:
        Tensor of shape ``(height, width, channels)``.

    Examples::

        >> tens = caer.data.drone()
        >> tens.shape
        (358, 640, 3)
        
    """
    return imread(here+'drone.jpg', target_size=target_size, rgb=rgb)


def dusk(target_size: Optional[tuple] = None, rgb: bool = True) -> Tensor:
    r"""
        Returns a standard 640x427 image Tensor (RGB, by default) of a dusk landscape.

    Args:
        target_size (Optional[tuple]): Intended target size (follows the ``(width, height)`` format).
            If None, the unaltered tensor will be returned.
        rgb (bool): Boolean whether to return an RGB Tensor (default is ``True``).
    
    Returns:
        Tensor of shape ``(height, width, channels)``.

    Examples::

        >> tens = caer.data.dusk()
        >> tens.shape
        (427, 640, 3)
        
    """
    return imread(here+'dusk.jpg', target_size=target_size, rgb=rgb)


def fighter_fish(target_size: Optional[tuple] = None, rgb: bool = True) -> Tensor:
    r"""
        Returns a standard 640x640 image Tensor (RGB, by default) of a fighter fish.

    Args:
        target_size (Optional[tuple]): Intended target size (follows the ``(width, height)`` format).
            If None, the unaltered tensor will be returned.
        rgb (bool): Boolean whether to return an RGB Tensor (default is ``True``).
    
    Returns:
        Tensor of shape ``(height, width, channels)``.

    Examples::

        >> tens = caer.data.fighter_fish()
        >> tens.shape
        (640, 640, 3)
        
    """
    return imread(here+'fighter_fish.jpg', target_size=target_size, rgb=rgb)


def gold_fish(target_size: Optional[tuple] = None, rgb: bool = True) -> Tensor:
    r"""
        Returns a standard 640x901 image Tensor (RGB, by default) of a gold fish.

    Args:
        target_size (Optional[tuple]): Intended target size (follows the ``(width, height)`` format).
            If None, the unaltered tensor will be returned.
        rgb (bool): Boolean whether to return an RGB Tensor (default is ``True``).
    
    Returns:
        Tensor of shape ``(height, width, channels)``.

    Examples::

        >> tens = caer.data.gold_fish()
        >> tens.shape
        (901, 640, 3)
        
    """
    return imread(here+'gold_fish.jpg', target_size=target_size, rgb=rgb)


def green_controller(target_size: Optional[tuple] = None, rgb: bool = True) -> Tensor:
    r"""
        Returns a standard 640x512 image Tensor (RGB, by default) of a green game controller.

    Args:
        target_size (Optional[tuple]): Intended target size (follows the ``(width, height)`` format).
            If None, the unaltered tensor will be returned.
        rgb (bool): Boolean whether to return an RGB Tensor (default is ``True``).
    
    Returns:
        Tensor of shape ``(height, width, channels)``.

    Examples::

        >> tens = caer.data.green_controller()
        >> tens.shape
        (512, 640, 3)
        
    """
    return imread(here+'green_controller.jpg', target_size=target_size, rgb=rgb)


def green_fish(target_size: Optional[tuple] = None, rgb: bool = True) -> Tensor:
    r"""
        Returns a standard 640x430 image Tensor (RGB, by default) of a green fish.

    Args:
        target_size (Optional[tuple]): Intended target size (follows the ``(width, height)`` format).
            If None, the unaltered tensor will be returned.
        rgb (bool): Boolean whether to return an RGB Tensor (default is ``True``).
    
    Returns:
        Tensor of shape ``(height, width, channels)``.

    Examples::

        >> tens = caer.data.green_fish()
        >> tens.shape
        (430, 640, 3)
        
    """
    return imread(here+'green_fish.jpg', target_size=target_size, rgb=rgb)


def guitar(target_size: Optional[tuple] = None, rgb: bool = True) -> Tensor:
    r"""
        Returns a standard 640x427 image Tensor (RGB, by default) of a guitar.

    Args:
        target_size (Optional[tuple]): Intended target size (follows the ``(width, height)`` format).
            If None, the unaltered tensor will be returned.
        rgb (bool): Boolean whether to return an RGB Tensor (default is ``True``).
    
    Returns:
        Tensor of shape ``(height, width, channels)``.

    Examples::

        >> tens = caer.data.guitar()
        >> tens.shape
        (427, 640, 3)
        
    """
    return imread(here+'guitar.jpg', target_size=target_size, rgb=rgb)


def island(target_size: Optional[tuple] = None, rgb: bool = True) -> Tensor:
    r"""
        Returns a standard 640x426 image Tensor (RGB, by default) of an island.

    Args:
        target_size (Optional[tuple]): Intended target size (follows the ``(width, height)`` format).
            If None, the unaltered tensor will be returned.
        rgb (bool): Boolean whether to return an RGB Tensor (default is ``True``).
    
    Returns:
        Tensor of shape ``(height, width, channels)``.

    Examples::

        >> tens = caer.data.island()
        >> tens.shape
        (426, 640, 3)
        
    """
    return imread(here+'island.jpg', target_size=target_size, rgb=rgb)


def jellyfish(target_size: Optional[tuple] = None, rgb: bool = True) -> Tensor:
    r"""
        Returns a standard 640x427 image Tensor (RGB, by default) of a jellyfish.

    Args:
        target_size (Optional[tuple]): Intended target size (follows the ``(width, height)`` format).
            If None, the unaltered tensor will be returned.
        rgb (bool): Boolean whether to return an RGB Tensor (default is ``True``).
    
    Returns:
        Tensor of shape ``(height, width, channels)``.

    Examples::

        >> tens = caer.data.jellyfish()
        >> tens.shape
        (427, 640, 3)
        
    """
    return imread(here+'jellyfish.jpg', target_size=target_size, rgb=rgb)


def laptop(target_size: Optional[tuple] = None, rgb: bool = True) -> Tensor:
    r"""
        Returns a standard 640x427 image Tensor (RGB, by default) of a laptop.

    Args:
        target_size (Optional[tuple]): Intended target size (follows the ``(width, height)`` format).
            If None, the unaltered tensor will be returned.
        rgb (bool): Boolean whether to return an RGB Tensor (default is ``True``).
    
    Returns:
        Tensor of shape ``(height, width, channels)``.

    Examples::

        >> tens = caer.data.laptop()
        >> tens.shape
        (427, 640, 3)
        
    """
    return imread(here+'laptop.jpg', target_size=target_size, rgb=rgb)


def mountain(target_size: Optional[tuple] = None, rgb: bool = True) -> Tensor:
    r"""
        Returns a standard 640x427 image Tensor (RGB, by default) of a mountain.

    Args:
        target_size (Optional[tuple]): Intended target size (follows the ``(width, height)`` format).
            If None, the unaltered tensor will be returned.
        rgb (bool): Boolean whether to return an RGB Tensor (default is ``True``).
    
    Returns:
        Tensor of shape ``(height, width, channels)``.

    Examples::

        >> tens = caer.data.mountain()
        >> tens.shape
        (427, 640, 3)
        
    """
    return imread(here+'mountain.jpg', target_size=target_size, rgb=rgb)


def night(target_size: Optional[tuple] = None, rgb: bool = True) -> Tensor:
    r"""
        Returns a standard 640x427 image Tensor (RGB, by default) of a night landscape.

    Args:
        target_size (Optional[tuple]): Intended target size (follows the ``(width, height)`` format).
            If None, the unaltered tensor will be returned.
        rgb (bool): Boolean whether to return an RGB Tensor (default is ``True``).
    
    Returns:
        Tensor of shape ``(height, width, channels)``.

    Examples::

        >> tens = caer.data.night()
        >> tens.shape
        (427, 640, 3)
        
    """
    return imread(here+'night.jpg', target_size=target_size, rgb=rgb)


def puppies(target_size: Optional[tuple] = None, rgb: bool = True) -> Tensor:
    r"""
        Returns a standard 640x427 image Tensor (RGB, by default) of a litter of puppies.

    Args:
        target_size (Optional[tuple]): Intended target size (follows the ``(width, height)`` format).
            If None, the unaltered tensor will be returned.
        rgb (bool): Boolean whether to return an RGB Tensor (default is ``True``).
    
    Returns:
        Tensor of shape ``(height, width, channels)``.

    Examples::

        >> tens = caer.data.puppies()
        >> tens.shape
        (427, 640, 3)
        
    """
    return imread(here+'puppies.jpg', target_size=target_size, rgb=rgb)


def puppy(target_size: Optional[tuple] = None, rgb: bool = True) -> Tensor:
    r"""
        Returns a standard 640x512 image Tensor (RGB, by default) of a puppy.

    Args:
        target_size (Optional[tuple]): Intended target size (follows the ``(width, height)`` format).
            If None, the unaltered tensor will be returned.
        rgb (bool): Boolean whether to return an RGB Tensor (default is ``True``).
    
    Returns:
        Tensor of shape ``(height, width, channels)``.

    Examples::

        >> tens = caer.data.puppy()
        >> tens.shape
        (512, 640, 3)
        
    """
    return imread(here+'puppy.jpg', target_size=target_size, rgb=rgb)


def red_fish(target_size: Optional[tuple] = None, rgb: bool = True) -> Tensor:
    r"""
        Returns a standard 640x427 image Tensor (RGB, by default) of a red fish.

    Args:
        target_size (Optional[tuple]): Intended target size (follows the ``(width, height)`` format).
            If None, the unaltered tensor will be returned.
        rgb (bool): Boolean whether to return an RGB Tensor (default is ``True``).
    
    Returns:
        Tensor of shape ``(height, width, channels)``.

    Examples::

        >> tens = caer.data.red_fish()
        >> tens.shape
        (427, 640, 3)
        
    """
    return imread(here+'red_fish.jpg', target_size=target_size, rgb=rgb)


def phone(target_size: Optional[tuple] = None, rgb: bool = True) -> Tensor:
    r"""
        Returns a standard 640x427 image Tensor (RGB, by default) of a rotary phone.

    Args:
        target_size (Optional[tuple]): Intended target size (follows the ``(width, height)`` format).
            If None, the unaltered tensor will be returned.
        rgb (bool): Boolean whether to return an RGB Tensor (default is ``True``).
    
    Returns:
        Tensor of shape ``(height, width, channels)``.

    Examples::

        >> tens = caer.data.phone()
        >> tens.shape
        (427, 640, 3)
        
    """
    return imread(here+'rotary_phone.jpg', target_size=target_size, rgb=rgb)


def sea_turtle(target_size: Optional[tuple] = None, rgb: bool = True) -> Tensor:
    r"""
        Returns a standard 640x400 image Tensor (RGB, by default) of a sea turtle.

    Args:
        target_size (Optional[tuple]): Intended target size (follows the ``(width, height)`` format).
            If None, the unaltered tensor will be returned.
        rgb (bool): Boolean whether to return an RGB Tensor (default is ``True``).
    
    Returns:
        Tensor of shape ``(height, width, channels)``.

    Examples::

        >> tens = caer.data.sea_turtle()
        >> tens.shape
        (400, 640, 3)
        
    """
    return imread(here+'sea_turtle.jpg', target_size=target_size, rgb=rgb)


def snow(target_size: Optional[tuple] = None, rgb: bool = True) -> Tensor:
    r"""
        Returns a standard 640x360 image Tensor (RGB, by default) of snow.

    Args:
        target_size (Optional[tuple]): Intended target size (follows the ``(width, height)`` format).
            If None, the unaltered tensor will be returned.
        rgb (bool): Boolean whether to return an RGB Tensor (default is ``True``).
    
    Returns:
        Tensor of shape ``(height, width, channels)``.

    Examples::

        >> tens = caer.data.snow()
        >> tens.shape
        (360, 640, 3)
        
    """
    return imread(here+'snow.jpg', target_size=target_size, rgb=rgb)


def snowflake(target_size: Optional[tuple] = None, rgb: bool = True) -> Tensor:
    r"""
        Returns a standard 640x480 image Tensor (RGB, by default) of a snowflake.

    Args:
        target_size (Optional[tuple]): Intended target size (follows the ``(width, height)`` format).
            If None, the unaltered tensor will be returned.
        rgb (bool): Boolean whether to return an RGB Tensor (default is ``True``).
    
    Returns:
        Tensor of shape ``(height, width, channels)``.

    Examples::

        >> tens = caer.data.snowflake()
        >> tens.shape
        (480, 640, 3)
        
    """
    return imread(here+'snowflake.jpg', target_size=target_size, rgb=rgb)


def sunrise(target_size: Optional[tuple] = None, rgb: bool = True) -> Tensor:
    r"""
        Returns a standard 640x427 image Tensor (RGB, by default) of a sunrise landscape.

    Args:
        target_size (Optional[tuple]): Intended target size (follows the ``(width, height)`` format).
            If None, the unaltered tensor will be returned.
        rgb (bool): Boolean whether to return an RGB Tensor (default is ``True``).
    
    Returns:
        Tensor of shape ``(height, width, channels)``.

    Examples::

        >> tens = caer.data.sunrise()
        >> tens.shape
        (427, 640, 3)
        
    """
    return imread(here+'sunrise.jpg', target_size=target_size, rgb=rgb)


def tent(target_size: Optional[tuple] = None, rgb: bool = True) -> Tensor:
    r"""
        Returns a standard 640x427 image Tensor (RGB, by default) of a tent.

    Args:
        target_size (Optional[tuple]): Intended target size (follows the ``(width, height)`` format).
            If None, the unaltered tensor will be returned.
        rgb (bool): Boolean whether to return an RGB Tensor (default is ``True``).
    
    Returns:
        Tensor of shape ``(height, width, channels)``.

    Examples::

        >> tens = caer.data.tent()
        >> tens.shape
        (427, 640, 3)
        
    """
    return imread(here+'tent.jpg', target_size=target_size, rgb=rgb)


__all__ = [d for d in dir() if not d.startswith('_')]
# __all__ = [
#     'audio_mixer',
#     'bear',
#     'beverages',
#     'black_cat',
#     'blue_tang',
#     'camera',
#     'controller',
#     'drone',
#     'dusk',
#     'fighter_fish',
#     'gold_fish',
#     'green_controller',
#     'green_fish',
#     'guitar',
#     'island',
#     'jellyfish',
#     'laptop',
#     'mountain',
#     'night',
#     'puppies',
#     'puppy',
#     'red_fish',
#     'phone',
#     'sea_turtle',
#     'snow',
#     'sunrise',
#     'tent'
# ]