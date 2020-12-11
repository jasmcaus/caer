#    _____           ______  _____ 
#  / ____/    /\    |  ____ |  __ \
# | |        /  \   | |__   | |__) | Caer - Modern Computer Vision
# | |       / /\ \  |  __|  |  _  /  Languages: Python, C, C++
# | |___   / ____ \ | |____ | | \ \  http://github.com/jasmcaus/caer
#  \_____\/_/    \_ \______ |_|  \_\

# Licensed under the MIT License <http://opensource.org/licenses/MIT>
# SPDX-License-Identifier: MIT
# Copyright (c) 2020 The Caer Authors <http://github.com/jasmcaus>

from ..io import imread 
from ..path import abspath, minijoin
from .._base import __curr__ 
import numpy as np 
Tensor = np.ndarray 

here = minijoin(__curr__, 'data').replace('\\', "/") + "/"

def _get_path_to_data(name) -> str:
    return minijoin(here, name)


def audio_mixer(target_size=None, rgb=True, gray=False) -> Tensor:
    r"""
        Returns a standard 640x427 image (RGB, by default) of an audio mixer.

    Args:
        target_size (tuple): Intended target size (follows the ``(width,height)`` format)
        rgb (bool): Return an RGB image?
    
    Returns:
        Image array of shape ``(height, width, channels)``

    Examples::

        >> img = caer.data.audio_mixer()
        >> img.shape
        (427, 640, 3)

    """
    return imread(here+'audio_mixer.jpg', target_size=target_size, rgb=rgb, gray=gray)


def bear(target_size=None, rgb=True, gray=False) -> Tensor:
    r"""
        Returns a standard 640x427 image (RGB, by default) of a bear.

    Args:
        target_size (tuple): Intended target size (follows the ``(width,height)`` format)
        rgb (bool): Return an RGB image?
    
    Returns:
        Image array of shape ``(height, width, channels)``

    Examples::

        >> img = caer.data.bear()
        >> img.shape
        (427, 640, 3)

    """
    return imread(here+'bear.jpg', target_size=target_size, rgb=rgb, gray=gray)


def beverages(target_size=None, rgb=True, gray=False) -> Tensor:
    r"""
        Returns a standard 640x427 image (RGB, by default) of beverages.

    Args:
        target_size (tuple): Intended target size (follows the ``(width,height)`` format)
        rgb (bool): Return an RGB image?
    
    Returns:
        Image array of shape ``(height, width, channels)``

    Examples::

        >> img = caer.data.beverages()
        >> img.shape
        (427, 640, 3)
        
    """
    return imread(here+'beverages.jpg', target_size=target_size, rgb=rgb, gray=gray)


def black_cat(target_size=None, rgb=True, gray=False) -> Tensor:
    r"""
        Returns a standard 640x427 image (RGB, by default) of a black cat.

    Args:
        target_size (tuple): Intended target size (follows the ``(width,height)`` format)
        rgb (bool): Return an RGB image?
    
    Returns:
        Image array of shape ``(height, width, channels)``

    Examples::

        >> img = caer.data.black_cat()
        >> img.shape
        (427, 640, 3)
        
    """
    return imread(here+'black_cat.jpg', target_size=target_size, rgb=rgb, gray=gray)


def blue_tang(target_size=None, rgb=True, gray=False) -> Tensor:
    r"""
        Returns a standard 640x414 image (RGB, by default) of a blue tang (a type of fish).

    Args:
        target_size (tuple): Intended target size (follows the ``(width,height)`` format)
        rgb (bool): Return an RGB image?
    
    Returns:
        Image array of shape ``(height, width, channels)``

    Examples::

        >> img = caer.data.audio_mixer()
        >> img.shape
        (414, 640, 3)
        
    """
    return imread(here+'blue_tang.jpg', target_size=target_size, rgb=rgb, gray=gray)


def camera(target_size=None, rgb=True, gray=False) -> Tensor:
    r"""
        Returns a standard 640x427 image (RGB, by default) of a camera.

    Args:
        target_size (tuple): Intended target size (follows the ``(width,height)`` format)
        rgb (bool): Return an RGB image?
    
    Returns:
        Image array of shape ``(height, width, channels)``

    Examples::

        >> img = caer.data.camera()
        >> img.shape
        (427, 640, 3)
        
    """
    return imread(here+'camera.jpg', target_size=target_size, rgb=rgb, gray=gray)


def controller(target_size=None, rgb=True, gray=False) -> Tensor:
    r"""
        Returns a standard 640x427 image (RGB, by default) of a game controller.

    Args:
        target_size (tuple): Intended target size (follows the ``(width,height)`` format)
        rgb (bool): Return an RGB image?
    
    Returns:
        Image array of shape ``(height, width, channels)``

    Examples::

        >> img = caer.data.controller()
        >> img.shape
        (427, 640, 3)
        
    """
    return imread(here+'controller.jpg', target_size=target_size, rgb=rgb, gray=gray)


def drone(target_size=None, rgb=True, gray=False) -> Tensor:
    r"""
        Returns a standard 640x358 image (RGB, by default) of a robotic drone.

    Args:
        target_size (tuple): Intended target size (follows the ``(width,height)`` format)
        rgb (bool): Return an RGB image?
    
    Returns:
        Image array of shape ``(height, width, channels)``

    Examples::

        >> img = caer.data.drone()
        >> img.shape
        (358, 640, 3)
        
    """
    return imread(here+'drone.jpg', target_size=target_size, rgb=rgb, gray=gray)


def dusk(target_size=None, rgb=True, gray=False) -> Tensor:
    r"""
        Returns a standard 640x427 image (RGB, by default) of a dusk landscape.

    Args:
        target_size (tuple): Intended target size (follows the ``(width,height)`` format)
        rgb (bool): Return an RGB image?
    
    Returns:
        Image array of shape ``(height, width, channels)``

    Examples::

        >> img = caer.data.dusk()
        >> img.shape
        (427, 640, 3)
        
    """
    return imread(here+'dusk.jpg', target_size=target_size, rgb=rgb, gray=gray)


def fighter_fish(target_size=None, rgb=True, gray=False) -> Tensor:
    r"""
        Returns a standard 640x640 image (RGB, by default) of a fighter fish.

    Args:
        target_size (tuple): Intended target size (follows the ``(width,height)`` format)
        rgb (bool): Return an RGB image?
    
    Returns:
        Image array of shape ``(height, width, channels)``

    Examples::

        >> img = caer.data.fighter_fish()
        >> img.shape
        (640, 640, 3)
        
    """
    return imread(here+'fighter_fish.jpg', target_size=target_size, rgb=rgb, gray=gray)


def gold_fish(target_size=None, rgb=True, gray=False) -> Tensor:
    r"""
        Returns a standard 640x901 image (RGB, by default) of a gold fish.

    Args:
        target_size (tuple): Intended target size (follows the ``(width,height)`` format)
        rgb (bool): Return an RGB image?
    
    Returns:
        Image array of shape ``(height, width, channels)``

    Examples::

        >> img = caer.data.gold_fish()
        >> img.shape
        (901, 640, 3)
        
    """
    return imread(here+'gold_fish.jpg', target_size=target_size, rgb=rgb, gray=gray)


def green_controller(target_size=None, rgb=True, gray=False) -> Tensor:
    r"""
        Returns a standard 640x512 image (RGB, by default) of a green game controller.

    Args:
        target_size (tuple): Intended target size (follows the ``(width,height)`` format)
        rgb (bool): Return an RGB image?
    
    Returns:
        Image array of shape ``(height, width, channels)``

    Examples::

        >> img = caer.data.green_controller()
        >> img.shape
        (512, 640, 3)
        
    """
    return imread(here+'green_controller.jpg', target_size=target_size, rgb=rgb, gray=gray)


def green_fish(target_size=None, rgb=True, gray=False) -> Tensor:
    r"""
        Returns a standard 640x430 image (RGB, by default) of a green fish.

    Args:
        target_size (tuple): Intended target size (follows the ``(width,height)`` format)
        rgb (bool): Return an RGB image?
    
    Returns:
        Image array of shape ``(height, width, channels)``

    Examples::

        >> img = caer.data.green_fish()
        >> img.shape
        (430, 640, 3)
        
    """
    return imread(here+'green_fish.jpg', target_size=target_size, rgb=rgb, gray=gray)


def guitar(target_size=None, rgb=True, gray=False) -> Tensor:
    r"""
        Returns a standard 640x427 image (RGB, by default) of a guitar.

    Args:
        target_size (tuple): Intended target size (follows the ``(width,height)`` format)
        rgb (bool): Return an RGB image?
    
    Returns:
        Image array of shape ``(height, width, channels)``

    Examples::

        >> img = caer.data.guitar()
        >> img.shape
        (427, 640, 3)
        
    """
    return imread(here+'guitar.jpg', target_size=target_size, rgb=rgb, gray=gray)


def island(target_size=None, rgb=True, gray=False) -> Tensor:
    r"""
        Returns a standard 640x426 image (RGB, by default) of an island.

    Args:
        target_size (tuple): Intended target size (follows the ``(width,height)`` format)
        rgb (bool): Return an RGB image?
    
    Returns:
        Image array of shape ``(height, width, channels)``

    Examples::

        >> img = caer.data.island()
        >> img.shape
        (426, 640, 3)
        
    """
    return imread(here+'island.jpg', target_size=target_size, rgb=rgb, gray=gray)


def jellyfish(target_size=None, rgb=True, gray=False) -> Tensor:
    r"""
        Returns a standard 640x427 image (RGB, by default) of a jellyfish.

    Args:
        target_size (tuple): Intended target size (follows the ``(width,height)`` format)
        rgb (bool): Return an RGB image?
    
    Returns:
        Image array of shape ``(height, width, channels)``

    Examples::

        >> img = caer.data.jellyfish()
        >> img.shape
        (427, 640, 3)
        
    """
    return imread(here+'jellyfish.jpg', target_size=target_size, rgb=rgb, gray=gray)


def laptop(target_size=None, rgb=True, gray=False) -> Tensor:
    r"""
        Returns a standard 640x427 image (RGB, by default) of a laptop.

    Args:
        target_size (tuple): Intended target size (follows the ``(width,height)`` format)
        rgb (bool): Return an RGB image?
    
    Returns:
        Image array of shape ``(height, width, channels)``

    Examples::

        >> img = caer.data.laptop()
        >> img.shape
        (427, 640, 3)
        
    """
    return imread(here+'laptop.jpg', target_size=target_size, rgb=rgb, gray=gray)


def mountain(target_size=None, rgb=True, gray=False) -> Tensor:
    r"""
        Returns a standard 640x427 image (RGB, by default) of a mountain.

    Args:
        target_size (tuple): Intended target size (follows the ``(width,height)`` format)
        rgb (bool): Return an RGB image?
    
    Returns:
        Image array of shape ``(height, width, channels)``

    Examples::

        >> img = caer.data.mountain()
        >> img.shape
        (427, 640, 3)
        
    """
    return imread(here+'mountain.jpg', target_size=target_size, rgb=rgb, gray=gray)


def night(target_size=None, rgb=True, gray=False) -> Tensor:
    r"""
        Returns a standard 640x427 image (RGB, by default) of a night landscape.

    Args:
        target_size (tuple): Intended target size (follows the ``(width,height)`` format)
        rgb (bool): Return an RGB image?
    
    Returns:
        Image array of shape ``(height, width, channels)``

    Examples::

        >> img = caer.data.night()
        >> img.shape
        (427, 640, 3)
        
    """
    return imread(here+'night.jpg', target_size=target_size, rgb=rgb, gray=gray)


def puppies(target_size=None, rgb=True, gray=False) -> Tensor:
    r"""
        Returns a standard 640x427 image (RGB, by default) of a litter of puppies.

    Args:
        target_size (tuple): Intended target size (follows the ``(width,height)`` format)
        rgb (bool): Return an RGB image?
    
    Returns:
        Image array of shape ``(height, width, channels)``

    Examples::

        >> img = caer.data.puppies()
        >> img.shape
        (427, 640, 3)
        
    """
    return imread(here+'puppies.jpg', target_size=target_size, rgb=rgb, gray=gray)


def puppy(target_size=None, rgb=True, gray=False) -> Tensor:
    r"""
        Returns a standard 640x512 image (RGB, by default) of a puppy.

    Args:
        target_size (tuple): Intended target size (follows the ``(width,height)`` format)
        rgb (bool): Return an RGB image?
    
    Returns:
        Image array of shape ``(height, width, channels)``

    Examples::

        >> img = caer.data.puppy()
        >> img.shape
        (512, 640, 3)
        
    """
    return imread(here+'puppy.jpg', target_size=target_size, rgb=rgb, gray=gray)


def red_fish(target_size=None, rgb=True, gray=False) -> Tensor:
    r"""
        Returns a standard 640x427 image (RGB, by default) of a red fish.

    Args:
        target_size (tuple): Intended target size (follows the ``(width,height)`` format)
        rgb (bool): Return an RGB image?
    
    Returns:
        Image array of shape ``(height, width, channels)``

    Examples::

        >> img = caer.data.red_fish()
        >> img.shape
        (427, 640, 3)
        
    """
    return imread(here+'red_fish.jpg', target_size=target_size, rgb=rgb, gray=gray)


def phone(target_size=None, rgb=True, gray=False) -> Tensor:
    r"""
        Returns a standard 640x427 image (RGB, by default) of a rotary phone.

    Args:
        target_size (tuple): Intended target size (follows the ``(width,height)`` format)
        rgb (bool): Return an RGB image?
    
    Returns:
        Image array of shape ``(height, width, channels)``

    Examples::

        >> img = caer.data.phone()
        >> img.shape
        (427, 640, 3)
        
    """
    return imread(here+'rotary_phone.jpg', target_size=target_size, rgb=rgb, gray=gray)


def sea_turtle(target_size=None, rgb=True, gray=False) -> Tensor:
    r"""
        Returns a standard 640x400 image (RGB, by default) of a sea turtle.

    Args:
        target_size (tuple): Intended target size (follows the ``(width,height)`` format)
        rgb (bool): Return an RGB image?
    
    Returns:
        Image array of shape ``(height, width, channels)``

    Examples::

        >> img = caer.data.sea_turtle()
        >> img.shape
        (400, 640, 3)
        
    """
    return imread(here+'sea_turtle.jpg', target_size=target_size, rgb=rgb, gray=gray)


def snow(target_size=None, rgb=True, gray=False) -> Tensor:
    r"""
        Returns a standard 640x360 image (RGB, by default) of snow.

    Args:
        target_size (tuple): Intended target size (follows the ``(width,height)`` format)
        rgb (bool): Return an RGB image?
    
    Returns:
        Image array of shape ``(height, width, channels)``

    Examples::

        >> img = caer.data.snow()
        >> img.shape
        (360, 640, 3)
        
    """
    return imread(here+'snow.jpg', target_size=target_size, rgb=rgb, gray=gray)


def snowflake(target_size=None, rgb=True, gray=False) -> Tensor:
    r"""
        Returns a standard 640x480 image (RGB, by default) of a snowflake.

    Args:
        target_size (tuple): Intended target size (follows the ``(width,height)`` format)
        rgb (bool): Return an RGB image?
    
    Returns:
        Image array of shape ``(height, width, channels)``

    Examples::

        >> img = caer.data.snowflake()
        >> img.shape
        (480, 640, 3)
        
    """
    return imread(here+'snowflake.jpg', target_size=target_size, rgb=rgb, gray=gray)


def sunrise(target_size=None, rgb=True, gray=False) -> Tensor:
    r"""
        Returns a standard 640x427 image (RGB, by default) of a sunrise landscape.

    Args:
        target_size (tuple): Intended target size (follows the ``(width,height)`` format)
        rgb (bool): Return an RGB image?
    
    Returns:
        Image array of shape ``(height, width, channels)``

    Examples::

        >> img = caer.data.sunrise()
        >> img.shape
        (427, 640, 3)
        
    """
    return imread(here+'sunrise.jpg', target_size=target_size, rgb=rgb, gray=gray)


def tent(target_size=None, rgb=True, gray=False) -> Tensor:
    r"""
        Returns a standard 640x427 image (RGB, by default) of a tent.

    Args:
        target_size (tuple): Intended target size (follows the ``(width,height)`` format)
        rgb (bool): Return an RGB image?
    
    Returns:
        Image array of shape ``(height, width, channels)``

    Examples::

        >> img = caer.data.tent()
        >> img.shape
        (427, 640, 3)
        
    """
    return imread(here+'tent.jpg', target_size=target_size, rgb=rgb, gray=gray)


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