# Copyright 2020 The Caer Authors. All Rights Reserved.
#
# Licensed under the MIT License (see LICENSE);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at <https://opensource.org/licenses/MIT>
#
# ==============================================================================

from ..images import load_img 
from ..path import abspath, minijoin
from .._base import __curr__ 


here = minijoin(__curr__, 'data').replace('\\', "/") + "/"


def audio_mixer(target_size=None, rgb=True):
    return load_img(here + 'audio_mixer.jpg', target_size=target_size, rgb=rgb)


def bear(target_size=None, rgb=True):
    return load_img(here + 'bear.jpg', target_size=target_size, rgb=rgb)


def beverages(target_size=None, rgb=True):
    return load_img(here + 'beverages.jpg', target_size=target_size, rgb=rgb)


def black_cat(target_size=None, rgb=True):
    return load_img(here + 'black_cat.jpg', target_size=target_size, rgb=rgb)


def blue_tang(target_size=None, rgb=True):
    return load_img(here + 'blue_tang.jpg', target_size=target_size, rgb=rgb)


def camera(target_size=None, rgb=True):
    return load_img(here + 'camera.jpg', target_size=target_size, rgb=rgb)


def controller(target_size=None, rgb=True):
    return load_img(here + 'controller.jpg', target_size=target_size, rgb=rgb)


def drone(target_size=None, rgb=True):
    return load_img(here + 'drone.jpg', target_size=target_size, rgb=rgb)


def dusk(target_size=None, rgb=True):
    return load_img(here + 'dusk.jpg', target_size=target_size, rgb=rgb)


def fighter_fish(target_size=None, rgb=True):
    return load_img(here + 'fighter_fish.jpg', target_size=target_size, rgb=rgb)


def gold_fish(target_size=None, rgb=True):
    return load_img(here + 'gold_fish.jpg', target_size=target_size, rgb=rgb)


def green_controller(target_size=None, rgb=True):
    return load_img(here + 'green_controller.jpg', target_size=target_size, rgb=rgb)


def green_fish(target_size=None, rgb=True):
    return load_img(here + 'green_fish.jpg', target_size=target_size, rgb=rgb)


def guitar(target_size=None, rgb=True):
    return load_img(here + 'guitar.jpg', target_size=target_size, rgb=rgb)


def island(target_size=None, rgb=True):
    return load_img(here + 'island.jpg', target_size=target_size, rgb=rgb)


def jellyfish(target_size=None, rgb=True):
    return load_img(here + 'jellyfish.jpg', target_size=target_size, rgb=rgb)


def laptop(target_size=None, rgb=True):
    return load_img(here + 'laptop.jpg', target_size=target_size, rgb=rgb)


def mountain(target_size=None, rgb=True):
    return load_img(here + 'mountain.jpg', target_size=target_size, rgb=rgb)


def night(target_size=None, rgb=True):
    return load_img(here + 'night.jpg', target_size=target_size, rgb=rgb)


def puppies(target_size=None, rgb=True):
    return load_img(here + 'puppies.jpg', target_size=target_size, rgb=rgb)


def puppy(target_size=None, rgb=True):
    return load_img(here + 'puppy.jpg', target_size=target_size, rgb=rgb)


def red_fish(target_size=None, rgb=True):
    return load_img(here+'red_fish.jpg', target_size=target_size, rgb=rgb)


def phone(target_size=None, rgb=True):
    return load_img(here+'rotary_phone.jpg', target_size=target_size, rgb=rgb)


def sea_turtle(target_size=None, rgb=True):
    return load_img(here+'sea_turtle.jpg', target_size=target_size, rgb=rgb)


def snow(target_size=None, rgb=True):
    return load_img(here+'snow.jpg', target_size=target_size, rgb=rgb)


def snowflake(target_size=None, rgb=True):
    return load_img(here+'snowflake.jpg', target_size=target_size, rgb=rgb)


def sunrise(target_size=None, rgb=True):
    return load_img(here+'sunrise.jpg', target_size=target_size, rgb=rgb)


def tent(target_size=None, rgb=True):
    return load_img(here+'tent.jpg', target_size=target_size, rgb=rgb)



__all__ = [
    'audio_mixer',
    'bear',
    'beverages',
    'black_cat',
    'blue_tang',
    'camera',
    'controller',
    'drone',
    'dusk',
    'fighter_fish',
    'gold_fish',
    'green_controller',
    'green_fish',
    'guitar',
    'island',
    'jellyfish',
    'laptop',
    'mountain',
    'night',
    'puppies',
    'puppy',
    'red_fish',
    'phone',
    'sea_turtle',
    'snow',
    'sea_turtle',
    'sunrise',
    'tent'
]