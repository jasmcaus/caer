#
#  _____ _____ _____ _____
# |     |     | ___  | __|  Caer - Modern Computer Vision
# |     | ___ |      | \    Languages: Python, C, C++
# |_____|     | ____ |  \   http://github.com/jasmcaus/caer

# Licensed under the MIT License <http://opensource.org/licenses/MIT>
# SPDX-License-Identifier: MIT
# Copyright (c) 2020 The Caer Authors <http://github.com/jasmcaus>


from ..io import imread 
from ..path import abspath, minijoin
from .._base import __curr__ 


here = minijoin(__curr__, 'data').replace('\\', "/") + "/"


def audio_mixer(target_size=None, rgb=False):
    return imread(here+'audio_mixer.jpg', target_size=target_size, rgb=rgb)


def bear(target_size=None, rgb=False):
    return imread(here+'bear.jpg', target_size=target_size, rgb=rgb)


def beverages(target_size=None, rgb=False):
    return imread(here+'beverages.jpg', target_size=target_size, rgb=rgb)


def black_cat(target_size=None, rgb=False):
    return imread(here+'black_cat.jpg', target_size=target_size, rgb=rgb)


def blue_tang(target_size=None, rgb=False):
    return imread(here+'blue_tang.jpg', target_size=target_size, rgb=rgb)


def camera(target_size=None, rgb=False):
    return imread(here+'camera.jpg', target_size=target_size, rgb=rgb)


def controller(target_size=None, rgb=False):
    return imread(here+'controller.jpg', target_size=target_size, rgb=rgb)


def drone(target_size=None, rgb=False):
    return imread(here+'drone.jpg', target_size=target_size, rgb=rgb)


def dusk(target_size=None, rgb=False):
    return imread(here+'dusk.jpg', target_size=target_size, rgb=rgb)


def fighter_fish(target_size=None, rgb=False):
    return imread(here+'fighter_fish.jpg', target_size=target_size, rgb=rgb)


def gold_fish(target_size=None, rgb=False):
    return imread(here+'gold_fish.jpg', target_size=target_size, rgb=rgb)


def green_controller(target_size=None, rgb=False):
    return imread(here+'green_controller.jpg', target_size=target_size, rgb=rgb)


def green_fish(target_size=None, rgb=False):
    return imread(here+'green_fish.jpg', target_size=target_size, rgb=rgb)


def guitar(target_size=None, rgb=False):
    return imread(here+'guitar.jpg', target_size=target_size, rgb=rgb)


def island(target_size=None, rgb=False):
    return imread(here+'island.jpg', target_size=target_size, rgb=rgb)


def jellyfish(target_size=None, rgb=False):
    return imread(here+'jellyfish.jpg', target_size=target_size, rgb=rgb)


def laptop(target_size=None, rgb=False):
    return imread(here+'laptop.jpg', target_size=target_size, rgb=rgb)


def mountain(target_size=None, rgb=False):
    return imread(here+'mountain.jpg', target_size=target_size, rgb=rgb)


def night(target_size=None, rgb=False):
    return imread(here+'night.jpg', target_size=target_size, rgb=rgb)


def puppies(target_size=None, rgb=False):
    return imread(here+'puppies.jpg', target_size=target_size, rgb=rgb)


def puppy(target_size=None, rgb=False):
    return imread(here+'puppy.jpg', target_size=target_size, rgb=rgb)


def red_fish(target_size=None, rgb=False):
    return imread(here+'red_fish.jpg', target_size=target_size, rgb=rgb)


def phone(target_size=None, rgb=False):
    return imread(here+'rotary_phone.jpg', target_size=target_size, rgb=rgb)


def sea_turtle(target_size=None, rgb=False):
    return imread(here+'sea_turtle.jpg', target_size=target_size, rgb=rgb)


def snow(target_size=None, rgb=False):
    return imread(here+'snow.jpg', target_size=target_size, rgb=rgb)


def snowflake(target_size=None, rgb=False):
    return imread(here+'snowflake.jpg', target_size=target_size, rgb=rgb)


def sunrise(target_size=None, rgb=False):
    return imread(here+'sunrise.jpg', target_size=target_size, rgb=rgb)


def tent(target_size=None, rgb=False):
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
#     'sea_turtle',
#     'sunrise',
#     'tent'
# ]