# Copyright (c) 2020 Jason Dsouza <jasmcaus@gmail.com>
# Protected under the MIT License (see LICENSE)

from ..images import load_img 
from ..paths import abspath, minijoin
from .._base import __curr__ 

here = minijoin(__curr__, 'data').replace('\\', "/") + "/"


def audio_mixer(target_size=None, resize_factor=None,  keep_aspect_ratio=False):
    return load_img(here+'audio_mixer.jpg', target_size=target_size, resize_factor=resize_factor,  keep_aspect_ratio=keep_aspect_ratio)


def bear(target_size=None, resize_factor=None,  keep_aspect_ratio=False):
    return load_img(here+'bear.jpg', target_size=target_size, resize_factor=resize_factor,  keep_aspect_ratio=keep_aspect_ratio)


def beverages(target_size=None, resize_factor=None,  keep_aspect_ratio=False):
    return load_img(here+'beverages.jpg', target_size=target_size, resize_factor=resize_factor,  keep_aspect_ratio=keep_aspect_ratio)


def black_cat(target_size=None, resize_factor=None,  keep_aspect_ratio=False):
    return load_img(here+'black_cat.jpg', target_size=target_size, resize_factor=resize_factor,  keep_aspect_ratio=keep_aspect_ratio)


def blue_tang(target_size=None, resize_factor=None,  keep_aspect_ratio=False):
    return load_img(here+'blue_tang.jpg', target_size=target_size, resize_factor=resize_factor,  keep_aspect_ratio=keep_aspect_ratio)


def camera(target_size=None, resize_factor=None,  keep_aspect_ratio=False):
    return load_img(here+'camera.jpg', target_size=target_size, resize_factor=resize_factor,  keep_aspect_ratio=keep_aspect_ratio)


def controller(target_size=None, resize_factor=None,  keep_aspect_ratio=False):
    return load_img(here+'controller.jpg', target_size=target_size, resize_factor=resize_factor,  keep_aspect_ratio=keep_aspect_ratio)


def drone(target_size=None, resize_factor=None,  keep_aspect_ratio=False):
    return load_img(here+'drone.jpg', target_size=target_size, resize_factor=resize_factor,  keep_aspect_ratio=keep_aspect_ratio)


def fighter_fish(target_size=None, resize_factor=None,  keep_aspect_ratio=False):
    return load_img(here+'fighter_fish.jpg', target_size=target_size, resize_factor=resize_factor,  keep_aspect_ratio=keep_aspect_ratio)


def gold_fish(target_size=None, resize_factor=None,  keep_aspect_ratio=False):
    return load_img(here+'gold_fish.jpg', target_size=target_size, resize_factor=resize_factor,  keep_aspect_ratio=keep_aspect_ratio)


def green_controller(target_size=None, resize_factor=None,  keep_aspect_ratio=False):
    return load_img(here+'green_controller.jpg', target_size=target_size, resize_factor=resize_factor,  keep_aspect_ratio=keep_aspect_ratio)


def green_fish(target_size=None, resize_factor=None,  keep_aspect_ratio=False):
    return load_img(here+'green_fish.jpg', target_size=target_size, resize_factor=resize_factor,  keep_aspect_ratio=keep_aspect_ratio)


def guitar(target_size=None, resize_factor=None,  keep_aspect_ratio=False):
    return load_img(here+'guitar.jpg', target_size=target_size, resize_factor=resize_factor,  keep_aspect_ratio=keep_aspect_ratio)


def island(target_size=None, resize_factor=None,  keep_aspect_ratio=False):
    return load_img(here+'island.jpg', target_size=target_size, resize_factor=resize_factor,  keep_aspect_ratio=keep_aspect_ratio)


def jellyfish(target_size=None, resize_factor=None,  keep_aspect_ratio=False):
    return load_img(here+'jellyfish.jpg', target_size=target_size, resize_factor=resize_factor,  keep_aspect_ratio=keep_aspect_ratio)


def laptop(target_size=None, resize_factor=None,  keep_aspect_ratio=False):
    return load_img(here+'laptop.jpg', target_size=target_size, resize_factor=resize_factor,  keep_aspect_ratio=keep_aspect_ratio)


def mountain(target_size=None, resize_factor=None,  keep_aspect_ratio=False):
    return load_img(here+'mountain.jpg', target_size=target_size, resize_factor=resize_factor,  keep_aspect_ratio=keep_aspect_ratio)


def puppies(target_size=None, resize_factor=None,  keep_aspect_ratio=False):
    return load_img(here+'puppies.jpg', target_size=target_size, resize_factor=resize_factor,  keep_aspect_ratio=keep_aspect_ratio)


def puppy(target_size=None, resize_factor=None,  keep_aspect_ratio=False):
    return load_img(here+'puppy.jpg', target_size=target_size, resize_factor=resize_factor,  keep_aspect_ratio=keep_aspect_ratio)


def red_fish(target_size=None, resize_factor=None,  keep_aspect_ratio=False):
    return load_img(here+'red_fish.jpg', target_size=target_size, resize_factor=resize_factor,  keep_aspect_ratio=keep_aspect_ratio)


def phone(target_size=None, resize_factor=None,  keep_aspect_ratio=False):
    return load_img(here+'rotary_phone.jpg', target_size=target_size, resize_factor=resize_factor,  keep_aspect_ratio=keep_aspect_ratio)


def sea_turtle(target_size=None, resize_factor=None,  keep_aspect_ratio=False):
    return load_img(here+'sea_turtle.jpg', target_size=target_size, resize_factor=resize_factor,  keep_aspect_ratio=keep_aspect_ratio)


def snow(target_size=None, resize_factor=None,  keep_aspect_ratio=False):
    return load_img(here+'snow.jpg', target_size=target_size, resize_factor=resize_factor,  keep_aspect_ratio=keep_aspect_ratio)

def snowflake(target_size=None, resize_factor=None,  keep_aspect_ratio=False):
    return load_img(here+'snowflake.jpg', target_size=target_size, resize_factor=resize_factor,  keep_aspect_ratio=keep_aspect_ratio)


def sunrise(target_size=None, resize_factor=None,  keep_aspect_ratio=False):
    return load_img(here+'sunrise.jpg', target_size=target_size, resize_factor=resize_factor,  keep_aspect_ratio=keep_aspect_ratio)


def tent(target_size=None, resize_factor=None,  keep_aspect_ratio=False):
    return load_img(here+'tent.jpg', target_size=target_size, resize_factor=resize_factor,  keep_aspect_ratio=keep_aspect_ratio)



__all__ = (
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
    'snow_turtle',
    'sunrise',
    'tent'
)