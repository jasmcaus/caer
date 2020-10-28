# Copyright (c) 2020 Jason Dsouza <jasmcaus@gmail.com>
# Protected under the MIT License (see LICENSE)

from ..images import load_img 
from ..paths import abspath, minijoin
from .._base import __curr__ 

here = minijoin(__curr__, 'data').replace('\\', "/") + "/"


def audio_mixer(target_size=None):
    return load_img('audio_mixer.jpg', target_size=target_size)


def bear(target_size=None):
    return load_img('bear.jpg', target_size=target_size)


def beverages(target_size=None):
    return load_img('beverages.jpg', target_size=target_size)


def black_cat(target_size=None):
    return load_img('black_cat.jpg', target_size=target_size)


def blue_tang(target_size=None):
    return load_img('blue_tang.jpg', target_size=target_size)


def camera(target_size=None):
    return load_img('camera.jpg', target_size=target_size)


def controller(target_size=None):
    return load_img('controller.jpg', target_size=target_size)


def drone(target_size=None):
    return load_img('drone.jpg', target_size=target_size)


def fighter_fish(target_size=None):
    return load_img('fighter_fish.jpg', target_size=target_size)


def gold_fish(target_size=None):
    return load_img('gold_fish.jpg', target_size=target_size)


def green_controller(target_size=None):
    return load_img('green_controller.jpg', target_size=target_size)


def green_fish(target_size=None):
    return load_img('green_fish.jpg', target_size=target_size)


def guitar(target_size=None):
    return load_img('guitar.jpg', target_size=target_size)


def island(target_size=None):
    return load_img('island.jpg', target_size=target_size)


def jellyfish(target_size=None):
    return load_img('jellyfish.jpg', target_size=target_size)


def laptop(target_size=None):
    return load_img('laptop.jpg', target_size=target_size)


def mountain(target_size=None):
    return load_img('mountain.jpg', target_size=target_size)


def puppies(target_size=None):
    return load_img('puppies.jpg', target_size=target_size)


def puppy(target_size=None):
    return load_img('puppy.jpg', target_size=target_size)


def red_fish(target_size=None):
    return load_img('red_fish.jpg', target_size=target_size)


def phone(target_size=None):
    return load_img('rotary_phone.jpg', target_size=target_size)


def sea_turtle(target_size=None):
    return load_img('sea_turtle.jpg', target_size=target_size)


def snow(target_size=None):
    return load_img('snow.jpg', target_size=target_size)

def snowflake(target_size=None):
    return load_img('snowflake.jpg', target_size=target_size)


def sunrise(target_size=None):
    return load_img('sunrise.jpg', target_size=target_size)


def tent(target_size=None):
    return load_img('tent.jpg', target_size=target_size)



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