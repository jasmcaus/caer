# Copyright (c) 2020 Jason Dsouza <jasmcaus@gmail.com>
# Protected under the MIT License (see LICENSE)

from ..images import load_img 
from ..paths import abspath, minijoin
from .._base import __curr__ 

here = minijoin(__curr__, 'data').replace('\\', "/") + "/"


def bird(target_size=None):
    return load_img(here+'bird.jpg', target_size=target_size)


def blue_siamese(target_size=None):
    return load_img('blue_siamese.jpg', target_size=target_size)


def camera(target_size=None):
    return load_img('camera.jpg', target_size=target_size)


def candles(target_size=None):
    return load_img('candles.jpg', target_size=target_size)


def coast(target_size=None):
    return load_img('coast.jpg', target_size=target_size)


def controller(target_size=None):
    return load_img('controller.jpg', target_size=target_size)


def cycle(target_size=None):
    return load_img('cycle.jpg', target_size=target_size)


def fish(target_size=None):
    return load_img('fish.jpg', target_size=target_size)


def gold_fish(target_size=None):
    return load_img('gold_fish.jpg', target_size=target_size)


def jellyfish(target_size=None):
    return load_img('jellyfish.jpg', target_size=target_size)


def keyboard(target_size=None):
    return load_img('keyboard.jpg', target_size=target_size)


def lantern(target_size=None):
    return load_img('lantern.jpg', target_size=target_size)


def laptop(target_size=None):
    return load_img('laptop.jpg', target_size=target_size)


def lighthouse(target_size=None):
    return load_img('lighthouse.jpg', target_size=target_size)


def lights(target_size=None):
    return load_img('lights.jpg', target_size=target_size)


def maple(target_size=None):
    return load_img('maple.jpg', target_size=target_size)


def mountain(target_size=None):
    return load_img('mountain.jpg', target_size=target_size)


def night_tent(target_size=None):
    return load_img('night_tent.jpg', target_size=target_size)


def night(target_size=None):
    return load_img('night.jpg', target_size=target_size)


def phone(target_size=None):
    return load_img('phone.jpg', target_size=target_size)


def red_fish(target_size=None):
    return load_img('red_fish.jpg', target_size=target_size)


def sea(target_size=None):
    return load_img('sea.jpg', target_size=target_size)


def snow(target_size=None):
    return load_img('snow.jpg', target_size=target_size)


def squirrel(target_size=None):
    return load_img('snow.jpg', target_size=target_size)


def statue(target_size=None):
    return load_img('statue.jpg', target_size=target_size)


def tent(target_size=None):
    return load_img('tent.jpg', target_size=target_size)


def venice(target_size=None):
    return load_img('venice.jpg', target_size=target_size)


def whale(target_size=None):
    return load_img('whale.jpg', target_size=target_size)

__all__ = (
    'bird',
    'blue_siamese',
    'camera',
    'candles',
    'coast',
    'controller',
    'cycle',
    'fish',
    'gold_fish',
    'jellyfish',
    'keyboard',
    'lantern',
    'laptop',
    'lighthouse',
    'lights',
    'maple',
    'mountain',
    'night_tent',
    'night',
    'phone',
    'red_fish',
    'sea',
    'snow',
    'squirrel',
    'statue',
    'tent',
    'venice',
    'whale'
)