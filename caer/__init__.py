r""" \
Caer
=====
Caer is a *lightweight, scalable* Computer Vision library for high-performance AI research. It simplifies your approach towards Computer Vision by abstracting away unnecessary boilerplate code giving you the flexibility to quickly prototype deep learning models or research ideas. 

Our design philosophy makes Caer ideal for students, researchers, hobbyists and even experts in the fields of Deep Learning and Computer Vision.

Documentation: https://caer.rtfd.io

Available subpackages
---------------------
augment
    Powerful image augmentation operations.
color
    Colorspace manipulation
data
    Standard high-quality test images
filters
    Advanced Image Filters
path
    OS-specific Path Manipulations
preprocessing
    Image preprocessing utilities
video
    Video processing utilities


Utilities
---------
__version__
    Caer version string

__author__
    Author of Caer

__contributors__
    List of all contributors to the project

__homepage__
    Web URL of the Caer documentation

"""

#pylint:disable=undefined-all-variable, redefined-builtin

from ._meta import (
    version,
    release,
    author,
    author_email,
    contributors,
    homepage
)

r"""
Root Package Info
"""

__version__ = version
__release__ = release 
__author__ = author 
__author_email__ = author_email
__contributors__ = contributors
__license__ = 'MIT License'
__copyright__ = r"""
Copyright (c) 2020-2021 Jason Dsouza <jasmcaus>
All Rights Reserved.
"""
__homepage__ = homepage


def license():
    return __license__ 

def copyright():
    return __copyright__


# IO-stuff
from .io import (
    imread,
    imsave,
    resize,
    smart_resize,
    __all__ as __all_io__
)

# Preprocessing
# Kept for backward compatibility (with my online OpenCV course published on FreeCodeCamp.org)
from .preprocess import (
    preprocess_from_dir,
    sep_train,
    shuffle,
    reshape,
    normalize,
    __all__ as __all_preprocess__
)

# General utilities
from .core import (
    median,
    mean,
    merge,
    split,
    npmean,
    to_array,
    array,
    asarray,
    train_val_split,
    get_classes_from_dir,
    sort_dict,
    get_opencv_version,
    __all__ as __all_core__
)

# Color Spaces
from .color import (
    to_bgr,
    to_rgb,
    to_gray,
    to_hsv,
    to_hls,
    to_lab,

    bgr2rgb,
    bgr2gray,
    bgr2hsv,    
    bgr2hls,
    bgr2lab,

    rgb2bgr,
    rgb2gray,
    rgb2hsv,
    rgb2hls,
    rgb2lab,

    gray2rgb,
    gray2bgr,    
    gray2hsv,
    gray2hls,
    gray2lab,    

    hsv2rgb,
    hsv2bgr,
    hsv2gray,
    hsv2hls,
    hsv2lab,

    hls2rgb,
    hls2bgr,
    hls2gray,
    hls2hsv,
    hls2lab,

    lab2rgb,
    lab2bgr,
    lab2gray,
    lab2hsv,
    lab2hls,

    __all__ as __all_color__
)


# Bringing in configuration variables from globals.py
# from .globals import *

# Tensor-stuff 
from .adorad import *


def get_caer_version():
    return __version__


def get_caer_functions():
    return __all__


def get_caer_methods():
    return __all__


# __all__ configs
from .globals import __all__ as __all_globals__

from .transforms import __all__ as __all_transforms__
from .data import __all__ as __all_data__
# from .filters import __all__ as __all_filters__
from .path import __all__ as __all_path__
from .preprocessing import __all__ as __all_preprocessing__
from .video import __all__ as __all_video__


__all__ = __all_globals__ + __all_core__ + __all_preprocess__

__all__ += __all_transforms__
__all__ += __all_color__
__all__ += __all_data__ 
# __all__ += __all_filters__ 
__all__ += __all_io__ 
__all__ += __all_path__
__all__ += __all_preprocessing__ 
__all__ += __all_video__ 


# Stop polluting the namespace

## Remove root package info
del author
del version 
del release
del contributors 

## Remove everything else 
del __all_globals__ 
del __all_core__ 
del __all_preprocess__ 

del __all_transforms__ 
del __all_color__ 
del __all_data__ 
# del __all_filters__
del __all_preprocessing__ 
del __all_video__ 
