""" \
Caer
=====
A Computer Vision library in Python with powerful image and video processing operations.
Caer is a set of utility functions designed to help speed up your Computer Vision workflow. Functions inside `caer` will help reduce the number of calculation calls your code makes, ultimately making it neat, concise and readable.

Documentation: https://caer.rtfd.io

Available subpackages
---------------------
data
    Standard high-quality test images
filters
    Advanced Image Filters
path
    OS-specific Path Manipulations
preprocessing
    Image preprocessing utilities
utils
    General utilities
video
    Video processing utilities


Utilities
---------
__version__
    Caer version string

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

"""
Root Package Info
"""

__version__ = version
__release__ = release 
__author__ = author 
__author_email__ = author_email
__contributors__ = contributors
__license__ = 'MIT License'
__copyright__ = """
Copyright (c) 2020 Jason Dsouza <jasmcaus>
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
    bgr_to_gray,
    bgr_to_hsv,
    bgr_to_lab,
    bgr_to_rgb,
    rgb_to_gray,
    rgb_to_hsv,
    rgb_to_lab,
    rgb_to_bgr,
    hsv_to_bgr,
    hsv_to_gray,
    hsv_to_lab,
    hsv_to_rgb,
    lab_to_bgr,
    lab_to_gray,
    lab_to_hsv,
    lab_to_rgb,
    is_rgb_image,
    is_bgr_image,
    is_hsv_image,
    is_lab_image,
    __all__ as __all_color__
)

# Bringing in configuration variables from globals.py
from .globals import *


def get_caer_version():
    return __version__


def get_caer_functions():
    return __all__


def get_caer_methods():
    return __all__


# __all__ configs
from .globals import __all__ as __all_globals__

from .video import __all__ as __all_video__
from .preprocessing import __all__ as __all_preprocessing__
from .data import __all__ as __all_data__
from .utils import __all__ as __all_utils__
from .path import __all__ as __all_path__
from .filters import __all__ as __all_filters__
from .augment import __all__ as __all_augment__


__all__ = __all_globals__ + __all_core__ + __all_preprocess__ + __all_filters__

__all__ += __all_io__ 
__all__ += __all_preprocessing__ 
__all__ += __all_video__ 
__all__ += __all_data__ 
__all__ += __all_utils__ 
__all__ += __all_path__
__all__ += __all_color__
__all__ += __all_augment__


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
del __all_visualizations__

del __all_augment__ 
del __all_color__ 
del __all_data__ 
del __all_filters__
del __all_preprocessing__ 
del __all_utils__ 
del __all_video__ 
