""" \
Caer
=====
A Computer Vision library in Python with powerful image and video processing operations.
Caer is a set of utility functions designed to help speed up your Computer Vision workflow. Functions inside `caer` will help reduce the number of calculation calls your code makes, ultimately making it neat, concise and readable.

Documentation: https://github.com/jasmcaus/caer/tree/dev/docs

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

from ._meta import version as __version__
from ._meta import author as __author__
from ._meta import release as __release__
from ._meta import contributors as __contributors__
__license__ = 'MIT License'
__copyright__ = """
Copyright (c) 2020 Jason Dsouza <jasmcaus>
All Rights Reserved.
"""

def license():
    return __license__ 

def copyright():
    return __copyright__


# Preprocessing
from .preprocess import preprocess_from_dir
from .preprocess import sep_train
from .preprocess import shuffle
from .preprocess import reshape
from .preprocess import normalize

# General utilities
from .utilities import saveNumpy
from .utilities import median
from .utilities import npmean
from .utilities import to_array
from .utilities import array
from .utilities import asarray
from .utilities import npload
from .utilities import train_val_split
from .utilities import get_classes_from_dir
from .utilities import sort_dict

# Opencv-specific methods
from .opencv import get_opencv_version
from .opencv import mean 
from .opencv import merge 
from .opencv import split 
from .opencv import url_to_image 
# from .opencv import energy_map 
# from .opencv import color_map 
from .opencv import translate
from .opencv import rotate 
from .opencv import edges 

# General visualizations
from .visualizations import hex_to_rgb
from .visualizations import draw_rectangle


# Resize
from .resize import resize 
from .resize import center_crop 

# Image 
from .io import imread
from .io import imsave

# Color Space
from .color import (
    bgr_to_gray,
    bgr_to_hsv,
    bgr_to_lab,
    bgr_to_rgb,
    rgb_to_gray,
    rgb_to_hsv,
    rgb_to_lab,
    rgb_to_bgr,
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
from .io import __all__ as __all_io__
from .resize import __all__ as __all_resize__
from .opencv import __all__ as __all_opencv__
from .preprocess import __all__ as __all_preprocess__
from .utilities import __all__ as __all_utilities__
from .visualizations import __all__ as __all_visualizations__

from .video import __all__ as __all_video__
from .preprocessing import __all__ as __all_preprocessing__
from .data import __all__ as __all_data__
from .utils import __all__ as __all_utils__
from .path import __all__ as __all_path__
from .color import __all__ as __all_color__
# from .filters import __all__ as __all_filters__
# from .distance import __all__ as __all_distance__
# from .morph import __all__ as __all_morph__


__all__ = __all_globals__ + __all_io__ + __all_resize__ + __all_opencv__ + __all_preprocess__ + __all_utilities__ + __all_visualizations__
# + __all_filters__ + __all_distance__ + __all_morph__

__all__ += __all_preprocessing__ 
__all__ += __all_video__ 
__all__ += __all_data__ 
__all__ += __all_utils__ 
__all__ += __all_path__
__all__ += __all_color__

# Stop polluting the namespace
del __all_globals__ 
del __all_io__ 
del __all_resize__ 
del __all_opencv__ 
del __all_preprocess__ 
del __all_utilities__ 
del __all_visualizations__

del __all_preprocessing__ 
del __all_video__ 
del __all_data__ 
del __all_utils__ 
del __all_color__ 
# del __all_distance__
# del __all_filters__
# del __all_morph__