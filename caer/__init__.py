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

#pylint:disable=undefined-all-variable

from ._meta import version as __version__
from ._meta import author as __author__
from ._meta import release as __release__
from ._meta import contributors as __contributors__
__license__ = 'MIT License'
__copyright__ = 'Copyright (c) 2020 Jason Dsouza'


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
from .utilities import load
from .utilities import train_val_split
from .utilities import get_classes_from_dir
from .utilities import sort_dict
# from .utilities import plotAcc

# Opencv-specific methods
from .opencv import get_opencv_version
from .opencv import mean 
from .opencv import merge 
from .opencv import split 
from .opencv import to_gray
from .opencv import to_hsv
from .opencv import to_lab
from .opencv import to_rgb
from .opencv import url_to_image 
from .opencv import translate
from .opencv import rotate 
from .opencv import edges 

# General visualizations
from .visualizations import hex_to_rgb
from .visualizations import draw_rectangle

# Time
from .time import now

# Resize
from .resize import resize 
from .resize import center_crop 

# Image-related
from .io import imread 
from .io import imsave 

# Distance
from .distance import distance

# Convolve
from .convolve import daubechies
from .convolve import idaubechies
from .convolve import find
from .convolve import haar
from .convolve import ihaar
from .convolve import rank_filter
from .convolve import mean_filter
from .convolve import convolve
from .convolve import convolve1d
from .convolve import gaussian_filter
from .convolve import gaussian_filter1d
from .convolve import laplacian_2D

# Morphology
from .morph import cwatershed
from .morph import cerode
from .morph import erode
from .morph import cdilate
from .morph import dilate
from .morph import get_structuring_elem
from .morph import hitmiss

from .pillow import cv_to_pill
from .pillow import pill_to_cv


# Bringing in configuration variables from configs.py
from .configs import CROP_CENTRE
from .configs import CROP_TOP
from .configs import CROP_LEFT
from .configs import CROP_RIGHT 
from .configs import CROP_BOTTOM
from .configs import VALID_URL_NO_EXIST
from .configs import INVALID_URL_STRING

# except ImportError:
#     import sys 
#     _, e, _ = sys.exc_info()
#     sys.stderr.write(f"""\
#         Could not import submodules (exact error was: {e}).
#         There are many reasons for this error the most common one is that you have either not built the packages, built (using `python setup.py build`) or installed them (using `python setup.py install`) and then proceeded to test caer **without changing the current directory**.
#         Try installing and then changing to another directory before importing caer.
#         """)


def get_caer_version():
    return __version__


def get_caer_functions():
    return __all__


def get_caer_methods():
    return __all__


# __all__ configs
from .configs import __all__ as __all_configs__
from .io import __all__ as __all_io__
from .resize import __all__ as __all_resize__
from .opencv import __all__ as __all_opencv__
from .preprocess import __all__ as __all_preprocess__
from .time import __all__ as __all_time__
from .utilities import __all__ as __all_utilities__
from .visualizations import __all__ as __all_visualizations__
from .distance import __all__ as __all_distance__
from .convolve import __all__ as __all_convolve__
from .morph import __all__ as __all_morph__

from .video import __all__ as __all_video__
from .preprocessing import __all__ as __all_preprocessing__
from .data import __all__ as __all_data__
from .utils import __all__ as __all_utils__
from .path import __all__ as __all_path__
from .filters import __all__ as __all_filters__


__all__ = __all_configs__ + __all_io__ + __all_resize__ + __all_opencv__ + __all_preprocess__ + __all_time__ + __all_utilities__ + __all_visualizations__ + __all_distance__ + __all_convolve__ + __all_morph__ + __all_filters__

__all__ += __all_preprocessing__ 
__all__ += __all_video__ 
__all__ += __all_data__ 
__all__ += __all_utils__ 
__all__ += __all_path__