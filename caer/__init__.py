# Copyright 2020 The Caer Authors. All Rights Reserved.
#
# Licensed under the MIT License (see LICENSE);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at <https://opensource.org/licenses/MIT>
#
# ==============================================================================

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
from .utilities import train_val_split
from .utilities import get_classes_from_dir
from .utilities import sort_dict
# from .utilities import plotAcc

# Opencv-specific methods
from .opencv import get_opencv_version
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

# Image-related
from .images import resize 
from .images import center_crop 
from .images import load_img 
from .images import mean 

# Bringing in configuration variables from configs.py
from .configs import CROP_CENTRE
from .configs import CROP_TOP
from .configs import CROP_LEFT
from .configs import CROP_RIGHT 
from .configs import CROP_BOTTOM
from .configs import VALID_URL_NO_EXIST
from .configs import INVALID_URL_STRING


def get_caer_version():
    return __version__


def get_caer_functions():
    return __all__


def get_caer_methods():
    return __all__


# __all__ configs
from .configs import __all__ as __all_configs__
from .images import __all__ as __all_images__
from .opencv import __all__ as __all_opencv__
from .preprocess import __all__ as __all_preprocess__
from .time import __all__ as __all_time__
from .utilities import __all__ as __all_utilities__
from .visualizations import __all__ as __all_visualizations__

from .video import __all__ as __all_video__
from .preprocessing import __all__ as __all_preprocessing__
from .data import __all__ as __all_data__
from .utils import __all__ as __all_utils__
from .path import __all__ as __all_path__


__all__ = __all_configs__ + __all_images__ + __all_opencv__ + __all_preprocess__ + __all_time__ + __all_utilities__ + __all_visualizations__

__all__ += __all_preprocessing__ 
__all__ += __all_video__ 
__all__ += __all_data__ 
__all__ += __all_utils__ 
__all__ += __all_path__