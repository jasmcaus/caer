# Copyright (c) 2020 Jason Dsouza <jasmcaus@gmail.com>
# Protected under the MIT License (see LICENSE)
#pylint:disable=undefined-all-variable

from ._meta import version as v
from ._meta import author as __author__
from ._meta import release as r
from ._meta import contributors as c
__version__ = v
__contributors__ = c
__license__ = 'MIT License'
__copyright__ = 'Copyright (c) 2020 Jason Dsouza'
version = v
release = r
contributors = c


from .preprocess import preprocess_from_dir
from .preprocess import sep_train
from .preprocess import shuffle
from .preprocess import reshape
from .preprocess import normalize

from .utilities import saveNumpy
from .utilities import train_val_split
from .utilities import get_classes_from_dir
from .utilities import sort_dict
from .utilities import plotAcc

from .paths import list_media
from .paths import list_images
from .paths import list_videos
from .paths import listdir
from .paths import is_image
from .paths import is_video
from .paths import cwd
from .paths import get_size
from .paths import abspath
from .paths import osname
from .paths import chdir
from .paths import minijoin

from .opencv import get_opencv_version
from .opencv import to_gray
from .opencv import to_hsv
from .opencv import to_lab
from .opencv import to_rgb
from .opencv import url_to_image 
from .opencv import translate
from .opencv import rotate 
from .opencv import edges 

from .visualizations import hex_to_rgb
from .visualizations import draw_rectangle

from .images import resize 
from .images import center_crop 
from .images import load_img 

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


# from .io import __all__ as __all_io__
from .video import __all__ as __all_video__
from .preprocessing import __all__ as __all_preprocessing__
from .data import __all__ as __all_data__
from .utils import __all__ as __all_utils__


__all__ = (
    'preprocess_from_dir',
    'sep_train',
    'shuffle',
    'resize',
    'center_crop'
    'reshape',
    'normalize',
    'load_img',
    'train_val_split',
    'get_classes_from_dir',
    'sort_dict',
    'plotAcc',
    'list_media',
    'list_images',
    'list_videos',
    'is_image',
    'is_video'
    'listdir',
    'cwd',
    'minijoin',
    'get_size',
    'chdir',
    'osname',
    'abspath',
    'get_opencv_version',
    'get_caer_version',
    'get_caer_functions',
    'get_caer_methods',
    'to_gray',
    'to_rgb',
    'to_lab',
    'to_hsv',
    'url_to_image',
    'translate',
    'rotate',
    'edges',
    'hex_to_rgb',
    'draw_rectangle',
) + __all_preprocessing__ + __all_video__ + __all_data__ + __all_utils__
# ) + __all_preprocessing__ + __all_video__ + __all_io__ 