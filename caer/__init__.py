# Copyright (c) 2020 Jason Dsouza <jasmcaus@gmail.com>
# Protected under the MIT License (see LICENSE)

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
from .preprocess import _printTotal

from .utils import load_img
from .utils import saveNumpy
from .utils import train_val_split
from .utils import get_classes_from_dir
from .utils import sort_dict
from .utils import plotAcc

from .paths import list_media
from .paths import list_images
from .paths import list_videos
from .paths import listdir

from .opencv import get_opencv_version
from .opencv import to_gray
from .opencv import to_hsv
from .opencv import to_lab
from .opencv import to_rgb
from .opencv import url_to_image 
from .opencv import translate
from .opencv import rotate 
from .opencv import canny 

from .visualizations import hex_to_rgb
from .visualizations import draw_rectangle

from .io_disk import _read_image


#pylint:disable=undefined-all-variable

__all__ = (
    'preprocess_from_dir',
    'sep_train',
    'shuffle',
    'reshape',
    'normalize',
    '_printTotal',
    'load_img',
    'train_val_split',
    'get_classes_from_dir',
    'sort_dict',
    'plotAcc',
    'list_media',
    '_read_image'
    'list_images',
    'list_videos',
    'listdir',
    'get_opencv_version',
    'to_gray',
    'to_rgb',
    'to_lab',
    'to_hsv',
    'url_to_image',
    'translate',
    'rotate',
    'canny',
    'hex_to_rgb',
    'draw_rectangle',
    'extract_frames',
    'LiveVideoStream',
    'VideoStream',
    'count_frames',
    'get_fps',
    'MeanProcess',
    'compute_mean',
    'compute_mean_from_dir',
    'subtract_mean',
    '_check_mean_sub_values',
    'PatchPreProcess',
    'HDF5Dataset',
    'load_dataset'
)