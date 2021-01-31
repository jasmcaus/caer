#    _____           ______  _____ 
#  / ____/    /\    |  ____ |  __ \
# | |        /  \   | |__   | |__) | Caer - Modern Computer Vision
# | |       / /\ \  |  __|  |  _  /  Languages: Python, C, C++, Cuda
# | |___   / ____ \ | |____ | | \ \  http://github.com/jasmcaus/caer
#  \_____\/_/    \_ \______ |_|  \_\

# Licensed under the MIT License <http://opensource.org/licenses/MIT>
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021 The Caer Authors <http://github.com/jasmcaus>


import cv2 as cv
import numpy as np 
from urllib.request import urlopen
# from urllib.error import URLError

from .resize import resize
from ..adorad import to_tensor, Tensor
from ..color import rgb2bgr, to_bgr
from ..path import exists
from .._internal import _check_target_size

__all__ = [
    'imread',
    'imsave'
]

IMREAD_COLOR = 1


def imread(image_path, rgb=True, target_size=None, resize_factor=None, preserve_aspect_ratio=False, interpolation='bilinear') -> Tensor:
    r"""
        Loads in an image from `image_path` (can be either a system filepath or a URL)

        Args:
            image_path (str): Filepath/URL to read the image from.
            rgb (bool): Boolean to keep RGB ordering. Default: True

        Returns:
            Tensor with shape ``(height, width, channels)``.

        Examples::

            >> tens = caer.imread(tens_path) # From FilePath
            >> tens.shape
            (427, 640, 3)

            >> tens = caer.imread('https://raw.githubusercontent.com/jasmcaus/caer/master/caer/data/beverages.jpg') # From URL
            >> tens.shape
            (427, 640, 3)

    """    

    return _imread(image_path, rgb=rgb,target_size=target_size, resize_factor=resize_factor, preserve_aspect_ratio=preserve_aspect_ratio, interpolation=interpolation)


def _imread(image_path, rgb=True, target_size=None, resize_factor=None, preserve_aspect_ratio=False, interpolation='bilinear') -> Tensor:   
    if target_size is not None:
        _ = _check_target_size(target_size)
    
    # if not isinstance(channels, int) or channels not in [1, 3]:
    #     raise ValueError('channels must be an integer - 1 (Grayscale) or 3 (RGB)')

    interpolation_methods = {
        'nearest':  0, '0': 0,  0: 0, # 0
        'bilinear': 1, '1': 1,  1: 1, # 1
        'bicubic':  2, '2': 2,  2: 2, # 2
        'area':     3, '3': 3,  3: 3  # 3
    }

    if interpolation not in interpolation_methods:
        raise ValueError('Specify a valid interpolation type - area/nearest/bicubic/bilinear')


    if exists(image_path):
        tens = _read_image(image_path) # returns RGB

    # TODO: Create URL validator
    elif image_path.startswith(('http://', 'https://')):
        # Returns RGB image
        tens = _url_to_image(image_path)
        
        # If the URL is valid, but no image at that URL, NoneType is returned
        if tens is None:
            raise ValueError('The URL specified does not point to an image')

    else:
        raise ValueError('Specify either a valid URL or filepath')

    # try:
    #     # Returns RGB image
    #     tens = _url_to_image(image_path)
        
    #     # If the URL is valid, but no image at that URL, NoneType is returned
    #     if tens is None:
    #         raise ValueError('The URL specified does not point to an image')

    #     # return tens

    # # If the URL is invalid
    # except (Exception, URLError):
    #     if exists(image_path):
    #         tens = _read_image(image_path) # returns RGB

    #     else:
    #         raise ValueError('Specify either a valid URL or filepath')
    
    if target_size is not None or resize_factor is not None:
        # Enforce a Tensor is passed to resize() 
        to_tensor(tens, cspace='rgb')
        tens = resize(tens, target_size, resize_factor=resize_factor, preserve_aspect_ratio=preserve_aspect_ratio,interpolation=interpolation)


    # If `rgb=False`, then we assume that BGR is expected
    if not rgb:
        tens = rgb2bgr(tens)
        # We need to convert back to tensor
        return to_tensor(tens, cspace='bgr')
    
    return to_tensor(tens, cspace='rgb')


def _read_image(image_path):
    r"""
        Returns an RGB ndarray
    """
    if not exists(image_path):
        raise FileNotFoundError('The image file was not found')

    # BGR image
    tens =  cv.imread(image_path)

    # Convert to RGB
    # WARNING: DO NOT USE to_rgb() as it creates a brand new Tensor (which defaults to RGB)
    # This issue will, hopefully, be fixed in a future update.
    # tens = to_rgb(tens)
    return cv.cvtColor(tens, cv.COLOR_BGR2RGB)


def _url_to_image(url):
    r"""
        Returns an RGB ndarray.
    """
    response = urlopen(url)
    tens = np.asarray(bytearray(response.read()), dtype='uint8')
    # BGR image
    tens = cv.imdecode(tens, IMREAD_COLOR)

    if tens is not None:
        return cv.cvtColor(tens, cv.COLOR_BGR2RGB)
        
    else:
        raise ValueError(f'No image found at "{url}"')


def imsave(path, tens) -> bool:
    r"""
        Saves a Tensor to `path`
            
    Args:
        path (str): Filepath to save the image to 
    
    Returns
        ``True``; if `tens` was written to `path`
        ``False``; otherwise

    Examples::

        >> tens = caer.data.audio_mixer()
        >> caer.imsave('audio_mixer.png', tens)
        True

    """
    if not isinstance(tens, Tensor):
        raise TypeError('`tens` must be a caer.Tensor')

    # Convert to tensor 
    _ = tens._nullprt() # raises a ValueError if we're dealing with a Foreign Tensor with illegal `.cspace` value

    try:
        # OpenCV uses BGR Tensors and saves them as RGB images
        if tens.cspace !='bgr':
            tens = to_bgr(tens)
        return cv.imwrite(path, tens)
    except:
        raise ValueError('`tens` needs to be a caer Tensor. Try reading the image using `caer.imread()`. More support for additional platforms will follow. Check the Changelog for further details.')