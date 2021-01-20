#    _____           ______  _____ 
#  / ____/    /\    |  ____ |  __ \
# | |        /  \   | |__   | |__) | Caer - Modern Computer Vision
# | |       / /\ \  |  __|  |  _  /  Languages: Python, C, C++
# | |___   / ____ \ | |____ | | \ \  http://github.com/jasmcaus/caer
#  \_____\/_/    \_ \______ |_|  \_\

# Licensed under the MIT License <http://opensource.org/licenses/MIT>
# SPDX-License-Identifier: MIT
# Copyright (c) 2020 The Caer Authors <http://github.com/jasmcaus>


import cv2 as cv
import numpy as np 
from urllib.request import urlopen
# from urllib.error import URLError

from .resize import resize
from ..adorad import to_tensor_
from ..color import to_rgb
from ..path import exists
from .._internal import _check_target_size

__all__ = [
    'imread',
    'imsave'
]

IMREAD_COLOR = 1


def imread(image_path, rgb=True, target_size=None, resize_factor=None, preserve_aspect_ratio=False, interpolation='bilinear'):
    r"""
        Loads in an image from `image_path` (can be either a system filepath or a URL)

        Args:
            image_path (str): Filepath/URL to read the image from.
            rgb (bool): Boolean to keep RGB ordering. Default: True

        Returns:
            Tensor with shape ``(height, width, channels)``.

        Examples::

            >> img = caer.imread(img_path) # From FilePath
            >> img.shape
            (427, 640, 3)

            >> img = caer.imread('https://raw.githubusercontent.com/jasmcaus/caer/master/caer/data/beverages.jpg') # From URL
            >> img.shape
            (427, 640, 3)

    """    

    return _imread(image_path, rgb=rgb,target_size=target_size, resize_factor=resize_factor, preserve_aspect_ratio=preserve_aspect_ratio, interpolation=interpolation)


def _imread(image_path, rgb=True, target_size=None, resize_factor=None, preserve_aspect_ratio=False, interpolation='bilinear'):   
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
        img = _read_image(image_path) # returns RGB

    # TODO: Create URL validator 
    elif image_path.startswith(('http://', 'https://')):
        # Returns RGB image
        img = _url_to_image(image_path)
        
        # If the URL is valid, but no image at that URL, NoneType is returned
        if img is None:
            raise ValueError('The URL specified does not point to an image')

    else:
        raise ValueError('Specify either a valid URL or filepath')

    # try:
    #     # Returns RGB image
    #     img = _url_to_image(image_path)
        
    #     # If the URL is valid, but no image at that URL, NoneType is returned
    #     if img is None:
    #         raise ValueError('The URL specified does not point to an image')

    #     # return img

    # # If the URL is invalid
    # except (Exception, URLError):
    #     if exists(image_path):
    #         img = _read_image(image_path) # returns RGB

    #     else:
    #         raise ValueError('Specify either a valid URL or filepath')
    
    if target_size is not None or resize_factor is not None:
        image_array = resize(image_array, 
                             target_size, 
                             resize_factor=resize_factor, 
                             preserve_aspect_ratio=preserve_aspect_ratio, 
                             interpolation=interpolation
        )

    img = to_tensor_(img)
    img.cspace = 'rgb'
    
    if not rgb:
        img = to_rgb(img)
        # We need to convert back to tensor
        img = to_tensor_(img)
        img.cspace = 'bgr'
    
    return to_tensor_(img)


def _read_image(image_path):
    if not exists(image_path):
        raise FileNotFoundError('The image file was not found')

    # BGR image
    image =  cv.imread(image_path)
    # Convert to RGB
    image = to_rgb(image)

    return image 


def _url_to_image(url):
    # Converts the image to a Numpy array and reads it in OpenCV
    response = urlopen(url)
    image = np.asarray(bytearray(response.read()), dtype='uint8')
    # BGR image
    image = cv.imdecode(image, IMREAD_COLOR)

    if image is not None:
        # Convert to RGB
        image = to_rgb(image)

        return image 
        
    else:
        raise ValueError(f'No image found at "{url}"')


def imsave(path, img, rgb=True):
    r"""
        Saves an image file to `path`
            
    Args:
        path (str): Filepath to save the image to 
    
    Returns
        ``True``; if `img` was written to `path`
        ``False``; otherwise

    Examples::

        >> img = caer.data.audio_mixer()
        >> caer.imsave('audio_mixer.png', img)
        True

    """
    try:
        # OpenCV uses BGR images and saves them as RGB images
        if rgb:
            img = to_rgb(img)
        return cv.imwrite(path, img)
    except:
        raise ValueError('`img` needs to be a caer Tensor. Try reading the image using `caer.imread()`. More support for additional platforms will follow. Check the Changelog for further details.')