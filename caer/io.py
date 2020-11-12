#
#  _____ _____ _____ ____
# |     | ___ | ___  | __|  Caer - Modern Computer Vision
# |     |     |      | \    version 3.9.1
# |_____|     | ____ |  \   http://github.com/jasmcaus/caer

# Licensed under the MIT License <http://opensource.org/licenses/MIT>.
# SPDX-License-Identifier: MIT
# Copyright (c) 2020 The Caer Authors <http://github.com/jasmcaus>.
# 


import cv2 as cv
# import numpy as np 

from ._internal import _check_target_size
from .opencv import bgr_to_rgb, bgr_to_gray, url_to_image
from .utils.validators import is_valid_url
from .path import exists
from .resize import resize


def imread(image_path, target_size=None, channels=3, rgb=False, resize_factor=None, keep_aspect_ratio=False):
    """
        Loads in an image from `image_path`
        Arguments
            image_path: Filepath/URL to read the image from
            target_size: Target image size
            channels: 1 (grayscale) or 3 (RGB/BGR). Default: 3
            rgb: Boolean to keep RGB ordering. Default: False
            resize_factor: Resizes the image using `resize_factor`. Default: None
            keep_aspect_ratio: Resized image to `target_size` keeping aspect ratio. Some parts of the image may not be included. Default: False
    """
    return _imread(image_path, target_size=target_size, channels=channels, rgb=rgb, resize_factor=resize_factor, keep_aspect_ratio=keep_aspect_ratio)


def _imread(image_path, target_size=None, channels=3, rgb=False, resize_factor=None, keep_aspect_ratio=False):   
    if target_size is not None:
        _ = _check_target_size(target_size)
        
    if not isinstance(channels, int) or channels not in [1,3]:
        raise ValueError('channels must be an integer - 1 (Grayscale) or 3 (RGB)')

    if not isinstance(rgb, bool):
        raise ValueError('rgb must be a boolean')
    
    if rgb and channels == 1:
        # Preference goes to Grayscale
        rgb = False

    if is_valid_url(image_path) is True:
        image_array = url_to_image(image_path, rgb=False)
    elif exists(image_path):  
        image_array = _read_image(image_path)
    else:
        raise ValueError('Specify either a valid URL or valid filepath')
            

    # [INFO] Using the following piece of code results in a 'None'
    # if image_array == None:
    #     pass
    
    if channels == 1:
        image_array = bgr_to_gray(image_array)

    if target_size is not None or resize_factor is not None:
        image_array = resize(image_array, target_size, resize_factor=resize_factor, keep_aspect_ratio=keep_aspect_ratio)

    if rgb:
        image_array = bgr_to_rgb(image_array)

    return image_array


def _read_image(image_path):
    """Reads an image located at `path` into an array.
    Arguments:
        path (str): Path to a valid image file in the filesystem.
    Returns:
        `numpy.ndarray` of size `(height, width, channels)`.
    """
    if not exists(image_path):
        raise FileNotFoundError('The image file was not found')
    
    return cv.imread(image_path)
    

def imsave(filename, img):
    cv.imwrite(filename, img)


__all__ = [
    'imread',
    'imsave'
]