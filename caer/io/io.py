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
from urllib.request import urlopen

from ..core import asarray
from ..color import bgr_to_rgb, bgr_to_gray, rgb_to_bgr, IMREAD_COLOR
from .._internal import _check_target_size
from ..path import exists
from .resize import resize


__all__ = [
    'imread',
    'imsave'
]


def imread(image_path, target_size=None, channels=3, rgb=True, resize_factor=None, preserve_aspect_ratio=False, interpolation='bilinear'):
    r"""
        Loads in an image from `image_path` (can be either a system filepath or a URL)

        Args:
            image_path (str): Filepath/URL to read the image from.
            target_size (tuple): Target size. Must be a tuple of ``(width, height)`` integer.
            channels: 1 (grayscale) or 3 (RGB/BGR). Default: 3
            rgb: Boolean to keep RGB ordering. Default: True
            resize_factor (float, tuple): Resizing Factor to employ. 
                Shrinks the image if ``resize_factor < 1``
                Enlarges the image if ``resize_factor > 1``
            preserve_aspect_ratio (bool): Prevent aspect ratio distortion (employs center crop).
            interpolation (str): Interpolation to use for resizing. Defaults to `'bilinear'`. 
                Supports `'bilinear'`, `'bicubic'`, `'area'`, `'nearest'`.


        Returns:
            Array with shape ``(height, width, channels)``.

        
        Examples::
            >> img = caer.imread(img_path) # From FilePath
            >> img.shape
            (427, 640, 3)

            >> img = caer.imread('https://raw.githubusercontent.com/jasmcaus/caer/master/caer/data/beverages.jpg') # From URL
            >> img.shape
            (427, 640, 3)

    """
    return _imread(image_path, target_size=target_size, channels=channels, rgb=rgb, resize_factor=resize_factor, preserve_aspect_ratio=preserve_aspect_ratio, interpolation=interpolation)


def imsave(path, img, rgb=True):
    r"""
        Saves an image file to `path`
            
    Args:
        path (str): Filepath to save the image to 
    
    Returns
        True; if `img` was written to `path`
        False; otherwise

    """
    try:
        # OpenCV uses BGR images and saves them as RGB images
        if rgb:
            img = rgb_to_bgr(img)
        return cv.imwrite(path, img)
    except:
        raise ValueError('`img` needs to be an opencv-specific image. Try reading the image using `caer.imread()`. More support for additional platforms will follow. Check the Changelog for further details.')


def _imread(image_path, target_size=None, channels=3, rgb=True, resize_factor=None, preserve_aspect_ratio=False, interpolation='bilinear'):   
    if target_size is not None:
        _ = _check_target_size(target_size)
        
    if not isinstance(channels, int) or channels not in [1,3]:
        raise ValueError('channels must be an integer - 1 (Grayscale) or 3 (RGB)')

    if not isinstance(rgb, bool):
        raise ValueError('rgb must be a boolean')
    
    if rgb and channels == 1:
        # Preference goes to Grayscale
        rgb = False
    
    interpolation_methods = {
        'nearest': 0,  '0': 0, 
        'bilinear': 1, '1': 1,
        'bicubic': 2,  '2': 2,
        'area': 3,     '3': 3,
    }

    if interpolation not in interpolation_methods:
        raise ValueError('Specify a valid interpolation type - area/nearest/bicubic/bilinear')

    try:
        image_array = _url_to_image(image_path, rgb=True)
    except Exception:
        if exists(image_path):
            image_array = _read_image(image_path)
        else:
            raise ValueError('Specify either a valid URL or valid filepath')
            

    # [INFO] Using the following piece of code results in a 'None'
    # if image_array == None:
    #     pass
    
    if channels == 1:
        image_array = bgr_to_gray(image_array)

    if target_size is not None or resize_factor is not None:
        image_array = resize(image_array, target_size, resize_factor=resize_factor, preserve_aspect_ratio=preserve_aspect_ratio, interpolation=interpolation)

    if rgb:
        image_array = bgr_to_rgb(image_array)

    return image_array


def _read_image(image_path):
    """Reads an image located at `path` into an array.

    Args:
        path (str): Path to a valid image file in the filesystem.

    Returns:
        Array of size `(height, width, channels)`.
    """
    if not exists(image_path):
        raise FileNotFoundError('The image file was not found')
    
    return cv.imread(image_path)


def _url_to_image(url, rgb=True):
    # Converts the image to a Numpy array and reads it in OpenCV
    response = urlopen(url)
    image = asarray(bytearray(response.read()), dtype='uint8')
    image = cv.imdecode(image, IMREAD_COLOR)

    if image:
        if rgb:
            image = bgr_to_rgb(image)
        return image
    else:
        raise ValueError(f'No such image at {url}')