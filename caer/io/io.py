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
import urllib

from ..adorad import to_tensor
from ..color.constants import IMREAD_COLOR
from ..color.bgr import _bgr_to_rgb
from ..path import exists


__all__ = [
    'imread',
    'imsave'
]


def imread(image_path, rgb=True):
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

    return _imread(image_path, rgb=rgb)


def _imread(image_path, rgb=True):   
    try:
        # Returns RGB image
        img = _url_to_image(image_path)
        
        # If the URL is valid, but no image at that URL, NoneType is returned
        if img is None:
            raise ValueError('The URL specified does not point to an image')

        # return img

    # If the URL is invalid
    except Exception, urllib.error.URLError:
        if exists(image_path):
            img = _read_image(image_path) # returns RGB

        else:
            raise ValueError('Specify either a valid URL or filepath')
    
    img = to_tensor(img)
    img.cspace = 'rgb'
    
    if rgb is False:
        img = _bgr_to_rgb(img)
        # We need to convert back to tensor
        img = to_tensor(img)
        img.cspace = 'bgr'
    
    return img 

def _read_image(image_path):
    if not exists(image_path):
        raise FileNotFoundError('The image file was not found')

    # BGR image
    image =  cv.imread(image_path)
    # Convert to RGB
    image = _bgr_to_rgb(image)

    return image 


def _url_to_image(url):
    # Converts the image to a Numpy array and reads it in OpenCV
    response = urlopen(url)
    image = np.asarray(bytearray(response.read()), dtype='uint8')
    # BGR image
    image = cv.imdecode(image, IMREAD_COLOR)

    if image is not None:
        # Convert to RGB
        image = _bgr_to_rgb(image)

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
            img = _bgr_to_rgb(img)
        return cv.imwrite(path, img)
    except:
        raise ValueError('`img` needs to be a caer Tensor. Try reading the image using `caer.imread()`. More support for additional platforms will follow. Check the Changelog for further details.')