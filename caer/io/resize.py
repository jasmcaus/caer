#    _____           ______  _____ 
#  / ____/    /\    |  ____ |  __ \
# | |        /  \   | |__   | |__) | Caer - Modern Computer Vision
# | |       / /\ \  |  __|  |  _  /  Languages: Python, C, C++
# | |___   / ____ \ | |____ | | \ \  http://github.com/jasmcaus/caer
#  \_____\/_/    \_ \______ |_|  \_\

# Licensed under the MIT License <http://opensource.org/licenses/MIT>
# SPDX-License-Identifier: MIT
# Copyright (c) 2020 The Caer Authors <http://github.com/jasmcaus>


import math 
import cv2 as cv

from .._internal import _check_target_size
from ..globals import (
    INTER_AREA, INTER_CUBIC, INTER_NEAREST, INTER_LINEAR
)

__all__ = [
    'resize'
]


def resize(image, target_size=None, resize_factor=None, preserve_aspect_ratio=False, interpolation='bilinear'):
    r"""
        Resizes an image to a target_size without aspect ratio distortion.
        
        Your output images will be of size ``target_size``, and will not be distorted. Instead, the parts of the image that do not fit within the target size get cropped out.

        The resizing process is:
        1. Resize the image as minimally as possible.
        2. Take the largest centered crop of the image with dimensions = ``target_size``.
        
        Alternatively, you may use:
        ```python
        size = (200,200)
        img = caer.resize(img, target_size=size, preserve_aspect_ratio=True)
        ``` 

        Note:
            ``caer.imread()`` comes with an in-built functionality to resize your images, eliminating the need for you to call ``caer.resize()``. This is purely optional and may appeal to certain users. 

            You may also use ``caer.smart_resize()`` for on-the-fly image resizing that `preserves the aspect ratio`. 

    
        Args:
            img (ndarray): Input Image. Must be in the format ``(height, width, channels)``.
            target_size (tuple): Target size. Must be a tuple of ``(width, height)`` integer.
            resize_factor (float, tuple): Resizing Factor to employ. 
                Shrinks the image if ``resize_factor < 1``
                Enlarges the image if ``resize_factor > 1``
            preserve_aspect_ratio (bool): Prevent aspect ratio distortion (employs center crop).
            interpolation (str): Interpolation to use for resizing. Defaults to `'bilinear'`. 
                Supports `'bilinear'`, `'bicubic'`, `'area'`, `'nearest'`.
        
        
        Returns:
            Array with shape ``(height, width, channels)``.


        Examples::

            >> img = caer.data.sunrise()
            >> img.shape
            (427, 640, 3)

            >> resized = caer.resize(img, target_size=(200,200)) # Hard-resize. May distort aspect ratio
            >> resized.shape
            (200, 200, 3)

            >> resized_wf = caer.resize(img, resize_factor=.5) # Resizes the image to half its dimensions
            >> resized_wf.shape
            (213, 320, 3)

            >> resized = caer.resize(img, target_size=(200,200), preserve_aspect_ratio=True) # Preserves aspect ratio
            >> resized.shape
            (200, 200, 3)

    """
    # Opencv uses the (h,w) format
    height, width = image.shape[:2]
    interpolation = str(interpolation)

    if resize_factor is None:
        if target_size is None:
            if preserve_aspect_ratio:
                raise ValueError('Specify a target size')
            else:
                raise ValueError('Specify either a resize factor or target dimensions')
        if target_size is not None:
            if len(target_size) == 2:
                new_shape = target_size
            else:
                raise ValueError('Tuple shape must be = 2 (width, height)')

    if resize_factor is not None:
        target_size = None
        preserve_aspect_ratio = False 

        if not isinstance(resize_factor, (int, float)):
            raise ValueError('resize_factor must be an integer or float')

        if resize_factor > 1:
            interpolation = 'bicubic'

        new_shape = (int(resize_factor * width), int(resize_factor * height))
            
    interpolation_methods = {
        'nearest': INTER_NEAREST, '0': INTER_NEAREST, # 0
        'bilinear': INTER_LINEAR, '1': INTER_LINEAR,# 1
        'bicubic': INTER_CUBIC, '2': INTER_CUBIC,# 2
        'area': INTER_AREA, '3': INTER_AREA,# 3
    }

    if interpolation not in interpolation_methods:
        raise ValueError('Specify a valid interpolation type - area/nearest/bicubic/bilinear')

    if preserve_aspect_ratio:
        return _resize_with_ratio(image, target_size=target_size, preserve_aspect_ratio=preserve_aspect_ratio, interpolation=interpolation_methods[interpolation])
    else:
        width, height = new_shape[:2]
        return _cv2_resize(image, (width, height), interpolation=interpolation_methods[interpolation])


def smart_resize(img, target_size, interpolation='bilinear'):
    r"""
        Resizes an image to a target_size without aspect ratio distortion.
        
        Your output images will be of size `target_size`, and will not be distorted. Instead, the parts of the image that do not fit within the target size get cropped out.

        The resizing process is:
        1. Resize the image as minimally as possible.
        2. Take the largest centered crop of the image with dimensions = `target_size`.
        
        Alternatively, you may use:
        ```python
        size = (200,200)
        img = caer.resize(img, target_size=size, preserve_aspect_ratio=True)
        ``` 
    
        Args:
            img (ndarray): Input Image. Must be in the format `(height, width, channels)`.
            target_size (tuple): Target size. Must be a tuple of `(width, height)` integer.
            interpolation (str): Interpolation to use for resizing. Defaults to `'bilinear'`. 
                Supports `'bilinear'`, `'bicubic'`, `'area'`, `'nearest'`.
        
        Returns:
            Array with shape `(height, width, channels)`

        Examples::
        
            >> img = caer.data.sunrise()
            >> img.shape
            (427, 640, 3)
            >> resized = caer.smart_resize(img, target_size=(200,200))
            >> resized.shape
            (200, 200, 3)

    """

    return _resize_with_ratio(img, target_size=target_size, preserve_aspect_ratio=True, interpolation=interpolation)


def _cv2_resize(image, target_size, interpolation=None):
    """
    ONLY TO BE USED INTERNALLY. NOT AVAILABLE FOR EXTERNAL USAGE. 
    Resizes the image ignoring the aspect ratio of the original image
    """
    _ = _check_target_size(target_size)

    width, height = target_size[:2]

    if interpolation is None:
        interpolation = INTER_AREA

    dimensions = (width, height)

    return cv.resize(image, dimensions, interpolation=interpolation)


def _resize_with_ratio(img, target_size, preserve_aspect_ratio=False, interpolation='bilinear'):
    """
        Resizes an image using advanced algorithms
        :param target_size: Tuple of size 2 in the format (width,height)
        :param preserve_aspect_ratio: Boolean to keep/ignore aspect ratio when resizing
    """
    _ = _check_target_size(target_size)
    interpolation = str(interpolation)
    
    if not isinstance(preserve_aspect_ratio, bool):
        raise ValueError('preserve_aspect_ratio must be a boolean')
    
    interpolation_methods = {
        'nearest': INTER_NEAREST, '0': INTER_NEAREST, # 0
        'bilinear': INTER_LINEAR, '1': INTER_LINEAR,# 1
        'bicubic': INTER_CUBIC, '2': INTER_CUBIC,# 2
        'area': INTER_AREA, '3': INTER_AREA,# 3
    }

    if interpolation not in interpolation_methods:
        raise ValueError('Specify a valid interpolation type - area/nearest/bicubic/bilinear')

    org_h, org_w = img.shape[:2]
    target_w, target_h = target_size

    if target_h > org_h or target_w > org_w:
        raise ValueError('To compute resizing keeping the aspect ratio, the target size dimensions must be <= actual image dimensions')

    # Computing minimal resize
    # min_width, w_factor = _compute_minimal_resize(org_w, target_w)
    # min_height, h_factor = _compute_minimal_resize(org_h, target_h)
    minimal_resize_factor = _compute_minimal_resize((org_w, org_h), (target_w, target_h))

    # Resizing minimally
    img = _cv2_resize(img, (org_w//minimal_resize_factor, org_h//minimal_resize_factor))

    # Computing centre crop (to avoid extra crop, we resize minimally first)
    img = _compute_centre_crop(img, (target_w, target_h))

    if img.shape[:2] != target_size[:2]:
        img = _cv2_resize(img, (target_w, target_h), interpolation=interpolation_methods[interpolation])
    
    return img
    

def _compute_minimal_resize(org_size, target_dim):
    # for i in range(10):
    #     i += 1
    #     d = dim*i
    #     if org_dim >= d and dim < dim*(i+1):
    #         if (org_dim - dim*(i+1)) > dim:
    #             continue
    #         else:
    #             return d, i
    # import math 
    # mi = math.floor(org_dim/dim)
    # d = dim * mi 
    # return d, mi

    if not isinstance(org_size, tuple) or not isinstance(target_dim, tuple):
        raise ValueError('org_size and target_dim must be a tuple')

    if len(org_size) != 2 or len(target_dim) != 2:
        raise ValueError('Size of tuple must be = 2')

    org_h, org_w = org_size[:2]
    targ_w, targ_h = target_dim[:2]

    h_factor = math.floor(org_h/targ_h)
    w_factor = math.floor(org_w/targ_w)

    if h_factor <= w_factor:
        return h_factor 
    else:
        return w_factor


def _compute_centre_crop(img, target_size):
    _ = _check_target_size(target_size)

    # Getting org height and target
    org_h, org_w = img.shape[:2]
    target_w, target_h = target_size

    # The following line is actually the right way of accessing height and width of an opencv-specific image (height, width). However for some reason, while the code runs, this is flipped (it now becomes (width,height)). Testing needs to be done to catch this little bug
    # org_h, org_w = img.shape[:2]


    if target_h > org_h or target_w > org_w:
        raise ValueError('To compute centre crop, target size dimensions must be <= img dimensions')

    diff_h = (org_h - target_h) // 2
    diff_w = (org_w - target_w ) // 2
    
    # img[y:y+h, x:x+h]
    return img[diff_h:diff_h + target_h, diff_w:diff_w + target_w]