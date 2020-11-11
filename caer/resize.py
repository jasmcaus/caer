# Copyright 2020 The Caer Authors. All Rights Reserved.
#
# Licensed under the MIT License (see LICENSE);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at <https://opensource.org/licenses/MIT>
#
# ==============================================================================

import math 
import cv2 as cv

from ._checks import _check_target_size
from .globals import (
    INTER_AREA, INTER_CUBIC, INTER_NEAREST, INTER_LINEAR
)


def resize(image, target_size=None, resize_factor=None, keep_aspect_ratio=False, interpolation='area'):
    """
        Resizes an image with a specified resizing factor, this factor can also be
        the target shape of the resized image specified as a tuple.
        Tuple must be of a shape 2 (width, height)
        Priority given to `resize_factor`
    """
    width, height = image[:2]

    if resize_factor is None:
        if target_size is None:
            if keep_aspect_ratio:
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
        keep_aspect_ratio = False 

        if not isinstance(resize_factor, (int, float)):
            raise ValueError('resize_factor must be an integer or float')

        if resize_factor > 1:
            interpolation = 'bicubic'

        new_shape = (int(resize_factor * image.shape[1]), int(resize_factor * image.shape[0]))
            
    interpolation_methods = {
        'nearest': INTER_NEAREST, # 0
        'bilinear': INTER_LINEAR, # 1
        'bicubic': INTER_CUBIC, # 2
        'area': INTER_AREA, # 3
    }

    if interpolation not in interpolation_methods:
        raise ValueError('Specify a valid interpolation type - area/nearest/bicubic/bilinear')

    if keep_aspect_ratio:
        return _resize_with_ratio(image, target_size=target_size, keep_aspect_ratio=keep_aspect_ratio)
    else:
        width, height = new_shape[:2]
        return _cv2_resize(image, (width, height), interpolation=interpolation_methods[interpolation])


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


def _resize_with_ratio(image, target_size, keep_aspect_ratio=False):
    """
        Resizes an image using advanced algorithms
        :param target_size: Tuple of size 2 in the format (width,height)
        :param keep_aspect_ratio: Boolean to keep/ignore aspect ratio when resizing
    """
    _ = _check_target_size(target_size)
    
    if not isinstance(keep_aspect_ratio, bool):
        raise ValueError('keep_aspect_ratio must be a boolean')

    org_h, org_w = image.shape[:2]
    target_w, target_h = target_size

    if target_h > org_h or target_w > org_w:
        raise ValueError('To compute resizing keeping the aspect ratio, the target size dimensions must be <= actual image dimensions')

    # Computing minimal resize
    # min_width, w_factor = _compute_minimal_resize(org_w, target_w)
    # min_height, h_factor = _compute_minimal_resize(org_h, target_h)
    minimal_resize_factor = _compute_minimal_resize((org_w, org_h), (target_w, target_h))

    # Resizing minimally
    image = _cv2_resize(image, (image.shape[1]//minimal_resize_factor, image.shape[0]//minimal_resize_factor))

    # Computing centre crop (to avoid extra crop, we resize minimally first)
    image = _compute_centre_crop(image, (target_w, target_h))

    if image.shape[:2] != target_size[:2]:
        image = _cv2_resize(image, (target_w, target_h))
    
    return image
    

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


def center_crop(image, target_size=None):
    """
        Computes the centre crop of an image using `target_size`
    """
    return _compute_centre_crop(image, target_size)


def _compute_centre_crop(image, target_size):
    _ = _check_target_size(target_size)

    # Getting org height and target
    org_w, org_h = image.shape[:2]
    target_w, target_h = target_size

    # The following line is actually the right way of accessing height and width of an opencv-specific image (height, width). However for some reason, while the code runs, this is flipped (it now becomes (width,height)). Testing needs to be done to catch this little bug
    # org_h, org_w = image.shape[:2]


    if target_h > org_h or target_w > org_w:
        raise ValueError('To compute centre crop, target size dimensions must be <= image dimensions')

    diff_h = (org_h - target_h) // 2
    diff_w = (org_w - target_w ) // 2
    
    return image[diff_h:diff_h + target_h, diff_w:diff_w + target_w]



__all__ = [
    'center_crop',
    'resize'
]