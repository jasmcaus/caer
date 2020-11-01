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
from .opencv import to_rgb, to_gray, url_to_image
from .utils.validators import is_valid_url
from .path import exists


def load_img(image_path, target_size=None, channels=3, rgb=True, resize_factor=None, keep_aspect_ratio=False):
    """
        Loads in an image from `image_path`
        Arguments
            image_path: Filepath/URL to read the image from
            target_size: Target image size
            channels: 1 (grayscale) or 3 (RGB/BGR). Default: 3
            rgb: Boolean to keep RGB ordering. Default: True
            resize_factor: Resizes the image using `resize_factor`. Default: None
            keep_aspect_ratio: Resized image to `target_size` keeping aspect ratio. Some parts of the image may not be included. Default: False
    """
    return _load_img(image_path, target_size=target_size, channels=channels, rgb=rgb, resize_factor=resize_factor, keep_aspect_ratio=keep_aspect_ratio)


def _load_img(image_path, target_size=None, channels=3, rgb=True, resize_factor=None, keep_aspect_ratio=False):   
    if target_size is not None:
        _ = _check_target_size(target_size)
        
    if not isinstance(channels, int) or channels not in [1,3]:
        raise ValueError('channels must be an integer - 1 (Grayscale) or 3 (RGB)')

    if not isinstance(rgb, bool):
        raise ValueError('rgb must be a boolean')

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
        image_array = to_gray(image_array)

    if target_size is not None or resize_factor is not None:
        image_array = resize(image_array, target_size, resize_factor=resize_factor, keep_aspect_ratio=keep_aspect_ratio)

    if rgb:
        image_array = to_rgb(image_array)

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
        interpolation = cv.INTER_AREA

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


def mean(image, mask=None):
    try:
        return cv.mean(image, mask=mask)
    except:
        raise ValueError('mean() expects an image')


def merge(matrix_vector):
    if not isinstance(matrix_vector, list):
        raise ValueError('matrix_vector must be a list of (ideally) shape = 3)')

    return cv.merge(matrix_vector)


def split(img):
    try:
        return cv.split(img)
    except:
        raise ValueError('mean() expects an image')


__all__ = [
    'load_img',
    'center_crop',
    'resize',
    'mean',
    'merge',
    'split'
]