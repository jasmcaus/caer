# Copyright (c) 2020 Jason Dsouza <jasmcaus@gmail.com>
# Protected under the MIT License (see LICENSE)

import cv2 as cv
from ._checks import _check_size


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
                raise ValueError('Tuple shape must be equal to 2 (width, height)')

    if resize_factor is not None:
        target_size = None
        keep_aspect_ratio = False 

        if not isinstance(resize_factor, (int, float)):
            raise ValueError('resize_factor must be an integer or float')

        if resize_factor > 1:
            interpolation = 'bicubic'

        new_shape = (int(resize_factor * image.shape[0]), int(resize_factor * image.shape[1]))
            
    interpolation_methods = {
        "area": cv.INTER_AREA,
        "nearest": cv.INTER_NEAREST,
        "bicubic": cv.INTER_CUBIC,
        "bilinear": cv.INTER_LINEAR
    }
    if interpolation not in interpolation_methods:
        raise ValueError('Specify a valid interpolation type - area/nearest/bicubic/bilinear')

    if keep_aspect_ratio:
        return resize_with_ratio(image, target_size=target_size, keep_aspect_ratio=keep_aspect_ratio)
    else:
        width, height = new_shape[:2]
        return _cv2_resize(image, width, height, interpolation=interpolation_methods[interpolation])


def _cv2_resize(image, width, height, interpolation=None):
    """
    ONLY TO BE USED INTERNALLY. NOT AVAILABLE FOR EXTERNAL USAGE. 
    Resizes the image ignoring the aspect ratio of the original image
    """
    if interpolation is None:
        interpolation = cv.INTER_AREA
    dimensions = (height, width)
    return cv.resize(image, dimensions, interpolation=interpolation)


def resize_with_ratio(image, target_size, keep_aspect_ratio=False):
    """
        Resizes an image using advanced algorithms
        :param target_size: Tuple of size 2 in the format (width,height)
        :param keep_aspect_ratio: Boolean to keep/ignore aspect ratio when resizing
    """
    _ = _check_size(target_size)
    
    if not isinstance(keep_aspect_ratio, bool):
        raise ValueError('keep_aspect_ratio must be a boolean')

    new_w, new_h = target_size

    org_height, org_width = image.shape[:2]
    # Computing minimal resize
    min_width, w_factor = _compute_minimal_resize(org_width, new_w)
    min_height, h_factor = _compute_minimal_resize(org_height, new_h)

    # Computing centre crop 
    image = _compute_centre_crop(image, (min_width, min_height))
    
    # Resizing minimally
    image = cv.resize(image, dsize=(min_width,min_height))

    # Resizing to new dimensions
    image = cv.resize(image, dsize=(image.shape[1]//w_factor, image.shape[0]//h_factor))
    return image


def _compute_minimal_resize(org_dim,dim):
    for i in range(10):
        i += 1
        d = dim*i
        if org_dim >= d and org_dim < dim*(i+1):
            return d, i


def _compute_centre_crop(image, target_size):
    _ = _check_size(target_size)
    # Getting org height and target
    org_height, org_width = image.shape[:2]
    new_w, new_h = target_size

    if new_h > org_height or new_w > org_width:
        raise ValueError('To compute centre crop, target size dimensions must be <= image dimensions')

    diff_h = org_height - new_h
    diff_w = org_width - new_w 
    
    cropped = image[diff_h:diff_h + new_h, diff_w:diff_w + new_w]

    return cropped 