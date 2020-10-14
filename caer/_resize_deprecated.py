# Copyright (c) 2020 Jason Dsouza <jasmcaus@gmail.com>
# Protected under the MIT License (see LICENSE)

#pylint: disable=bare-except

# Importing the necessary packages
import cv2 as cv
from .resize import _cv2_resize
from ._checks import _check_size

def resize(image, target_size, keep_aspect_ratio=False):
    """
        Resizes an image using advanced algorithms
        :param target_size: Tuple of size 2 in the format (width,height)
        :param keep_aspect_ratio: Boolean to keep/ignore aspect ratio when resizing
    """
    _ = _check_size(target_size)
    
    if not isinstance(keep_aspect_ratio, bool):
        raise ValueError('keep_aspect_ratio must be a boolean')

    new_w, new_h = target_size

    if not keep_aspect_ratio: # If False, use OpenCV's resize method
        return _cv2_resize(image, width=new_w, height=new_h)
    else:
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
    
    cropped = image[diff_h:diff_h+new_h, diff_w:diff_w+new_w]

    return cropped 