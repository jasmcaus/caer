# Copyright (c) 2020 Jason Dsouza <jasmcaus@gmail.com>
# Protected under the MIT License (see LICENSE)

import cv2 as cv

def resize(image, resize_factor, interpolation="area"):
    """
        Resizes an image with a specified resizing factor, this factor can also be
        the target shape of the resized image specified as a tuple.
        Tuple must be of a shape 2 (width, height)
    """
    interpolation_methods = {
        "area": cv.INTER_AREA,
        "nearest": cv.INTER_NEAREST,
        "bicubic": cv.INTER_CUBIC,
        "bilinear": cv.INTER_LINEAR
    }
    if interpolation not in interpolation_methods:
        raise ValueError('Specify a valid interpolation type')

    if type(resize_factor) is not tuple:
        new_shape = (int(resize_factor * image.shape[0]), int(resize_factor * image.shape[1]))
    else:
        if len(resize_factor) == 2:
            new_shape = resize_factor
        else:
            raise ValueError('Tuple Shape must be equal to 2 (width,height)')

    width, height = new_shape[:2]
    return _cv2_resize(image, width, height, interpolation=interpolation_methods[interpolation])


def _cv2_resize(image, width, height, interpolation=None):
    """
    ONLY TO BE USED INTERNALLY. NOT AVAILABLE FOR EXTERNAL USAGE. 
    Resizes the image ignoring the aspect ratio of the original image
    """
    if interpolation is None:
        interpolation = cv.INTER_AREA
    dimensions = (width,height)
    return cv.resize(image, dimensions, interpolation=interpolation)