# Copyright (c) 2020 Jason Dsouza <jasmcaus@gmail.com>
# Protected under the MIT License (see LICENSE)

import os
import cv2 as cv

def _read_image(image_path):
    """Reads an image located at `path` into an array.
    Arguments:
        path (str): Path to a valid image file in the filesystem.
    Returns:
        `numpy.ndarray` of size `(height, width, channels)`.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError('[ERROR] The image file was not found')
    
    return cv.imread(image_path)