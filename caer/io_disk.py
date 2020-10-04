# Author: Jason Dsouza
# Github: http://www.github.com/jasmcaus

import os
import cv2 as cv

def read_image(path):
    """Reads an image located at `path` into an array.
    Arguments:
        path (str): Path to a valid image file in the filesystem.
    Returns:
        `numpy.ndarray` of size `(height, width, channels)`.
    """
    if not os.path.exists(path):
        raise ValueError('[ERROR] Specified path does not exist')
    
    return cv.imread(path)