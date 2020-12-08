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

from .utilities import median


__all__ = [
    'get_opencv_version',
    'mean',
    'merge',
    'split',
    'url_to_image',
    # 'color_map',
    # 'energy_map',
    'translate',
    'rotate',
    'edges'
]


def get_opencv_version():
    return cv.__version__[0]


def edges(img, threshold1=None, threshold2=None, use_median=True, sigma=None):
    if not isinstance(use_median, bool):
        raise ValueError('use_median must be a boolean')

    if not isinstance(threshold1, int) or not isinstance(threshold2, int):
        raise ValueError('Threshold values must be integers')

    if img is None:
        raise ValueError('Image is of NoneType()')

    if not use_median and (threshold1 is None or threshold2 is None):
        raise ValueError('Specify valid threshold values')
    
    if use_median:
        if sigma is None:
            sigma = .3

        # computes the median of the single channel pixel intensities
        med = median(img)

        # Canny edge detection using the computed mean
        low = int(max(0, (1.0-sigma) * med))
        up = int(min(255, (1.0+sigma) * med))
        canny_edges = cv.Canny(img, low, up)
    
    else:
        canny_edges = cv.Canny(img, threshold1, threshold2)

    return canny_edges


def mean(image, mask=None):
    try:
        return cv.mean(image, mask=mask)
    except:
        raise ValueError('mean() expects an image')


def merge(img):
    # if not isinstance(img, (list, np.ndarray)):
    #     raise ValueError('img must be a list or numpy.ndarray of (ideally) shape = 3)')

    return cv.merge(img)


def split(img):
    try:
        return cv.split(img)
    except:
        raise ValueError('mean() expects an image')


# def energy_map(img):
#     img = bgr_to_gray(img.astype(np.uint8))

#     dx = cv.Sobel(img, cv.CV_16S, 1, 0, ksize=3)
#     abs_x = cv.convertScaleAbs(dx)
#     dy = cv.Sobel(img, cv.CV_16S, 0, 1, ksize=3)
#     abs_y = cv.convertScaleAbs(dy)
#     output = cv.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)

#     return output


# def color_map(img):
#     gray_img = bgr_to_gray(img) 

#     heatmap = cv.applyColorMap(gray_img, 11)
#     superimpose = cv.addWeighted(heatmap, 0.7, img, 0.3, 0)

#     return superimpose