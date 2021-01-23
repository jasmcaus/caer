#    _____           ______  _____ 
#  / ____/    /\    |  ____ |  __ \
# | |        /  \   | |__   | |__) | Caer - Modern Computer Vision
# | |       / /\ \  |  __|  |  _  /  Languages: Python, C, C++
# | |___   / ____ \ | |____ | | \ \  http://github.com/jasmcaus/caer
#  \_____\/_/    \_ \______ |_|  \_\

# Licensed under the MIT License <http://opensource.org/licenses/MIT>
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-21 The Caer Authors <http://github.com/jasmcaus>


import cv2 as cv
import numpy as np

from .path import listdir, minijoin


__all__ = [
    'get_classes_from_dir',
    'median',
    'npmean',
    'asarray',
    'to_array',
    'array',
    'train_val_split',
    'sort_dict',
    'get_opencv_version',
    'mean',
    'merge',
    'split',
    # 'color_map',
    # 'energy_map',
    'edges'
]


def median(arr, axis=None):
    return np.median(arr, axis=axis)


def mean(image, mask=None):
    try:
        return cv.mean(image, mask=mask)
    except:
        raise ValueError('mean() expects an image')


def merge(tens):
    # if not isinstance(tens, (list, caer.Tensor)):
    #     raise ValueError('tens must be a list or caer.Tensor of (ideally) shape = 3)')

    return cv.merge(tens)


def split(tens):
    try:
        return cv.split(tens)
    except:
        raise ValueError('mean() expects an image')
    

def npmean(arr):
    return np.mean(arr)


def array(obj, dtype=None, order='K'):
    return np.array(obj, dtype=dtype, order=order)


def to_array(obj, dtype=None, order='K'):
    return np.array(obj, dtype=dtype, order=order)


def asarray(obj, dtype=None, order=None):
    return np.asarray(obj, dtype=dtype, order=order)


def get_classes_from_dir(DIR, verbose=0):
    if len(listdir(DIR, verbose=0)) == 0:
        raise ValueError('The specified directory does not seem to have any folders in it')
    else:
        import os 
        classes = [i for i in listdir(DIR, recursive=False, verbose=verbose) if os.path.isdir(minijoin(DIR, i))]
        return classes


def _sep(data):
    
    x = [i[0] for i in data]
    y = [i[1] for i in data]

    return x, y


def train_val_split(X, y, val_ratio=.2):
    """
        Do not use if mean subtraction is being employed
        Returns X_train, X_val, y_train, y_val
    """
    if len(X) != len(y):
        raise ValueError('X must be equal to y')
    
    data = [] 
    for i in range(len(X)):
        data.append([X[i], y[i]])
    
    # import random
    # random.shuffle(data)

    split = int(len(X) - (len(X) * val_ratio)) - 1

    data_train = data[0:split]
    data_test = data[split:]

    X_train, y_train = _sep(data_train)
    X_val, y_val = _sep(data_test)

    return X_train, X_val, y_train, y_val


def sort_dict(unsorted_dict, descending=False):
    """ 
        Sorts a dictionary in ascending order (if descending = False) or descending order (if descending = True)
    """
    if not isinstance(descending, bool):
        raise ValueError('`descending` must be a boolean')

    return sorted(unsorted_dict.items(), key=lambda x:x[1], reverse=descending)


def get_opencv_version():
    return cv.__version__[0]


def edges(tens, threshold1=None, threshold2=None, use_median=True, sigma=None):
    if not isinstance(use_median, bool):
        raise ValueError('use_median must be a boolean')

    if not isinstance(threshold1, int) or not isinstance(threshold2, int):
        raise ValueError('Threshold values must be integers')

    if tens is None:
        raise ValueError('Image is of NoneType()')

    if not use_median and (threshold1 is None or threshold2 is None):
        raise ValueError('Specify valid threshold values')
    
    if use_median:
        if sigma is None:
            sigma = .3

        # computes the median of the single channel pixel intensities
        med = median(tens)

        # Canny edge detection using the computed mean
        low = int(max(0, (1.0-sigma) * med))
        up = int(min(255, (1.0+sigma) * med))
        canny_edges = cv.Canny(tens, low, up)
    
    else:
        canny_edges = cv.Canny(tens, threshold1, threshold2)

    return canny_edges


# def energy_map(tens):
#     tens = bgr_to_gray(tens.astype(np.uint8))

#     dx = cv.Sobel(tens, cv.CV_16S, 1, 0, ksize=3)
#     abs_x = cv.convertScaleAbs(dx)
#     dy = cv.Sobel(tens, cv.CV_16S, 0, 1, ksize=3)
#     abs_y = cv.convertScaleAbs(dy)
#     output = cv.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)

#     return output


# def color_map(tens):
#     gray_tens = bgr_to_gray(tens) 

#     heatmap = cv.applyColorMap(gray_tens, 11)
#     superimpose = cv.addWeighted(heatmap, 0.7, tens, 0.3, 0)

#     return superimpose