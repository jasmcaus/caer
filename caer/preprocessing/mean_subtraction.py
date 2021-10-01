#    _____           ______  _____ 
#  / ____/    /\    |  ____ |  __ \
# | |        /  \   | |__   | |__) | Caer - Modern Computer Vision
# | |       / /\ \  |  __|  |  _  /  Languages: Python, C, C++, Cuda
# | |___   / ____ \ | |____ | | \ \  http://github.com/jasmcaus/caer
#  \_____\/_/    \_ \______ |_|  \_\

# Licensed under the MIT License <http://opensource.org/licenses/MIT>
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021 The Caer Authors <http://github.com/jasmcaus>


#pylint:disable=pointless-string-statement

from .._internal import _check_mean_sub_values
from ..path import exists, list_images
from ..io import imread 
from ..adorad import Tensor
from ..jit.annotations import Tuple, List

import numpy as np
import cv2 as cv

"""
    Important notes:
    Mean subtract must be computed ONLY on the training set and then later applied on the validation/test set
"""

__all__ = [
    'MeanProcess',
    'compute_mean',
    'compute_mean_from_dir',
    'subtract_mean',
]

class MeanProcess:
    def __init__(self, mean_sub_values, channels) -> None:
        # mean_sub_values is a tuple
        flag = _check_mean_sub_values(mean_sub_values, channels)
        if flag:
            if channels == 3:
                self.rMean = mean_sub_values[0]
                self.gMean = mean_sub_values[1]
                self.bMean = mean_sub_values[2]
            else:
                self.bgrMean = mean_sub_values[0]

    def mean_preprocess(self, image, channels) -> Tensor:
        """
            Mean Subtraction performed per channel
            Mean must be calculated ONLY on the training set
        """

        if channels == 3:
            b, g, r = cv.split(image.astype('float32'))[:3]

            # Subtracting the mean
            r -= self.rMean
            g -= self.gMean
            b -= self.bMean

            # Merging 
            return cv.merge([b,g,r])
        
        if channels == 1:
            image -= self.bgrMean
            return image
            

def compute_mean_from_dir(DIR, channels, per_channel_subtraction=True, recursive=True) -> Tuple:
    """
        Computes mean per channel
        Mean must be computed ONLY on the train set
    """
    if not exists(DIR):
        raise ValueError('The specified directory does not exist')
    
    image_list = list_images(DIR, recursive=recursive, use_fullpath=True, verbose=0)

    if len(image_list) == 0:
        raise ValueError(f'No images found at {DIR}')

    if channels == 3:
        rMean, gMean, bMean = 0, 0, 0
    if channels == 1:
        bgrMean = 0

    count = 0

    for tens_filepath in image_list:
        count += 1
        tens = imread(tens_filepath, rgb=True)

        if channels == 3:
            b, g, r = cv.mean(tens.astype('float32'))[:3]
            rMean += r
            bMean += b
            gMean += g

        if channels == 1:
            bgrMean += np.mean(tens.astype('float32'))

    # Computing average mean
    if channels == 3:
        rMean /= count
        bMean /= count 
        gMean /= count

        if per_channel_subtraction:
            return rMean, bMean, gMean
        else:
            mean_of_means = (rMean + bMean + gMean) / 3
            return mean_of_means, mean_of_means, mean_of_means

    if channels == 1:
        bgrMean /= count
        return tuple([bgrMean])


def compute_mean(data, channels, per_channel_subtraction=True) -> Tuple:
    """
        Computes mean per channel over the train set and returns a tuple of dimensions=channels
        Train should not be normalized
    """
    if len(data) == 0:
        raise ValueError('Dataset is empty')
    
    if not isinstance(data, (list, Tensor)):
        raise ValueError('Dataset must be a list of size=number of images and shape=image shape')

    if channels == 3:
        rMean, gMean, bMean = 0,0,0
    if channels == 1:
        bgrMean = 0

    count = 0

    for tens in data:
        count += 1
        if channels == 3:
            b, g, r = cv.mean(tens.astype('float32'))[:3]
            rMean += r
            gMean += g
            bMean += b
        if channels == 1:
            bgrMean += np.mean(tens.astype('float32'))

    # Computing average mean
    if channels == 3:
        rMean /= count
        bMean /= count 
        gMean /= count
        if per_channel_subtraction:
            return rMean, bMean, gMean
        else:
            mean_of_means = (rMean + bMean + gMean) / 3
            return mean_of_means, mean_of_means, mean_of_means

    if channels == 1:
        bgrMean /= count
        return tuple([bgrMean])


def subtract_mean(data, channels, mean_sub_values) -> List[str]:
    """
        Per channel subtraction values computed from compute_mean() or compute_mean_from_dir()
        Subtracts mean from the validation set
    """

    mean_process = MeanProcess(mean_sub_values, channels)

    if len(data) == 0:
        raise ValueError('Dataset is empty')
    
    if not isinstance(data, (list, Tensor)):
        raise ValueError('Dataset must be a list of size = number of images and shape = image shape')

    data = [mean_process.mean_preprocess(tens, channels) for tens in data]
    
    return data