# Copyright 2020 The Caer Authors. All Rights Reserved.
#
# Licensed under the MIT License (see LICENSE);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at <https://opensource.org/licenses/MIT>
#
# ==============================================================================

#pylint:disable=pointless-string-statement

from .._checks import _check_mean_sub_values
from ..path import exists, list_images
from ..images import mean, merge, split, load_img


"""
    Important notes:
    Mean subtract must be computed ONLY on the training set and then later applied on the validation/test set
"""

class MeanProcess:
    def __init__(self, mean_sub_values, channels):
        # mean_sub_values is a tuple
        flag = _check_mean_sub_values(mean_sub_values, channels)
        if flag:
            if channels == 3:
                self.rMean = mean_sub_values[0]
                self.gMean = mean_sub_values[1]
                self.bMean = mean_sub_values[2]
            else:
                self.bgrMean = mean_sub_values

    def mean_preprocess(self, image, channels):
        """
            Mean Subtraction performed per channel
            Mean must be calculated ONLY on the training set
        """

        if channels == 3:
            b, g, r = split(image.astype('float32'))[:3]

            # Subtracting the mean
            r -= self.rMean
            b -= self.bMean
            g -= self.gMean

            # Merging 
            return merge([b,g,r])
        
        if channels == 1:
            image -= self.bgrMean
            return image
            

def compute_mean_from_dir(DIR, channels, per_channel_subtraction=True):
    """
        Computes mean per channel
        Mean must be computed ONLY on the train set
    """
    if exists(DIR) is False:
        raise ValueError('The specified directory does not exist')
    
    image_list = list_images(DIR, include_subdirs=True, use_fullpath=True, verbose=0)

    if len(image_list) == 0:
        raise ValueError(f'No images found at {DIR}')

    if channels == 3:
        rMean, gMean, bMean = 0,0,0
    if channels == 1:
        bgrMean = 0

    count = 0

    for img_filepath in image_list:
        count += 1
        img = load_img(img_filepath, rgb=False)

        if channels == 3:
            b, g, r = mean(img.astype('float32'))[:3]
            rMean += r
            bMean += b
            gMean += g

        if channels == 1:
            bgrMean += mean(img.astype('float32'))[0]

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


def compute_mean(data, channels, per_channel_subtraction=True):
    """
        Computes mean oer channel over the train set and returns a tuple of dimensions=channels
        Train should not be normalized
    """
    if len(data) == 0:
        raise ValueError('Dataset is empty')
    
    if isinstance(data, list):
        raise ValueError('Dataset must be a list of size=number of images and shape=image shape')

    if channels == 3:
        rMean, gMean, bMean = 0,0,0
    if channels == 1:
        bgrMean = 0
    count = 0

    for img in data:
        count += 1
        if channels == 3:
            b,g,r = mean(img.astype('float32'))[:3]
            rMean += r
            bMean += b
            gMean += g
        if channels == 1:
            bgrMean += mean(img.astype('float32'))[0]

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


def subtract_mean(data, channels, mean_sub_values):
    """
        Per channel subtraction values computed from compute_mean() or compute_mean_from_dir()
        Subtracts mean from the validation set
    """

    mean_process = MeanProcess(mean_sub_values, channels)

    if len(data) == 0:
        raise ValueError('Dataset is empty')
    
    if not isinstance(data, list):
        raise ValueError('Dataset must be a list of size = number of images and shape = image shape')

    data = [mean_process.mean_preprocess(img, channels) for img in data]
    
    return data


__all__ = [
    'MeanProcess',
    'compute_mean',
    'compute_mean_from_dir',
    'subtract_mean',
]