# Author: Jason Dsouza
# Github: http://www.github.com/jasmcaus

#pylint:disable=pointless-string-statement

# Importing the necessary packages
import cv2 as cv
import os 

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
                self.mean = mean_sub_values

    def mean_preprocess(self, image, channels):
        """
            Mean Subtraction performed per channel
            Mean must be calculated ONLY on the training set
        """

        if channels == 3:
            (b,g,r) = cv.split(image.astype('float32'))[:3]

            # Subtracting the mean
            r -= self.rMean
            b -= self.bMean
            g -= self.gMean

            # Merging 
            return cv.merge([b,g,r])
        
        if channels == 1:
            image -= self.mean
            return image
            

def compute_mean_from_dir(DIR, channels, per_channel_subtraction=True):
    """
        Computes mean per channel
        Mean must be computed ONLY on the train set
    """
    if os.path.exists(DIR) is False:
        raise ValueError('The specified directory does not exist', DIR)

    if channels == 3:
        rMean, gMean, bMean = 0,0,0
    if channels == 1:
        mean = 0
    count = 0

    for root, _, files in os.walk(DIR):
        for file in files:
            if file.endswith('.png') or file.endswith('.jpg'):
                count += 1
                filepath = root + os.sep + file
                img = cv.imread(filepath)
                if channels == 3:
                    b,g,r = cv.mean(img.astype('float32'))[:3]
                    rMean += r
                    bMean += b
                    gMean += g
                if channels == 1:
                    mean += cv.mean(img.astype('float32'))[0]

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
        mean /= count
        return tuple([mean])


def compute_mean(data, channels, per_channel_subtraction=True):
    """
        Computes mean oer channel over the train set and returns a tuple of dimensions=channels
        Train should not be normalized
    """
    if len(data) == 0:
        raise ValueError('[ERROR] Dataset is empty')
    
    if type(data) is not list:
        raise ValueError('[ERROR] Dataset must be a list of size=number of images and shape=image shape')

    if channels == 3:
        rMean, gMean, bMean = 0,0,0
    if channels == 1:
        mean = 0
    count = 0

    for img in data:
        count += 1
        if channels == 3:
            b,g,r = cv.mean(img.astype('float32'))[:3]
            rMean += r
            bMean += b
            gMean += g
        if channels == 1:
            mean += cv.mean(img.astype('float32'))[0]

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
        mean /= count
        return tuple([mean])


def subtract_mean(data, channels, mean_sub_values):
    """
        Per channel subtraction values computed from compute_mean() or compute_mean_from_dir()
        Subtracts mean from the validation set
    """

    mean_process = MeanProcess(mean_sub_values, channels)

    if len(data) == 0:
        raise ValueError('[ERROR] Dataset is empty')
    
    if type(data) is not list:
        raise ValueError('[ERROR] Dataset must be a list of size=number of images and shape=image shape')

    data = [mean_process.mean_preprocess(img, channels) for img in data]
    
    return data


def _check_mean_sub_values(value, channels):
    """
        Checks if mean subtraction values are valid based on the number of channels
        'value' must be a tuple of dimensions = number of channels
    Returns boolean:
        True -> Expression is valid
        False -> Expression is invalid
    """
    if value is None:
        raise ValueError('[ERROR] Value(s) specified is of NoneType()')
    
    if type(value) is not tuple:
        # If not a tuple, we convert it to one
        try:
            value = tuple(value)
        except TypeError:
            value = tuple([value])
    
    if channels not in [1,3]:
        raise ValueError('[ERROR] Number of channels must be either 1 (Grayscale) or 3 (RGB/BGR)')

    if len(value) not in [1,3]:
        raise ValueError('[ERROR] Tuple length must be either 1 (subtraction over the entire image) or 3 (per channel subtraction)', value)
    
    if len(value) == channels:
        return True 

    else:
        raise ValueError(f'[ERROR] Expected a tuple of dimension {channels}', value) 