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
        flag = check_mean_subtraction(mean_sub_values,channels)
        if flag:
            if channels == 3:
                self.rMean = mean_sub_values[0]
                self.gMean = mean_sub_values[1]
                self.bMean = mean_sub_values[2]
            if channels == 1:
                self.mean = mean_sub_values

    def mean_preprocess(self, image, channels):
        """
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

def compute_mean_from_dir(DIR, channels):
    """
        Mean must be computed ONLY on the train set
    """
    if channels == 3:
        rMean, gMean, bMean = 0,0,0
    if channels == 1:
        mean = 0
    count = 0

    if os.path.exists(DIR) is False:
        raise ValueError('The specified directory does not exist', DIR)

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
        return rMean, bMean, gMean

    if channels == 1:
        mean /= count
        return mean

def compute_mean(train, channels):
    """
        Computes mean over the train set and returns a tuple of dimensions=channels
        Train should not be normalized
    """
    if len(train) == 0:
        raise ValueError('[ERROR] Training set is empty')
    
    if type(train) is not list:
        raise ValueError('[ERROR] Training set must be a list of size=number of images and shape=image shape')

    if channels == 3:
        rMean, gMean, bMean = 0,0,0
    if channels == 1:
        mean = 0
    count = 0

    for img in train:
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
        return rMean, bMean, gMean

    if channels == 1:
        mean /= count
        return mean

def subtract_mean(val_data, channels, mean_sub_values):
    """
        Subtracts mean from the validation set
    """
    mean_process = MeanProcess(mean_sub_values,channels)
    if len(val_data) == 0:
        raise ValueError('[ERROR] Training set is empty')
    
    if type(val_data) is not list:
        raise ValueError('[ERROR] Training set must be a list of size=number of images and shape=image shape')

    for img in val_data:
        mean_process.mean_preprocess(img, channels)

def check_mean_subtraction(value, channels):
    """
        Checks if mean subtraction values are valid based on the number of channels
        Must be a tuple of dimensions = number of channels
    Returns boolean value
        True -> Expression is valid
        False -> Expression is invalid
    """
    if value is None:
        return False
    elif type(value) is tuple and len(value) == channels:
        return True
    else:
        raise ValueError(f'[ERROR] Expected a tuple of dimension {channels}', value) 