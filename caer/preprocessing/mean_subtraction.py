# Author: Jason Dsouza
# Github: http://www.github.com/jasmcaus

#pylint:disable=pointless-string-statement

# Importing the necessary packages
import cv2 as cv

"""
    Important notes:
    Mean subtract must be computed ONLY on the training set and then later applied on the validation/test set
"""

class MeanProcess:
    def __init__(self, mean_sub_values, channels):
        # mean_sub_values is a tuple
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
            (b,g,r) = cv.split(image.astype('float32')[:3])

            # Subtracting the mean
            r -= self.rMean
            b -= self.bMean
            g -= self.gMean

            # Merging 
            return cv.merge([b,g,r])
            
        if channels == 1:
            image -= self.mean
            return image