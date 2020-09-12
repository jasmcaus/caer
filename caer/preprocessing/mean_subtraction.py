# Author: Jason Dsouza
# Github: http://www.github.com/jasmcaus

# Importing the necessary packages
import cv2 as cv

class MeanProcess:
    def __init__(self, mean_sub_values):
        # mean_sub_values is a tuple
        self.rMean = mean_sub_values[0]
        self.gMean = mean_sub_values[1]
        self.bMean = mean_sub_values[2]

    def mean_preprocess(self, image):
        (b,g,r) = cv.split(image.astype('float32')[:3])

        # Subtracting the mean
        r -= self.rMean
        b -= self.bMean
        g -= self.gMean

        # Merging 
        return cv.merge([b,g,r])