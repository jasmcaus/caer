# Author: Jason Dsouza
# Github: http://www.github.com/jasmcaus

# Importing the necessary packages
import cv2 as cv

class MeanProcess:
    def __init__(self, rMean, gMean, bMean):
        self.rMean = rMean
        self.gMean = gMean
        self.bMean = bMean

    def mean_preprocess(self, image):
        (b,g,r) = cv.split(image.astype('float32')[:3])

        # Subtracting the mean
        r -= self.rMean
        b -= self.bMean
        g -= self.gMean

        # Merging 
        return cv.merge([b,g,r])