# Author: Jason Dsouza
# Github: http://www.github.com/jasmcaus

# Importing the necessary packages
import cv2 as cv

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