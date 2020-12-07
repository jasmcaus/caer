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

sep_matrix = np.matrix([[0.272, 0.534, 0.131],
                        [0.349, 0.686, 0.168],
                        [0.393, 0.769, 0.189]])

__all__ = [
    'brighten',
    'summer',
    'winter',
    'invert',
    'summer'
]

def brighten(img):
    """
        Brighten any BGR/RGB image

    Args:
        img (ndarray) : Any regular BGR/RGB image
    
    Returns:
        Brightened image
    """
    dup = img.copy()
    hsv = cv.cvtColor(dup, cv.COLOR_BGR2HSV) # convert image to HSV color space
    hsv = np.array(hsv, dtype = np.float64)
    hsv[:,:,1] = hsv[:,:,1]*1.25 # scale pixel values up for channel 1
    hsv[:,:,1][hsv[:,:,1]>255]  = 255
    hsv[:,:,2] = hsv[:,:,2]*1.25 # scale pixel values up for channel 2
    hsv[:,:,2][hsv[:,:,2]>255]  = 255
    hsv = np.array(hsv, dtype=np.uint8)
    dup = cv.cvtColor(hsv, cv.COLOR_HSV2BGR) # converting back to BGR used by OpenCV

    return dup
    

def invert(img):
    """
        Invert any BGR/RGB image

    Args:
        img (ndarray) : Any regular BGR/RGB image
    
    Returns:
        Inverted image
    """
    return cv.bitwise_not(img.copy())


def summer(img):
    """
        Convert an image using a "Winter" filter
        For values of gamma above 1, the values are increased and for below 1, the pixel values are decreased. 
        It is implemented by creating a lookup table and using `cv2.LUT()`. 
        The values of the Red channel of BGR and Saturation of HSV is increased by taking a value of gamma greater than 1 and the values of the Blue channel is decreased by taking a value less than 1.

    Args:
        img (ndarray) : Any regular BGR/RGB image
    
    Returns:
        Winter image
    """
    dup = img.copy()
    dup[:, :, 0] = _gamma_function(dup[:, :, 0], 0.75) # downscaling blue channel
    dup[:, :, 2] = _gamma_function(dup[:, :, 2], 1.25) # upscaling red channel
    hsv = cv.cvtColor(dup, cv.COLOR_BGR2HSV)
    hsv[:, :, 1] = _gamma_function(hsv[:, :, 1], 1.2) # upscaling saturation channel
    dup = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

    return dup 


def winter(img):
    """
        Convert an image using a "Winter" filter

    Args:
        img (ndarray) : Any regular BGR/RGB image
    
    Returns:
        Winter image
    """
    dup = img.copy()
    dup[:, :, 0] = _gamma_function(dup[:, :, 0], 1.25)
    dup[:, :, 2] = _gamma_function(dup[:, :, 2], 0.75)
    hsv = cv.cvtColor(dup, cv.COLOR_BGR2HSV)
    hsv[:, :, 1] = _gamma_function(hsv[:, :, 1], 0.8)
    dup = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

    return dup 


def _gamma_function(channel, gamma):
    invGamma = 1/gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8") #creating lookup table
    channel = cv.LUT(channel, table)
    return channel


def sepia(img):
    """
        Convert an image using a "Sepia" filter

    Args:
        img (ndarray) : Any regular BGR/RGB image
    
    Returns:
        Sepia image
    """
    dup = img.copy()
    dup = np.array(dup, dtype=np.float64) # converting to float to prevent loss
    dup = cv.transform(dup, sep_matrix) # multipying image with special sepia matrix
    dup[np.where(dup > 255)] = 255 # normalizing values greater than 255 to 255
    dup = np.array(dup, dtype=np.uint8) # converting back to int
    return dup 