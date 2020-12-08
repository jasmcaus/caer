#    _____           ______  _____ 
#  / ____/    /\    |  ____ |  __ \
# | |        /  \   | |__   | |__) | Caer - Modern Computer Vision
# | |       / /\ \  |  __|  |  _  /  Languages: Python, C, C++
# | |___   / ____ \ | |____ | | \ \  http://github.com/jasmcaus/caer
#  \_____\/_/    \_ \______ |_|  \_\

# Licensed under the MIT License <http://opensource.org/licenses/MIT>
# SPDX-License-Identifier: MIT
# Copyright (c) 2020 The Caer Authors <http://github.com/jasmcaus>

import numpy as np 
import cv2 as cv 

__all__ = [
    'hflip',
    'vflip',
    'hvflip',
    'transpose',
    'scale',
    'rotate',
    'translate'
]


def hflip(img):
    return np.ascontiguousarray(img[:, ::-1, ...])


def vflip(img):
    return np.ascontiguousarray(img[::-1, ...])


def hvflip(img):
    return hflip(vflip(img))


def transpose(img):
    if len(img.shape) > 2:
        return img.transpose(1, 0, 2)
    else:
        return img.transpose(1, 0)
    

def rot90(img, factor):
    img = np.rot90(img, factor)
    return np.ascontiguousarray(img)


def _proc_in_chunks(process_fn, **kwargs):
    """
    Wrap OpenCV function to enable processing images with more than 4 channels.
    Limitations:
        This wrapper requires image to be the first argument and rest must be sent via named arguments.
    Args:
        process_fn: Transform function (e.g cv.resize).
        kwargs: Additional parameters.
    Returns:
        numpy.ndarray: Transformed image.
    """

    def __process_fn(img):
        num_channels = get_num_channels(img)
        if num_channels > 4:
            chunks = []
            for index in range(0, num_channels, 4):
                if num_channels - index == 2:
                    # Many OpenCV functions cannot work with 2-channel images
                    for i in range(2):
                        chunk = img[:, :, index + i : index + i + 1]
                        chunk = process_fn(chunk, **kwargs)
                        chunk = np.expand_dims(chunk, -1)
                        chunks.append(chunk)
                else:
                    chunk = img[:, :, index : index + 4]
                    chunk = process_fn(chunk, **kwargs)
                    chunks.append(chunk)
            img = np.dstack(chunks)
        else:
            img = process_fn(img, **kwargs)
        return img

    return __process_fn


def rotate(image, angle, rotPoint=None):
    """
        Rotates an given image by an angle around a particular rotation point (if provided) or centre otherwise.
    """
    # h, w = image.shape[:2]
    # (cX, cY) = (w/2, h/2)

    # # Computing the sine and cosine (rotation components of the matrix)
    # transMat = cv.getRotationMatrix2D((cX, cY), angle, scale=1.0)
    # cos = np.abs(transMat[0, 0])
    # sin = np.abs(transMat[0, 1])

    # # compute the new bounding dimensions of the image
    # nW = int((h*sin) + (w*cos))
    # nH = int((h*cos) + (w*sin))

    # # Adjusts the rotation matrix to take into account translation
    # transMat[0, 2] += (nW/2) - cX
    # transMat[1, 2] += (nH/2) - cY

    # # Performs the actual rotation and returns the image
    # return cv.warpAffine(image, transMat, (nW, nH))

    height, width = image.shape[:2]

    # If no rotPoint is specified, we assume the rotation point to be around the centre
    if rotPoint is None:
        centre = (width//2, height//2)

    rotMat = cv.getRotationMatrix2D(centre, angle, scale=1.0)

    warp_fn = _proc_in_chunks(cv.warpAffine, src=image, M=rotMat, dsize=(width, height))

    return warp_fn(img)


def translate(image, x, y):
    """
        Translates a given image across the x-axis and the y-axis
        :param x: shifts the image right (positive) or left (negative)
        :param y: shifts the image down (positive) or up (negative)
    """
    transMat = np.float32([[1, 0, x], [0, 1, y]])
    return cv.warpAffine(image, transMat, (image.shape[1], image.shape[0]))


def scale(img, scale, interpolation=cv2.INTER_LINEAR):
    height, width = img.shape[:2]
    new_height, new_width = int(height * scale), int(width * scale)
    return resize(img, new_height, new_width, interpolation)