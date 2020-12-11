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

from ..color import is_rgb_image, is_gray_image
from .._internal import _check_target_size
from ..globals import (
    INTER_AREA, INTER_CUBIC, INTER_NEAREST, INTER_LINEAR
)

MAX_VALUES_BY_DTYPE = {
    np.dtype("uint8"): 255,
    np.dtype("uint16"): 65535,
    np.dtype("uint32"): 4294967295,
    np.dtype("float32"): 1.0,
}

__all__ = [
    'hflip',
    'vflip',
    'hvflip',
    'transpose',
    'scale',
    'rotate',
    'translate',
    'solarize',
    'posterize',
    'equalize',
    'clip'
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


def rotate(img, angle, rotPoint=None):
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

    height, width = img.shape[:2]

    # If no rotPoint is specified, we assume the rotation point to be around the centre
    if rotPoint is None:
        centre = (width//2, height//2)

    rotMat = cv.getRotationMatrix2D(centre, angle, scale=1.0)

    warp_fn = _proc_in_chunks(cv.warpAffine, src=img, M=rotMat, dsize=(width, height))

    return warp_fn(img)


def translate(image, x, y):
    r"""Translates a given image across the x-axis and the y-axis

    Args:
        x (int): shifts the image right (positive) or left (negative)
        y (int): shifts the image down (positive) or up (negative)
    
    Returns:
        The translated image

    """
    transMat = np.float32([[1, 0, x], [0, 1, y]])
    return cv.warpAffine(image, transMat, (image.shape[1], image.shape[0]))


def scale(img, scale_factor, interpolation='bilinear'):
    interpolation_methods = {
        'nearest': INTER_NEAREST, # 0
        'bilinear': INTER_LINEAR, # 1
        'bicubic': INTER_CUBIC, # 2
        'area': INTER_AREA, # 3
    }
    if interpolation not in interpolation_methods:
        raise ValueError('Specify a valid interpolation type - area/nearest/bicubic/bilinear')

    if scale_factor > 1:
        # Neater, more precise
        interpolation = 'bicubic'

    height, width = img.shape[:2]
    new_height, new_width = int(height * scale_factor), int(width * scale_factor)

    return cv.resize(img, (new_width,new_height), interpolation=interpolation)


def crop(img, x_min, y_min, x_max, y_max):
    height, width = img.shape[:2]
    if x_max <= x_min or y_max <= y_min:
        raise ValueError(
            "We should have x_min < x_max and y_min < y_max. But we got"
            " (x_min = {x_min}, y_min = {y_min}, x_max = {x_max}, y_max = {y_max})".format(
                x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max
            )
        )

    if x_min < 0 or x_max > width or y_min < 0 or y_max > height:
        raise ValueError(
            "Values for crop should be non negative and equal or smaller than image sizes"
            "(x_min = {x_min}, y_min = {y_min}, x_max = {x_max}, y_max = {y_max}, "
            "height = {height}, width = {width})".format(
                x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, height=height, width=width
            )
        )

    return img[y_min:y_max, x_min:x_max]


def center_crop(image, target_size=None):
    r"""Computes the centre crop of an image using `target_size`

    Args:
        image (ndarray): Valid image array
        target_size (tuple): Size of the centre crop. Must be in the format `(width,height)`
    
    Returns:
        Cropped Centre (ndarray)
    
    Examples::

        >> img = caer.data.bear() # Standard 640x427 image
        >> cropped = caer.center_crop(img, target_size=(200,200))
        >> cropped.shape
        (200,200,3)

    """
    return _compute_centre_crop(image, target_size)


def rand_crop(img, crop_height, crop_width, h_start, w_start):
    height, width = img.shape[:2]
    if height < crop_height or width < crop_width:
        raise ValueError(
            "Requested crop size ({crop_height}, {crop_width}) is "
            "larger than the image size ({height}, {width})".format(
                crop_height=crop_height, crop_width=crop_width, height=height, width=width
            )
        )
    x1, y1, x2, y2 = _get_random_crop_coords(height, width, crop_height, crop_width, h_start, w_start)
    img = img[y1:y2, x1:x2]
    return img


def _compute_centre_crop(img, target_size):
    _ = _check_target_size(target_size)

    # Getting org height and target
    org_h, org_w = img.shape[:2]
    target_w, target_h = target_size

    # The following line is actually the right way of accessing height and width of an opencv-specific image (height, width). However for some reason, while the code runs, this is flipped (it now becomes (width,height)). Testing needs to be done to catch this little bug
    # org_h, org_w = img.shape[:2]


    if target_h > org_h or target_w > org_w:
        raise ValueError('To compute centre crop, target size dimensions must be <= img dimensions')

    diff_h = (org_h - target_h) // 2
    diff_w = (org_w - target_w ) // 2
    
    # img[y:y+h, x:x+h]
    return img[diff_h:diff_h + target_h, diff_w:diff_w + target_w]


def _get_random_crop_coords(height, width, crop_height, crop_width, h_start, w_start):
    y1 = int((height - crop_height) * h_start)
    y2 = y1 + crop_height
    x1 = int((width - crop_width) * w_start)
    x2 = x1 + crop_width
    return x1, y1, x2, y2


def _get_num_channels(img):
    return img.shape[2] if len(img.shape) == 3 else 1


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
        num_channels = _get_num_channels(img)
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


def solarize(img, threshold=128):
    r"""Invert all pixel values above a threshold.

    Args:
        img (ndarray): The image to solarize.
        threshold (int): All pixels above this grayscale level are inverted.

    Returns:
        Solarized image (ndarray)
    
    Examples::

        >> img = caer.data.sunrise()
        >> solarized = caer.solarize(img, threshold=128)
        >> solarized.shape
        (427,640,3)

    """
    dtype = img.dtype
    max_val = MAX_VALUES_BY_DTYPE[dtype]

    if dtype == np.dtype("uint8"):
        lut = [(i if i < threshold else max_val - i) for i in range(max_val + 1)]

        prev_shape = img.shape
        img = cv.LUT(img, np.array(lut, dtype=dtype))

        if len(prev_shape) != len(img.shape):
            img = np.expand_dims(img, -1)
        return img

    result_img = img.copy()
    cond = img >= threshold
    result_img[cond] = max_val - result_img[cond]
    return result_img


def posterize(img, bits):
    r"""Reduce the number of bits for each color channel in the image.

    Args:
        img (ndarray): Image to posterize.
        bits (int): Number of high bits. Must be in range [0, 8]

    Returns:
        Image with reduced color channels (ndarray)
    
    Examples::

        >> img = caer.data.sunrise()
        >> posterized = caer.posterize(img, bits=4)
        >> posterized.shape
        (427,640,3)

    """
    bits = np.uint8(bits)

    if img.dtype != np.uint8:
        raise TypeError("Image must have uint8 channel type")

    if np.any((bits < 0) | (bits > 8)):
        raise ValueError("bits must be in range [0, 8]")

    if not bits.shape or len(bits) == 1:
        if bits == 0:
            return np.zeros_like(img)
        if bits == 8:
            return img.copy()

        lut = np.arange(0, 256, dtype=np.uint8)
        mask = ~np.uint8(2 ** (8 - bits) - 1)
        lut &= mask

        return cv.LUT(img, lut)

    if not is_rgb_image(img):
        raise TypeError("If `bits` is iterable, image must be RGB")

    result_img = np.empty_like(img)
    for i, channel_bits in enumerate(bits):
        if channel_bits == 0:
            result_img[..., i] = np.zeros_like(img[..., i])
        elif channel_bits == 8:
            result_img[..., i] = img[..., i].copy()
        else:
            lut = np.arange(0, 256, dtype=np.uint8)
            mask = ~np.uint8(2 ** (8 - channel_bits) - 1)
            lut &= mask

            result_img[..., i] = cv.LUT(img[..., i], lut)

    return result_img


def clip(img, dtype, maxval):
    return np.clip(img, 0, maxval).astype(dtype)


def _equalize_cv(img, mask=None):
    if mask is None:
        return cv.equalizeHist(img)

    histogram = cv.calcHist([img], [0], mask, [256], (0, 256)).ravel()
    i = 0
    for val in histogram:
        if val > 0:
            break
        i += 1
    i = min(i, 255)

    total = np.sum(histogram)
    if histogram[i] == total:
        return np.full_like(img, i)

    scale = 255.0 / (total - histogram[i])
    _sum = 0

    lut = np.zeros(256, dtype=np.uint8)
    i += 1
    for i in range(i, len(histogram)):
        _sum += histogram[i]
        lut[i] = clip(round(_sum * scale), np.dtype("uint8"), 255)

    return cv.LUT(img, lut)


def equalize(img, mask=None, by_channels=True):
    r"""Equalize the image histogram.

    Args:
        img (ndarray)*: RGB or grayscale image.
        mask (ndarray)*: An optional mask.  If given, only the pixels selected by the mask are included in the analysis. Maybe 1 channel or 3 channel array.
        by_channels (bool): If True, use equalization by channels separately, else convert image to YCbCr representation and use equalization by `Y` channel.

    Returns:
        Equalized image (ndarray)
    

    Examples::

        >> img = caer.data.beverages()   
        >> equalized = caer.equalize(img, mask=None)  
        >> equalized.shape   
        (427,640,3)
    
    """
    if img.dtype != np.uint8:
        raise TypeError("Image must have uint8 channel type")

    if mask is not None:
        if is_rgb_image(mask) and is_gray_image(img):
            raise ValueError("Wrong mask shape. Image shape: {}. Mask shape: {}".format(img.shape, mask.shape))

        if not by_channels and not is_gray_image(mask):
            raise ValueError(
                "When `by_channels=False`, only 1-channel mask is supported. Mask shape: {}".format(mask.shape)
            )

    if mask is not None:
        mask = mask.astype(np.uint8)

    if is_gray_image(img):
        return _equalize_cv(img, mask)

    if not by_channels:
        result_img = cv.cvtColor(img, cv.COLOR_RGB2YCrCb)
        result_img[..., 0] = _equalize_cv(result_img[..., 0], mask)
        return cv.cvtColor(result_img, cv.COLOR_YCrCb2RGB)

    result_img = np.empty_like(img)
    for i in range(3):
        if mask is None:
            _mask = None
        elif is_gray_image(mask):
            _mask = mask
        else:
            _mask = mask[..., i]

        result_img[..., i] = _equalize_cv(img[..., i], _mask)

    return result_img