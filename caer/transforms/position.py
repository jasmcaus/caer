#    _____           ______  _____ 
#  / ____/    /\    |  ____ |  __ \
# | |        /  \   | |__   | |__) | Caer - Modern Computer Vision
# | |       / /\ \  |  __|  |  _  /  Languages: Python, C, C++, Cuda
# | |___   / ____ \ | |____ | | \ \  http://github.com/jasmcaus/caer
#  \_____\/_/    \_ \______ |_|  \_\

# Licensed under the MIT License <http://opensource.org/licenses/MIT>
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021 The Caer Authors <http://github.com/jasmcaus>

import numpy as np 
import cv2 as cv 
import random 
import collections

from ..adorad import Tensor, to_tensor
from .._internal import _check_target_size
from ..globals import (
    INTER_AREA, INTER_CUBIC, INTER_NEAREST, INTER_LINEAR
)

pad_to_str = {
    'constant':  0,
    'edge':      1,
    'reflect':   4,
    'symmetric': 2
}

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
    'rand_flip',
    'transpose',
    'scale',
    'rotate',
    'translate',
    'solarize',
    'posterize',
    'equalize',
    'clip',
    'pad'
]

def _is_rgb_image(tens):
    tens = to_tensor(tens, override_checks=True)
    return tens.is_rgb()
    # return len(tens.shape) == 3 and tens.shape[-1] == 3


def _is_gray_image(tens):
    tens = to_tensor(tens, override_checks=True)
    return tens.is_gray()
    # return (len(tens.shape) == 2) or (len(tens.shape) == 3 and tens.shape[-1] == 1)


def _get_num_channels(tens):
    return tens.shape[2] if len(tens.shape) == 3 else 1


def hflip(tens) -> Tensor:
    r"""
        Flip an image horizontally. 
    Args:
        tens (Tensor): Image to be flipped.

    Returns:
        Flipped image.

    """

    tens = np.ascontiguousarray(tens[:, ::-1, ...])
    return to_tensor(tens, override_checks=True)


def vflip(tens) -> Tensor:
    r"""
        Flip an image vertically. 
    Args:
        tens (Tensor): Image to be flipped.

    Returns:
        Flipped image.
        
    """
    tens = np.ascontiguousarray(tens[::-1, ...])
    return to_tensor(tens, override_checks=True)


def hvflip(tens) -> Tensor:
    r"""
        Flip an image both horizontally and vertically. 

    Args:
        tens (Tensor): Image to be flipped.

    Returns:
        Flipped image.
        
    """
    return hflip(vflip(tens))


def rand_flip(tens) -> Tensor: 
    r"""
        Randomly flip an image vertically or horizontally. 

    Args:
        tens (Tensor): Image to be flipped.

    Returns:
        Flipped image.
        
    """
    p = random.uniform(0, 1)

    if p > 0.5:
        return vflip(tens)
    else:
        return hflip(tens)


def transpose(tens) -> Tensor:
    if len(tens.shape) > 2:
        return tens.transpose(1, 0, 2)
    else:
        return tens.transpose(1, 0)


def rotate(tens, angle, rotPoint=None) -> Tensor:
    r"""
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

    height, width = tens.shape[:2]

    # If no rotPoint is specified, we assume the rotation point to be around the centre
    if rotPoint is None:
        rotPoint = (width//2, height//2)

    rotMat = cv.getRotationMatrix2D(rotPoint, angle, scale=1.0)

    tens = cv.warpAffine(tens, rotMat, (width, height))
    return to_tensor(tens, override_checks=True)


def translate(image, x, y) -> Tensor:
    r"""Translates a given image across the x-axis and the y-axis

    Args:
        x (int): shifts the image right (positive) or left (negative)
        y (int): shifts the image down (positive) or up (negative)
    
    Returns:
        The translated image

    """
    transMat = np.float32([[1, 0, x], [0, 1, y]])
    return cv.warpAffine(image, transMat, (image.shape[1], image.shape[0]))


def scale(tens, scale_factor, interpolation='bilinear') -> Tensor:
    interpolation_methods = {
        'nearest': INTER_NEAREST, '0': INTER_NEAREST, 0: INTER_NEAREST, # 0
        'bilinear': INTER_LINEAR, '1': INTER_LINEAR,  1: INTER_LINEAR,  # 1
        'bicubic': INTER_CUBIC,   '2': INTER_CUBIC,   2: INTER_CUBIC,   # 2
        'area': INTER_AREA,       '3': INTER_AREA,    3: INTER_AREA     # 3
    }
    if interpolation not in interpolation_methods:
        raise ValueError('Specify a valid interpolation type - area/nearest/bicubic/bilinear')

    if scale_factor > 1:
        # Neater, more precise
        interpolation = 'bicubic'

    height, width = tens.shape[:2]
    new_height, new_width = int(height * scale_factor), int(width * scale_factor)

    tens = cv.resize(tens, (new_width,new_height), interpolation=interpolation)
    return to_tensor(tens, override_checks=True)


def pad(tens, padding, fill=0, padding_mode='constant') -> Tensor:
    r"""
        Pad the given image on all sides with specified padding mode and fill value.

    Args:
        tens (Tensor): image to be padded.
        padding (int or tuple): Padding on each border. If a single int is provided this 
            is used to pad all borders. If tuple of length 2 is provided this is the padding
            on left/right and top/bottom respectively. If a tuple of length 4 is provided
            this is the padding for the left, top, right and bottom borders
            respectively.
        fill: Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant
        padding_mode: Type of padding. Should be: constant, edge, reflect or symmetric. Default is constant.
            - constant: pads with a constant value, this value is specified with fill
            - edge: pads with the last value on the edge of the image
            - reflect: pads with reflection of image (without repeating the last value on the edge)
                       padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                       will result in [3, 2, 1, 2, 3, 4, 3, 2]
            - symmetric: pads with reflection of image (repeating the last value on the edge)
                         padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                         will result in [2, 1, 1, 2, 3, 4, 4, 3]
                         
    Returns:
        Tensor of shape ``(height, width, channels)``.

    """
    if not isinstance(padding, (tuple, list)):
        raise TypeError('Got inappropriate padding argument')

    if not isinstance(fill, (str, tuple)):
        raise TypeError('Got inappropriate fill argument')

    if not isinstance(padding_mode, str):
        raise TypeError('Got inappropriate padding_mode argument')

    if isinstance(padding, collections.Sequence) and len(padding) not in [2, 4]:
        raise ValueError(f'Padding must be an int or a 2, or 4 element tuple, not a {len(padding)} element tuple')

    assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric'], \
        'Padding mode should be either constant, edge, reflect or symmetric'

    if isinstance(padding, int):
        pad_left = pad_right = pad_top = pad_bottom = padding

    if isinstance(padding, collections.Sequence) and len(padding) == 2:
        pad_left = pad_right = padding[0]
        pad_top = pad_bottom = padding[1]

    if isinstance(padding, collections.Sequence) and len(padding) == 4:
        pad_left = padding[0]
        pad_top = padding[1]
        pad_right = padding[2]
        pad_bottom = padding[3]


        tens = cv.copyMakeBorder(tens,
                                top = pad_top,
                                bottom = pad_bottom,
                                left = pad_left,
                                right = pad_right,
                                borderType = pad_to_str[padding_mode],
                                value = fill)

        return to_tensor(tens, override_checks=True)

                                
def crop(tens, x_min, y_min, x_max, y_max) -> Tensor:
    height, width = tens.shape[:2]
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

    return to_tensor(tens[y_min:y_max, x_min:x_max], override_checks=True)


def center_crop(image, target_size=None) -> Tensor:
    r"""Computes the centre crop of an image using `target_size`

    Args:
        image (Tensor): Valid image Tensor
        target_size (tuple): Size of the centre crop. Must be in the format `(width,height)`
    
    Returns:
        Cropped Centre (Tensor)
    
    Examples::

        >> tens = caer.data.bear() # Standard 640x427 image
        >> cropped = caer.center_crop(tens, target_size=(200,200))
        >> cropped.shape
        (200,200,3)

    """
    return _compute_centre_crop(image, target_size)


def rand_crop(tens, crop_height, crop_width, h_start, w_start) -> Tensor:
    height, width = tens.shape[:2]
    if height < crop_height or width < crop_width:
        raise ValueError(
            "Requested crop size ({crop_height}, {crop_width}) is "
            "larger than the image size ({height}, {width})".format(
                crop_height=crop_height, crop_width=crop_width, height=height, width=width
            )
        )
    x1, y1, x2, y2 = _get_random_crop_coords(height, width, crop_height, crop_width, h_start, w_start)

    return to_tensor(tens[y1:y2, x1:x2], override_checks=True)


def _compute_centre_crop(tens, target_size) -> Tensor:
    _ = _check_target_size(target_size)

    # Getting org height and target
    org_h, org_w = tens.shape[:2]
    target_w, target_h = target_size

    # The following line is actually the right way of accessing height and width of an opencv-specific image (height, width). However for some reason, while the code runs, this is flipped (it now becomes (width,height)). Testing needs to be done to catch this little bug
    # org_h, org_w = tens.shape[:2]


    if target_h > org_h or target_w > org_w:
        raise ValueError('To compute centre crop, target size dimensions must be <= tens dimensions')

    diff_h = (org_h - target_h) // 2
    diff_w = (org_w - target_w ) // 2
    
    # tens[y:y+h, x:x+h]
    return to_tensor(tens[diff_h:diff_h + target_h, diff_w:diff_w + target_w], override_checks=True)


def _get_random_crop_coords(height, width, crop_height, crop_width, h_start, w_start):
    y1 = int((height - crop_height) * h_start)
    y2 = y1 + crop_height
    x1 = int((width - crop_width) * w_start)
    x2 = x1 + crop_width
    return x1, y1, x2, y2


def solarize(tens, threshold=128) -> Tensor:
    r"""
        Invert all pixel values above a threshold.

    Args:
        tens (Tensor): The image to solarize.
        threshold (int): All pixels above this grayscale level are inverted.

    Returns:
        Solarized image (Tensor)
    
    Examples::

        >> tens = caer.data.sunrise()
        >> solarized = caer.solarize(tens, threshold=128)
        >> solarized.shape
        (427,640,3)

    """
    tens = to_tensor(tens, override_checks=True)
    max_val = MAX_VALUES_BY_DTYPE[tens.dtype]

    if tens.dtype == np.dtype("uint8"):
        lut = [(i if i < threshold else max_val - i) for i in range(max_val + 1)]

        prev_shape = tens.shape
        tens = cv.LUT(tens, np.array(lut, dtype=tens.dtype))

        if len(prev_shape) != len(tens.shape):
            tens = np.expand_dims(tens, -1)

        return to_tensor(tens, override_checks=True)

    result_tens = tens.copy()
    cond = tens >= threshold
    result_tens[cond] = max_val - result_tens[cond]
    return to_tensor(result_tens, override_checks=True)


def posterize(tens, bits) -> Tensor:
    r"""Reduce the number of bits for each color channel in the image.

    Args:
        tens (Tensor): Image to posterize.
        bits (int): Number of high bits. Must be in range [0, 8]

    Returns:
        Image with reduced color channels (Tensor)
    
    Examples::

        >> tens = caer.data.sunrise()
        >> posterized = caer.posterize(tens, bits=4)
        >> posterized.shape
        (427,640,3)

    """
    tens = to_tensor(tens, override_checks=True)
    bits = np.uint8(bits)

    if tens.dtype != np.uint8:
        raise TypeError("Image must have uint8 channel type")

    if np.any((bits < 0) | (bits > 8)):
        raise ValueError("bits must be in range [0, 8]")

    if not bits.shape or len(bits) == 1:
        if bits == 0:
            return np.zeros_like(tens)
        if bits == 8:
            return tens.copy()

        lut = np.arange(0, 256, dtype=np.uint8)
        mask = ~np.uint8(2 ** (8 - bits) - 1)
        lut &= mask

        return to_tensor(cv.LUT(tens, lut), override_checks=True)
        

    if not _is_rgb_image(tens):
        raise TypeError("If `bits` is iterable, image must be RGB")

    result_tens = np.empty_like(tens)
    for i, channel_bits in enumerate(bits):
        if channel_bits == 0:
            result_tens[..., i] = np.zeros_like(tens[..., i])
        elif channel_bits == 8:
            result_tens[..., i] = tens[..., i].copy()
        else:
            lut = np.arange(0, 256, dtype=np.uint8)
            mask = ~np.uint8(2 ** (8 - channel_bits) - 1)
            lut &= mask

            result_tens[..., i] = cv.LUT(tens[..., i], lut)

    return to_tensor(result_tens, override_checks=True)


def clip(tens, dtype, maxval) -> Tensor:
    tens = np.clip(tens, 0, maxval).astype(dtype)
    return to_tensor(tens, override_checks=True)


def _equalize_cv(tens, mask=None) -> Tensor:
    if mask is None:
        tens = cv.equalizeHist(tens)
        return to_tensor(tens, override_checks=True)

    histogram = cv.calcHist([tens], [0], mask, [256], (0, 256)).ravel()
    i = 0
    for val in histogram:
        if val > 0:
            break
        i += 1
    i = min(i, 255)

    total = np.sum(histogram)
    if histogram[i] == total:
        tens = np.full_like(tens, i)
        return to_tensor(tens, override_checks=True)

    scale = 255.0 / (total - histogram[i])
    _sum = 0

    lut = np.zeros(256, dtype=np.uint8)
    i += 1
    for i in range(i, len(histogram)):
        _sum += histogram[i]
        lut[i] = clip(round(_sum * scale), np.dtype("uint8"), 255)

    tens = cv.LUT(tens, lut)
    return to_tensor(tens, override_checks=True)


def equalize(tens, mask=None, by_channels=True) -> Tensor:
    r"""Equalize the image histogram.

    Args:
        tens (Tensor): RGB or grayscale image.
        mask (Tensor): An optional mask.  If given, only the pixels selected by the mask are included in the analysis. Maybe 1 channel or 3 channel array.
        by_channels (bool): If True, use equalization by channels separately, else convert image to YCbCr representation and use equalization by `Y` channel.

    Returns:
        Equalized image (Tensor)
    

    Examples::

        >> tens = caer.data.beverages()   
        >> equalized = caer.equalize(tens, mask=None)  
        >> equalized.shape   
        (427,640,3)
    
    """
    if tens.dtype != np.uint8:
        raise TypeError("Image must have uint8 channel type")

    if mask is not None:
        if _is_rgb_image(mask) and _is_gray_image(tens):
            raise ValueError("Wrong mask shape. Image shape: {}. Mask shape: {}".format(tens.shape, mask.shape))

        if not by_channels and not _is_gray_image(mask):
            raise ValueError("When `by_channels=False`, only 1-channel mask is supported. Mask shape: {}".format(mask.shape))

    if mask is not None:
        mask = mask.astype(np.uint8)

    if _is_gray_image(tens): 
        return to_tensor(_equalize_cv(tens, mask), override_checks=True)

    if not by_channels:
        result_tens = cv.cvtColor(tens, cv.COLOR_RGB2YCrCb)
        result_tens[..., 0] = _equalize_cv(result_tens[..., 0], mask)
        tens = cv.cvtColor(result_tens, cv.COLOR_YCrCb2RGB) 
        return to_tensor(tens, override_checks=True)

    result_tens = np.empty_like(tens)
    for i in range(3):
        if mask is None:
            _mask = None
        elif _is_gray_image(mask):
            _mask = mask
        else:
            _mask = mask[..., i]

        result_tens[..., i] = _equalize_cv(tens[..., i], _mask)

    return to_tensor(result_tens, override_checks=True)