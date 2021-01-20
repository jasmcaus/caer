#    _____           ______  _____ 
#  / ____/    /\    |  ____ |  __ \
# | |        /  \   | |__   | |__) | Caer - Modern Computer Vision
# | |       / /\ \  |  __|  |  _  /  Languages: Python, C, C++
# | |___   / ____ \ | |____ | | \ \  http://github.com/jasmcaus/caer
#  \_____\/_/    \_ \______ |_|  \_\

# Licensed under the MIT License <http://opensource.org/licenses/MIT>
# SPDX-License-Identifier: MIT
# Copyright (c) 2020 The Caer Authors <http://github.com/jasmcaus>

#pylint:disable=unused-argument,unused-variable,eval-used

import numpy as np 
import random
import cv2 as cv 
import math

from ..adorad import Tensor, is_tensor

from .functional import (
    is_list,
    _hls,
    _exposure_process
)

from ..color import (
    to_bgr,
    to_rgb
)

__all__ = [
    'adjust_brightness',
    'adjust_contrast',
    'adjust_hue',
    'adjust_saturation',
    'adjust_gamma',
    'affine',
    'darken',
    'brighten',
    'random_brightness',
    'correct_exposure',
    'augment_random'
]


def adjust_brightness(img, coeff, rgb=True) -> Tensor:
    r"""
        Adjust the brightness of an image.

    Args:
        img (Tensor) : Any regular BGR/RGB image.
        coeff (int): Coefficient value.
            - ``coeff < 1``, the image is darkened.
            - ``coeff = 1``, the image is unchanged.
            - ``coeff > 1``, the image is lightened.
        rgb (bool): Operate on RGB images. Default: True.
    
    Returns:
        Array of shape ``(height, width, channels)``.

    Examples::

        >> img = caer.data.sunrise(rgb=True)
        >> filtered = caer.transforms.adjust_brightness(img, coeff=1.4, rgb=True)
        >> filtered
        (427, 640, 3)

    """
    img = _hls(img, rgb=rgb)

    img = np.array(img, dtype=np.float64) 
    img[:,:,1] = img[:,:,1]*coeff ## scale pixel values up or down for channel 1 (for lightness)

    if coeff > 1:
        img[:,:,1][img[:,:,1]>255]  = 255 # Set all values > 255 to 255
    else:
        img[:,:,1][img[:,:,1]<0]=0

    img = np.array(img, dtype=np.uint8)

    if rgb:
        return to_rgb(img)
    else:
        return to_bgr(img)


def brighten(img, coeff=-1, rgb=True) -> Tensor:
    r"""
        Brighten an image.

    Args:
        img (Tensor) : Any regular BGR/RGB image.
        coeff (int): Coefficient value.
        rgb (bool): Operate on RGB images. Default: True.
    
    Returns:
        Array of shape ``(height, width, channels)``.

    Examples::

        >> img = caer.data.sunrise(rgb=True)
        >> filtered = caer.transforms.brighten(img, coeff=-1, rgb=True)
        >> filtered
        (427, 640, 3)

    """
    if coeff !=-1:
        if coeff < 0.0 or coeff > 1.0:
            raise ValueError('Brightness coefficient can only be between 0.0 and 1.0')

    if coeff == -1:
        coeff_t = 1 + random.uniform(0, 1) # coeff between 1.0 and 1.5
    else:
        coeff_t = 1 + coeff  # coeff between 1.0 and 2.0

    return adjust_brightness(img, coeff_t, rgb=rgb)


def darken(img, darkness_coeff = -1, rgb=True) -> Tensor:
    r"""
        Darken an image.

    Args:
        img (Tensor) : Any regular BGR/RGB image.
        darkness_coeff (int): Coefficient value.
        rgb (bool): Operate on RGB images. Default: True.
    
    Returns:
        Array of shape ``(height, width, channels)``.

    Examples::

        >> img = caer.data.sunrise(rgb=True)
        >> filtered = caer.transforms.darken(img, coeff=-1, rgb=True)
        >> filtered
        (427, 640, 3)

    """
    if darkness_coeff != -1:
        if darkness_coeff < 0.0 or darkness_coeff > 1.0:
            raise ValueError('Darkness coeff must only be between 0.0 and 1.0') 

    if darkness_coeff == -1:
        darkness_coeff_t = 1 - random.uniform(0, 1)
    else:
        darkness_coeff_t = 1 - darkness_coeff  

    return adjust_brightness(img, darkness_coeff_t, rgb=True)


def random_brightness(img, rgb=True) -> Tensor:
    r"""
        Add random brightness to an image.

    Args:
        img (Tensor) : Any regular BGR/RGB image.
        rgb (bool): Operate on RGB images. Default: True.
    
    Returns:
        Array of shape ``(height, width, channels)``.

    Examples::

        >> img = caer.data.sunrise(rgb=True)
        >> filtered = caer.transforms.random_brightness(img, rgb=True)
        >> filtered
        (427, 640, 3)

    """
    rand_br_coeff = 2 * np.random.uniform(0, 1) # Generates a value between 0.0 and 2.0
    return adjust_brightness(img, rand_br_coeff, rgb=rgb)


def adjust_contrast(img, contrast_factor) -> Tensor:
    """
        Adjust contrast of an image.

    Args:
        img (Tensor): Any valid BGR/RGB image.
        contrast_factor (float): How much to adjust the contrast. Can be any
            non negative number. 0 gives a solid gray image, 1 gives the
            original image while 2 increases the contrast by a factor of 2.
    Returns:
        numpy Tensor: Contrast adjusted image.
    """
    # It's much faster to use the LUT construction because you have to change dtypes multiple times
    if not is_tensor(img):
        raise TypeError('Expected Numpy Tensor. Got {}'.format(type(img)))

    table = np.array([(i - 74) * contrast_factor + 74
                      for i in range(0, 256)]).clip(0, 255).astype('uint8')

    # enhancer = ImageEnhance.Contrast(img)
    # img = enhancer.enhance(contrast_factor)
    return cv.LUT(img, table)


def adjust_saturation(img, saturation_factor) -> Tensor:
    """Adjust color saturation of an image.
    Args:
        img (numpy Tensor): Any valid BGR/RGB image.
        saturation_factor (float):  How much to adjust the saturation. 0 will
            give a black and white image, 1 will give the original image while
            2 will enhance the saturation by a factor of 2.

    Returns:
        Saturation-adjusted image.
    
    Examples::

        >> img = caer.data.sunrise(rgb=True)
        >> filtered = caer.transforms.adjust_saturation(img, saturation_factor=1.5, rgb=True)
        >> filtered
        (427, 640, 3)

    """
    # ~10ms slower than PIL!
    if not is_tensor(img):
        raise TypeError('Expected Numpy Tensor. Got {}'.format(type(img)))

    try:
        from PIL import Image, ImageEnhance
    except ImportError:
        raise ImportError('Pillow must be installed to use this ``caer.color.adjust_saturation()``.')

    img = Image.fromarray(img)
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(saturation_factor)

    return np.array(img)


def adjust_hue(img, hue_factor) -> Tensor:
    r"""
        Adjust hue of an image.

        The image hue is adjusted by converting the image to HSV and cyclically shifting the intensities in the hue channel (H). The image is then converted back to original image mode.
        
        See `Hue`_ for more details.
        .. _Hue: https://en.wikipedia.org/wiki/Hue

    Args:
        img (Tensor): Any valid BGR/RGB image.
        hue_factor (float):  How much to shift the hue channel. Should be in the range [-0.5, 0.5]. 
            0.5 and -0.5 give complete reversal of hue channel in HSV space in positive and negative direction respectively.
            0 means no shift. Therefore, both -0.5 and 0.5 will give an image with complementary colors while 0 gives the original image.
            
    Returns:
        Hue adjusted image.

    Examples::

        >> img = caer.data.sunrise(rgb=True)
        >> filtered = caer.transforms.adjust_hue(img, hue_factor=1, rgb=True)
        >> filtered
        (427, 640, 3)

    """

    if not (-0.5 <= hue_factor <= 0.5):
        raise ValueError('`hue_factor` is not in [-0.5, 0.5].')

    if not is_tensor(img):
        raise TypeError('Expected Numpy Tensor. Got {}'.format(type(img)))

    try:
        from PIL import Image
    except ImportError:
        raise ImportError('Pillow must be installed to use this ``caer.color.adjust_hue()``.')

    img = Image.fromarray(img)
    input_mode = img.mode
    if input_mode in {'L', '1', 'I', 'F'}:
        return np.array(img)

    h, s, v = img.convert('HSV').split()

    np_h = np.array(h, dtype=np.uint8)

    # uint8 addition take cares of rotation across boundaries
    with np.errstate(over='ignore'):
        np_h += np.uint8(hue_factor * 255)
    h = Image.fromarray(np_h, 'L')

    img = Image.merge('HSV', (h, s, v)).convert(input_mode)
    return np.array(img)


def adjust_gamma(img, gamma, gain=1) -> Tensor:
    r"""
        Perform gamma correction on an image.

        Also known as Power Law Transform. Intensities in RGB mode are adjusted
        based on the following equation:
        .. math::
            I_{\text{out}} = 255 \times \text{gain} \times \left(\frac{I_{\text{in}}}{255}\right)^{\gamma}
        See `Gamma Correction`_ for more details.
        .. _Gamma Correction: https://en.wikipedia.org/wiki/Gamma_correction

    Args:
        img (Tensor): Any valid BGR/RGB image.
        gamma (float): Non negative real number, same as :math:`\gamma` in the equation.
            gamma larger than 1 make the shadows darker,
            while gamma smaller than 1 make dark regions lighter.
        gain (float): The constant multiplier.

    Examples::

        >> img = caer.data.sunrise(rgb=True)
        >> filtered = caer.transforms.adjust_gamma(img, gamma=1.5, rgb=True)
        >> filtered
        (427, 640, 3)

    """
    if not is_tensor(img):
        raise TypeError('Expected Numpy Tensor. Got {}'.format(type(img)))

    if gamma < 0:
        raise ValueError('Gamma should be a non-negative real number')

    # from here
    # https://stackoverflow.com/questions/33322488/how-to-change-image-illumination-in-opencv-python/41061351
    table = np.array([((i / 255.0)**gamma) * 255 * gain
                      for i in np.arange(0, 256)]).astype('uint8')

    return cv.LUT(img, table)


def _get_affine_matrix(center, angle, translate, scale, shear) -> Tensor:
    # Helper method to compute matrix for affine transformation
    # We need compute affine transformation matrix: M = T * C * RSS * C^-1
    # where T is translation matrix: [1, 0, tx | 0, 1, ty | 0, 0, 1]
    #       C is translation matrix to keep center: [1, 0, cx | 0, 1, cy | 0, 0, 1]
    #       RSS is rotation with scale and shear matrix
    #       RSS(a, scale, shear) = [ cos(a)*scale    -sin(a + shear)*scale     0]
    #                              [ sin(a)*scale    cos(a + shear)*scale     0]
    #                              [     0                  0          1]

    angle = math.radians(angle)
    shear = math.radians(shear)
    # scale = 1.0 / scale

    T = np.array([[1, 0, translate[0]], [0, 1, translate[1]], [0, 0, 1]])
    C = np.array([[1, 0, center[0]], [0, 1, center[1]], [0, 0, 1]])
    RSS = np.array(
        [[math.cos(angle) * scale, -math.sin(angle + shear) * scale, 0],
         [math.sin(angle) * scale,
          math.cos(angle + shear) * scale, 0], [0, 0, 1]])
    matrix = T @ C @ RSS @ np.linalg.inv(C)

    return matrix[:2, :]


def affine(img,
           angle,
           translate,
           scale,
           shear,
           interpolation='bilinear',
           mode=0,
           fillcolor=0) -> Tensor:
    """
        Apply affine transformation on the image keeping image center invariant.

    Args:
        img (Tensor): Any valid BGR/RGB image.
        angle (float or int): Rotation angle in degrees between -180 and 180, clockwise direction.
        translate (list or tuple of integers): Horizontal and vertical translations (post-rotation translation)
        scale (float): Overall scale
        shear (float): Shear angle value in degrees between -180 to 180, clockwise direction.
        interpolation (int, str): Interpolation to use for resizing. Defaults to `'bilinear'`. 
                Supports `'bilinear'`, `'bicubic'`, `'area'`, `'nearest'`.
        mode (int, str): Method for filling in border regions. 
                Defaults to ``constant`` meaning areas outside the image are filled with a value (val, default 0). 
                Supports ``'replicate'``, ``'reflect'``, ``'reflect-101'``.
        val (int): Optional fill color for the area outside the transform in the output image. Default: 0
    """

    if not is_tensor(img):
        raise TypeError('Expected Numpy Tensor. Got {}'.format(type(img)))

    assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
        'Argument translate should be a list or tuple of length 2'

    assert scale > 0.0, 'Argument scale should be positive'

    interpolation_methods = {
        'nearest': 0,  '0': 0,  0: 0, # 0
        'bilinear': 1, '1': 1,  1: 1, # 1
        'bicubic': 2,  '2': 2,  2: 2, # 2
        'area': 3,     '3': 3,  3: 3 # 3
    }
    border_methods = {
        'constant': 0,    '0': 0, 0: 0, # 0
        'replicate': 1,   '1': 1, 1: 1, # 1
        'reflect': 2,     '2': 2, 2: 2, # 2
        'reflect-101': 4, '4': 4, 4: 4 # 4
    }

    if interpolation not in interpolation_methods:
        raise ValueError('Specify a valid interpolation type - area/nearest/bicubic/bilinear')

    if mode not in border_methods:
        raise ValueError('Specify a valid border type - constant/replicate/reflect/reflect-101')

    output_size = img.shape[0:2]
    center = (img.shape[1] * 0.5 + 0.5, img.shape[0] * 0.5 + 0.5)
    matrix = _get_affine_matrix(center, angle, translate, scale, shear)

    return cv.warpAffine(img,
                            matrix,
                            output_size[::-1],
                            interpolation,
                            borderMode=mode,
                            borderValue=fillcolor)


def correct_exposure(img, rgb=True) -> Tensor:
    r"""
        Correct the exposure of an image.

    Args:
        img (Tensor) : Any regular BGR/RGB image.
        rgb (bool): Operate on RGB images. Default: True.
    
    Returns:
        Array of shape ``(height, width, channels)``.

    Examples::

        >> img = caer.data.sunrise(rgb=True)
        >> filtered = caer.transforms.correct_exposure(img, rgb=True)
        >> filtered
        (427, 640, 3)

    """
    return _exposure_process(img, rgb=rgb)


def augment_random(img, aug_types='', volume='expand' ) -> Tensor:
    aug_types_all = ['random_brightness','add_shadow','add_snow','add_rain','add_fog','add_gravel','add_sun_flare','add_motion_blur','add_autumn','random_flip']

    if aug_types=='':
        aug_types=aug_types_all
    output=[]

    if not is_list(aug_types):
        raise ValueError('`aug_types` should be a list of function names (str)')
    
    if volume == 'expand':
        for aug_type in aug_types:
            if not(aug_type in aug_types_all):
                raise ValueError('Incorrect transformation function defined')

            command = aug_type + '(img)'
            result = eval(command)

            output.append(result)

    elif volume == 'same':
        for aug_type in aug_types:
            if not (aug_type in aug_types_all):
                raise ValueError('Incorrect transformation function defined')

        selected_aug = aug_types[random.randint(0, len(aug_types)-1)]
        command = selected_aug+'(img)'
        output = eval(command)

    else: 
        raise ValueError('volume type can only be "same" or "expand"')

    return output