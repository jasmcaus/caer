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
import cv2 as cv 
import random
import math 

from .functional import (
    is_list, 
    is_numeric, 
    is_numeric_list_or_tuple, 
    is_tuple, 
    _is_numpy_array,
    _hls,
    _rgb,
    _snow_process,
    _generate_random_lines,
    _rain_process,
    _generate_random_blur_coordinates,
    _add_blur,
    _gravel_process,
    _add_sun_flare_line,
    _add_sun_process,
    _apply_motion_blur,
    _autumn_process,
    _exposure_process,
    _shadow_process
)

from ..color import (
    hls_to_bgr,
    hls_to_rgb
)

__all__ = [
    'change_light',
    'darken',
    'brighten',
    'random_brightness',
    'add_snow',
    'add_rain',
    'add_fog',
    'add_gravel',
    'add_sun_flare',
    'add_motion_blur',
    'add_autumn',
    'add_shadow',
    'correct_exposure',
    'augment_random'
]


def change_light(img, coeff, rgb=True):
    r"""
        Change the lighting of an image.

    Args:
        img (ndarray) : Any regular BGR/RGB image.
        coeff (int): Coefficient value.

            If ``coeff < 1``, the image is darkened.

            If ``coeff > 1``, the image is lightened.
        rgb (bool): Operate on RGB images. Default: True.
    
    Returns:
        Array of shape ``(height, width, channels)``.

    Examples::

        >> img = caer.data.sunrise(rgb=True)
        >> filtered = caer.filters.change_light(img, coeff=1.4, rgb=True)
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
        return hls_to_rgb(img)
    else:
        return hls_to_bgr(img)


def brighten(img, coeff=-1, rgb=True):
    r"""
        Brighten an image.

    Args:
        img (ndarray) : Any regular BGR/RGB image.
        coeff (int): Coefficient value.
        rgb (bool): Operate on RGB images. Default: True.
    
    Returns:
        Array of shape ``(height, width, channels)``.

    Examples::

        >> img = caer.data.sunrise(rgb=True)
        >> filtered = caer.filters.brighten(img, coeff=-1, rgb=True)
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

    return change_light(img, coeff_t, rgb=rgb)


def darken(img, darkness_coeff = -1, rgb=True):
    r"""
        Darken an image.

    Args:
        img (ndarray) : Any regular BGR/RGB image.
        darkness_coeff (int): Coefficient value.
        rgb (bool): Operate on RGB images. Default: True.
    
    Returns:
        Array of shape ``(height, width, channels)``.

    Examples::

        >> img = caer.data.sunrise(rgb=True)
        >> filtered = caer.filters.darken(img, coeff=-1, rgb=True)
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

    return change_light(img, darkness_coeff_t, rgb=True)


def random_brightness(img, rgb=True):
    r"""
        Add random brightness to an image.

    Args:
        img (ndarray) : Any regular BGR/RGB image.
        rgb (bool): Operate on RGB images. Default: True.
    
    Returns:
        Array of shape ``(height, width, channels)``.

    Examples::

        >> img = caer.data.sunrise(rgb=True)
        >> filtered = caer.filters.random_brightness(img, rgb=True)
        >> filtered
        (427, 640, 3)

    """
    rand_br_coeff = 2 * np.random.uniform(0, 1) # Generates a value between 0.0 and 2.0
    return change_light(img, rand_br_coeff, rgb=rgb)


def add_snow(img, snow_coeff=-1, rgb=True):
    r"""
        Simulate snowy conditions on an image.

    Args:
        img (ndarray) : Any regular BGR/RGB image.
        snow_coeff (int): Coefficient value.
        rgb (bool): Operate on RGB images. Default: True.
    
    Returns:
        Array of shape ``(height, width, channels)``.

    Examples::

        >> img = caer.data.sunrise(rgb=True)
        >> filtered = caer.filters.add_snow(img, snow_coeff=-1, rgb=True)
        >> filtered
        (427, 640, 3)

    """
    if snow_coeff != -1:
        if snow_coeff < 0.0 or snow_coeff > 1.0:
            raise ValueError('Snow coeff must only be between 0 and 1')
    else:
        snow_coeff=random.uniform(0, 1)

    snow_coeff*=255/2
    snow_coeff+=255/3

    return _snow_process(img, snow_coeff, rgb=rgb)


# Rain_type = 'drizzle', 'heavy', 'torrential'
def add_rain(img, slant=-1, drop_length=20, drop_width=1, drop_color=(200,200,200), rain_type='None', rgb=True): ## (200,200,200) is a shade of gray
    r"""
        Simulate rainy conditions on an image.

    Args:
        img (ndarray) : Any regular BGR/RGB image.
        slant (int): Slant value.
        drop_length (int): Length of the raindrop.
        drop_width (int): Width of the raindrop.
        drop_color (tuple): Color of the raindrop.
        rain_type (str): Type of rain. Can be either 'drizzle', 'heavy' or 'torrential'.
        rgb (bool): Operate on RGB images. Default: True.
    
    Returns:
        Array of shape ``(height, width, channels)``.

    Examples::

        >> img = caer.data.sunrise(rgb=True)
        >> filtered = caer.filters.add_rain(img, rgb=True)
        >> filtered
        (427, 640, 3)

    """
    
    slant_extreme = slant
    if not(is_numeric(slant_extreme) and (slant_extreme >=-20 and slant_extreme <= 20) or slant_extreme==-1):
        raise ValueError('Numeric value must be between -20 and 20')

    if not(is_numeric(drop_width) and drop_width>=1 and drop_width<=5):
        raise ValueError('Width must be between 1 and 5')

    if not(is_numeric(drop_length) and drop_length>=0 and drop_length<=100):
        raise ValueError('Length must be between 0 and 100')

    imshape = img.shape
    if slant_extreme == -1:
        slant= np.random.randint(-10,10) # generate random slant if no slant value is given

    rain_drops, drop_length= _generate_random_lines(imshape, slant, drop_length, rain_type)
    return _rain_process(img, slant_extreme, drop_length, drop_color, drop_width, rain_drops)


def add_fog(img, fog_coeff=-1, rgb=True):
    r"""
        Simulate foggy conditions on an image.

    Args:
        img (ndarray) : Any regular BGR/RGB image.
        fog_coeff (int): Coefficient value.
        rgb (bool): Operate on RGB images. Default: True.
    
    Returns:
        Array of shape ``(height, width, channels)``.

    Examples::

        >> img = caer.data.sunrise(rgb=True)
        >> filtered = caer.filters.add_fog(img, fog_coeff=-1, rgb=True)
        >> filtered
        (427, 640, 3)

    """
    if fog_coeff != -1:
        if fog_coeff < 0.0 or fog_coeff > 1.0:
            raise ValueError('Fog coefficient must be between 0 and 1')

    imshape = img.shape

    if fog_coeff == -1:
        fog_coeff_t = random.uniform(0.3,1)
    else:
        fog_coeff_t = fog_coeff

    hw = int(imshape[1]//3*fog_coeff_t)
    haze_list = _generate_random_blur_coordinates(imshape,hw)
    for haze_points in haze_list: 
        img = _add_blur(img, haze_points[0], haze_points[1], hw, fog_coeff_t) 

    img = cv.blur(img, (hw//10, hw//10))
    
    return _rgb(img, rgb=rgb)


def add_gravel(img, rectangular_roi=(-1,-1,-1,-1), num_patches=8, rgb=True):
    r"""
        Simulate gravelly conditions on an image.

    Args:
        img (ndarray) : Any regular BGR/RGB image.
        rectangular_roi (tuple): Rectanglar co-ordinates of the intended region of interest. Default: (-1,-1,-1,-1).
        num_patches (int): Number of patches to operate on.
        rgb (bool): Operate on RGB images. Default: True.
    
    Returns:
        Array of shape ``(height, width, channels)``.

    Examples::

        >> img = caer.data.sunrise(rgb=True)
        >> filtered = caer.filters.add_gravel(img, rgb=True)
        >> filtered
        (427, 640, 3)

    """
    if is_tuple(rectangular_roi) and is_numeric_list_or_tuple(rectangular_roi) and len(rectangular_roi)==4:
        x1 = rectangular_roi[0]
        y1 = rectangular_roi[1]
        x2 = rectangular_roi[2]
        y2 = rectangular_roi[3]
    else:
        raise ValueError('Rectangular ROI dimensions are invalid.')

    if rectangular_roi == (-1,-1,-1,-1):
        if _is_numpy_array(img):
            x1 = 0
            y1 = int(img.shape[0]*3/4)
            x2 = img.shape[1]
            y2 = img.shape[0]
        else:
            x1 = 0
            y1 = int(img[0].shape[0]*3/4)
            x2 = img[0].shape[1]
            y2 = img[0].shape[0]

    elif x1 == -1 or y1 == -1 or x2 == -1 or y2 == -1 or x2 <= x1 or y2 <= y1:
        raise ValueError('Rectangular ROI dimensions are invalid.')

    return _gravel_process(img, x1, x2, y1, y2, num_patches, rgb=rgb)


def add_sun_flare(img, flare_center=-1, angle=-1, num_flare_circles=8, src_radius=400, src_color=(255,255,255)):
    r"""
        Add a source of light (flare) on an specific region of an image.

    Args:
        img (ndarray) : Any regular BGR/RGB image.
        flare_center (int): Center of the flare. Default: -1.
        angle (int): Angle of the flare. Default: -1
        num_flare_circles (int): Number of flare circles to operate on.
        src_radius (int): Intended size of the flare
        src_color (tuple): Intended source color of the flare
    
    Returns:
        Array of shape ``(height, width, channels)``.

    Examples::

        >> img = caer.data.sunrise(rgb=True)
        >> filtered = caer.filters.add_sun_flare(img)
        >> filtered
        (427, 640, 3)

    """
    if angle != -1:
        angle = angle % (2*math.pi)

    if not(num_flare_circles >= 0 and num_flare_circles <= 20):
        raise ValueError('Numeric value must be between 0 and 20')

    imshape = img.shape
    if angle == -1:
        angle_t = random.uniform(0, 2*math.pi)
        if angle_t == math.pi/2:
            angle_t = 0
    else:
        angle_t = angle

    if flare_center == -1:
        flare_center_t = (random.randint(0,imshape[1]), random.randint(0,imshape[0]//2))
    else:
        flare_center_t = flare_center

    x, y = _add_sun_flare_line(flare_center_t, angle_t, imshape)

    return _add_sun_process(img, num_flare_circles, flare_center_t, src_radius, x, y, src_color)


def add_motion_blur(img, speed_coeff=-1):
    r"""
        Simulate motion-blur conditions on an image.

    Args:
        img (ndarray) : Any regular BGR/RGB image.
        speed_coeff (int, float): Speed coefficient. Value must be between 0 and 1.
    
    Returns:
        Array of shape ``(height, width, channels)``.

    Examples::

        >> img = caer.data.sunrise(rgb=True)
        >> filtered = caer.filters.add_motion_blur(img, speed_coeff=-1)
        >> filtered
        (427, 640, 3)

    """
    if speed_coeff != -1:
        if speed_coeff < 0.0 or speed_coeff > 1.0:
            raise ValueError('Speed coefficient must be between 0 and 1')

    if speed_coeff == -1:
        count_t = int(15 * random.uniform(0, 1))
    else:
        count_t = int(15 * speed_coeff)

    return _apply_motion_blur(img, count_t)


def add_autumn(img, rgb=True):
    r"""
        Simulate autumn conditions on an image.

    Args:
        img (ndarray) : Any regular BGR/RGB image.
        rgb (bool): Operate on RGB images. Default: True.
    
    Returns:
        Array of shape ``(height, width, channels)``.

    Examples::

        >> img = caer.data.sunrise(rgb=True)
        >> filtered = caer.filters.add_autumn(img, rgb=True)
        >> filtered
        (427, 640, 3)

    """
    return _autumn_process(img, rgb=rgb)


def correct_exposure(img, rgb=True):
    r"""
        Correct the exposure of an image.

    Args:
        img (ndarray) : Any regular BGR/RGB image.
        rgb (bool): Operate on RGB images. Default: True.
    
    Returns:
        Array of shape ``(height, width, channels)``.

    Examples::

        >> img = caer.data.sunrise(rgb=True)
        >> filtered = caer.filters.correct_exposure(img, rgb=True)
        >> filtered
        (427, 640, 3)

    """
    return _exposure_process(img, rgb=rgb)


def augment_random(img, aug_types='', volume='expand' ):
    aug_types_all = ['random_brightness','add_shadow','add_snow','add_rain','add_fog','add_gravel','add_sun_flare','add_motion_blur','add_autumn','random_flip']

    if aug_types=='':
        aug_types=aug_types_all
    output=[]

    if not is_list(aug_types):
        raise ValueError('`aug_types` should be a list of function names (str)')
    
    if volume == 'expand':
        for aug_type in aug_types:
            if not(aug_type in aug_types_all):
                raise ValueError('Incorrect augmentation function defined')

            command = aug_type + '(img)'
            result = eval(command)

            output.append(result)

    elif volume == 'same':
        for aug_type in aug_types:
            if not (aug_type in aug_types_all):
                raise ValueError('Incorrect augmentation function defined')

        selected_aug = aug_types[random.randint(0, len(aug_types)-1)]
        command = selected_aug+'(img)'
        output = eval(command)

    else: 
        raise ValueError('volume type can only be "same" or "expand"')

    return output


## ROI:(top-left x1,y1, bottom-right x2,y2), shadow_dimension=no. of sides of polygon generated
def add_shadow(img, num_shadows=1, rectangular_roi=(-1,-1,-1,-1), shadow_dimension=5, rgb=True): 
    r"""
        Simulate shadowy conditions on an image.

    Args:
        img (ndarray) : Any regular BGR/RGB image.
        num_shadows (int): Number of shadows to work with. Value must be between 1 and 10.
        rectangular_roi (tuple): Rectanglar co-ordinates of the intended region of interest. Default: (-1,-1,-1,-1).
        shadow_dimensions (int): Number of shadow dimensions. Value must be > 3. 
        rgb (bool): Operate on RGB images. Default: True.
    
    Returns:
        Array of shape ``(height, width, channels)``.

    Examples::

        >> img = caer.data.sunrise(rgb=True)
        >> filtered = caer.filters.add_shadow(img, rgb=True)
        >> filtered
        (427, 640, 3)

    """
    if not(is_numeric(num_shadows) and num_shadows >= 1 and num_shadows <= 10):
        raise ValueError('Only 1-10 shadows can be introduced in an image')

    if not(is_numeric(shadow_dimension) and shadow_dimension >= 3 and shadow_dimension <= 10):
        raise ValueError('Polygons with dimensions < 3 don\'t exist and take time to plot')

    if is_tuple(rectangular_roi) and is_numeric_list_or_tuple(rectangular_roi) and len(rectangular_roi)==4:
        x1 = rectangular_roi[0]
        y1 = rectangular_roi[1]
        x2 = rectangular_roi[2]
        y2 = rectangular_roi[3]
    else:
        raise ValueError('Rectangular ROI dimensions are not valid')

    if rectangular_roi==(-1,-1,-1,-1):
        x1 = 0
        
        if(_is_numpy_array(img)):
            y1 = img.shape[0] // 2
            x2 = img.shape[1]
            y2 = img.shape[0]
        else:
            y1 = img[0].shape[0] // 2
            x2 = img[0].shape[1]
            y2 = img[0].shape[0]

    elif x1 == -1 or y1 == -1 or x2 == -1 or y2 == -1 or x2 <= x1 or y2 <= y1:
        raise ValueError('Rectangular ROI dimensions are not valid')

    return _shadow_process(img,num_shadows, x1, y1, x2, y2, shadow_dimension, rgb=rgb)