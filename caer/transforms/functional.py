#    _____           ______  _____ 
#  / ____/    /\    |  ____ |  __ \
# | |        /  \   | |__   | |__) | Caer - Modern Computer Vision
# | |       / /\ \  |  __|  |  _  /  Languages: Python, C, C++, Cuda
# | |___   / ____ \ | |____ | | \ \  http://github.com/jasmcaus/caer
#  \_____\/_/    \_ \______ |_|  \_\

# Licensed under the MIT License <http://opensource.org/licenses/MIT>
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021 The Caer Authors <http://github.com/jasmcaus>

#pylint:disable=unused-variable

import cv2 as cv 
import random 
import math 
import numpy as np 

from ..adorad import Tensor, to_tensor

from ..color import (
    to_hls, 
    to_rgb,
    to_bgr,
    to_hsv,
    to_gray,
    to_lab
)

def _hls(tens) -> Tensor:
    tens = to_tensor(tens, enforce_tensor=True)
    return to_hls(tens)


def _bgr(tens) -> Tensor:
    tens = to_tensor(tens, enforce_tensor=True)
    return to_bgr(tens)


def _rgb(tens) -> Tensor:
    tens = to_tensor(tens, enforce_tensor=True)
    return to_rgb(tens)


def _hue(tens) -> Tensor:
    return _hls(tens)[:,:,0]


def _get_tens_size(tens) -> Tensor:
    r"""
        Returns image size as (width, height)
    """
    h, w = tens.shape[:2]

    return (w, h)


def _get_num_channels(tens) -> int:
    r"""
        We assume only images of 1 and 3 channels
    """
    if len(tens.shape) == 3 and tens.shape[2] == 3:
        return 3
    
    else:
        return 1


def is_tuple(x) -> bool:
    return isinstance(x, tuple)


def is_list(x):
    return isinstance(x, list)


def is_numeric(x):
    return isinstance(x, int)


def is_numeric_list_or_tuple(x):
    for i in x:
        if not is_numeric(i):
            return False
    return True


def _snow_process(tens, snow_coeff) -> Tensor:
    tens = _hls(tens)
    cspace = tens.cspace 

    tens = np.array(tens, dtype=np.float64) 

    brightness_coefficient = 2.5 

    ## scale pixel values up for channel 1 (lightness)
    tens[:,:,1][tens[:,:,1]<snow_coeff] = tens[:,:,1][tens[:,:,1]<snow_coeff]*brightness_coefficient 
    tens[:,:,1][tens[:,:,1]>255]  = 255 ##Sets all values above 255 to 255

    tens = np.array(tens, dtype=np.uint8)

    return to_tensor(tens, cspace=cspace)


def _generate_random_lines(imshape, slant, drop_length, rain_type):
    drops=[]
    area=imshape[0] * imshape[1]
    num_drops = area//600

    if rain_type.lower() == 'drizzle':
        num_drops = area//770
        drop_length = 10

    elif rain_type.lower() == 'heavy':
        drop_length = 30

    elif rain_type.lower() == 'torrential':
        num_drops = area//500
        drop_length = 60

    for i in range(num_drops) : ## If you need heavier rain, try increasing this
        if slant < 0:
            x = np.random.randint(slant, imshape[1])
        else:
            x = np.random.randint(0, imshape[1] - slant)

        y = np.random.randint(0, imshape[0] - drop_length)
        drops.append((x,y))

    return drops, drop_length


def _rain_process(tens, slant, drop_length, drop_color, drop_width, rain_drops) -> Tensor:
    tens = to_tensor(tens, enforce_tensor=True)
    tens_t = tens.clone()

    for rain_drop in rain_drops:
        cv.line(tens_t, (rain_drop[0],rain_drop[1]), (rain_drop[0]+slant,rain_drop[1]+drop_length), drop_color, drop_width)

    tens = cv.blur(tens_t,(7,7)) ## Rainy views are blurred
    brightness_coefficient = 0.7 ## Rainy days are usually shady 

    tens = _hls(tens) ## Conversion to hls
    cspace = tens.cspace 

    tens[:,:,1] = tens[:,:,1]*brightness_coefficient ## scale pixel values down for channel 1 (Lightness)

    return to_tensor(tens, cspace=cspace)


def _add_blur(tens, x, y, hw, fog_coeff) -> Tensor:
    tens = to_tensor(tens, override_checks=True)

    overlay = tens.clone()
    output = tens.clone()

    alpha = 0.08 * fog_coeff
    rad = hw // 2
    point = (x+hw//2, y+hw//2)
    cv.circle(overlay, point, int(rad), (255,255,255), -1)
    cv.addWeighted(overlay, alpha, output, 1 -alpha , 0, output)

    return to_tensor(output, override_checks=True)


def _generate_random_blur_coordinates(imshape, hw):
    blur_points=[]
    midx = imshape[1]//2-2*hw
    midy = imshape[0]//2-hw
    index=1

    while midx >- hw or midy >- hw:
        for i in range(hw//10*index):
            x= np.random.randint(midx, imshape[1]-midx-hw)
            y= np.random.randint(midy, imshape[0]-midy-hw)
            blur_points.append((x,y))
        midx -= 3*hw*imshape[1]//sum(imshape)
        midy -= 3*hw*imshape[0]//sum(imshape)
        index += 1
    
    return blur_points


def _generate_gravel_patch(rectangular_roi):
    x1 = rectangular_roi[0]
    y1 = rectangular_roi[1]
    x2 = rectangular_roi[2]
    y2 = rectangular_roi[3] 
    gravels = []
    area = abs((x2-x1)*(y2-y1))

    for i in range((int(area//10))):
        x= np.random.randint(x1,x2)
        y= np.random.randint(y1,y2)
        gravels.append((x,y))

    return gravels


def _gravel_process(tens, x1, x2, y1, y2, num_patches):
    tens = to_tensor(tens, enforce_tensor=True)
    
    x = tens.shape[1]
    y = tens.shape[0]
    rectangular_roi_default = []

    for i in range(num_patches):
        xx1 = random.randint(x1, x2)
        xx2 = random.randint(x1, xx1)
        yy1 = random.randint(y1, y2)
        yy2 = random.randint(y1, yy1)
        rectangular_roi_default.append((xx2, yy2, min(xx1, xx2+200), min(yy1, yy2+30)))

    tens = _hls(tens)
    cspace = tens.cspace 

    for roi in rectangular_roi_default:
        gravels = _generate_gravel_patch(roi)
        for gravel in gravels:
            x = gravel[0]
            y = gravel[1]
            r = random.randint(1, 4)
            r1 = random.randint(0, 255)
            tens[max(y-r,0):min(y+r,y), max(x-r,0):min(x+r,x), 1] = r1

    return to_tensor(tens, cspace=cspace)


def flare_source(tens, point, radius, src_color) -> Tensor:
    r"""
        Add a source of light (flare) on an specific region of an image.

    Args:
        tens (Tensor) : Any regular BGR/RGB image.
        point (int): Starting point of the flare.
        radius (int): Intended radius (in pixels) of the flare.
        src_color (tuple): Color of the flare. Must be in the format ``(R,G,B)``
        rectangular_roi (tuple): Rectanglar co-ordinates of the intended region of interest. Default: (-1,-1,-1,-1).
        num_patches (int): Number of patches to operate on.
        rgb (bool): Operate on RGB images. Default: True.
    
    Returns:
        Tensor of shape ``(height, width, channels)``.

    Examples::

        >> tens = caer.data.sunris)
        >> filtered = caer.filters.add_gravel(tens)
        >> filtered
        (427, 640, 3)

    """
    tens = to_tensor(tens, override_checks=True)

    if not isinstance(src_color, tuple):
        raise ValueError('`src_color` needs to be a tuple in the format `(R,G,B)`')
    
    # By default, assume a RGB-format
    # We reverse the tuple as OpenCV expects a BGR layout
    src_color = src_color[::-1]

    overlay = tens.clone()
    output = tens.clone()

    num_times = radius // 10
    alpha = np.linspace(0.0, 1, num=num_times)
    rad = np.linspace(1, radius, num=num_times)

    for i in range(num_times):
        cv.circle(overlay, point, int(rad[i]),  color=src_color, thickness=-1)
        alp = alpha[num_times-i-1]*alpha[num_times-i-1]*alpha[num_times-i-1]
        cv.addWeighted(overlay, alp, output, 1 - alp , 0, output)

    return to_tensor(tens, override_checks=True)


def _add_sun_flare_line(flare_center, angle, imshape):
    x = []
    y = []
    i = 0

    for rand_x in range(0, imshape[1], 10):
        rand_y = math.tan(angle)*(rand_x-flare_center[0])+flare_center[1]
        x.append(rand_x)
        y.append(2*flare_center[1]-rand_y)

    return x, y


def _add_sun_process(tens, num_flare_circles, flare_center, src_radius, x, y, src_color) -> Tensor:
    tens = to_tensor(tens, override_checks=True)
    overlay = tens.clone()
    output = tens.clone()
    imshape = tens.shape

    for i in range(num_flare_circles):
        alpha = random.uniform(0.05,0.2)
        r = random.randint(0, len(x)-1)
        rad = random.randint(1, imshape[0]//100-2)
        cv.circle(overlay, 
                 (int(x[r]),int(y[r])), 
                 rad*rad*rad, 
                 (random.randint(max(src_color[0]-50,0), src_color[0]),
                 random.randint(max(src_color[1]-50,0), src_color[1]), 
                 random.randint(max(src_color[2]-50,0), src_color[2])), 
                 -1
        )
        cv.addWeighted(overlay, alpha, output, 1 - alpha,0, output)            

    return flare_source(output, (int(flare_center[0]),int(flare_center[1])), src_radius, src_color)


def _apply_motion_blur(tens, count):
    tens = to_tensor(tens, override_checks=True)

    tens_t = tens.clone()
    imshape = tens_t.shape
    size = 15
    kernel_motion_blur = np.zeros((size, size))
    kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
    kernel_motion_blur = kernel_motion_blur / size

    i = imshape[1]*3//4 - 10 * count

    while i <= imshape[1]:
        tens_t[:,i:,:] = cv.filter2D(tens_t[:,i:,:], -1, kernel_motion_blur)
        tens_t[:,:imshape[1]-i,:] = cv.filter2D(tens_t[:,:imshape[1]-i,:], -1, kernel_motion_blur)
        i += imshape[1]//25 - count
        count += 1

    return to_tensor(tens_t, override_checks=True)


def _autumn_process(tens) -> Tensor:
    tens = to_tensor(tens, enforce_tensor=True)
    imshape = tens.shape
    tens = _hls(tens)
    cspace = tens.cspace

    step = 8
    aut_colors=[1, 5, 9, 11]
    col= aut_colors[random.randint(0, 3)]

    for i in range(0, imshape[1], step):
        for j in range(0, imshape[0], step):
            avg = np.average(tens[j:j+step,i:i+step,0])

            if avg > 20 and avg < 100 and np.average(tens[j:j+step,i:i+step,1]) < 100:
                tens[j:j+step,i:i+step,0] = col
                tens[j:j+step,i:i+step,2] =255

    return to_tensor(tens, cspace=cspace)


def _exposure_process(tens) -> Tensor:
    tens = to_tensor(tens, enforce_tensor=True)
    cspace = tens.cspace 

    tens = to_bgr(tens)
    tens_yuv = cv.cvtColor(tens, cv.COLOR_BGR2YUV)

    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
    ones = np.ones(tens_yuv[:,:,0].shape)
    ones[tens_yuv[:,:,0]>150] = 0.85
    tens_yuv[:,:,0] = tens_yuv[:,:,0]*ones

    tens_yuv[:,:,0] = clahe.apply(tens_yuv[:,:,0])
    tens_yuv[:,:,0] = cv.equalizeHist(tens_yuv[:,:,0])
    tens_yuv[:,:,0] = clahe.apply(tens_yuv[:,:,0])

    tens_res = cv.cvtColor(tens_yuv, cv.COLOR_YUV2BGR)
    tens = to_tensor(tens_res, cspace='bgr')
    tens = _cvt_color(tens, cspace)

    return cv.fastNlMeansDenoisingColored(tens, None, 3, 3, 7, 21)


def _cvt_color(tens, cspace):
    if cspace == 'rgb':
        return to_rgb(tens)
    if cspace == 'bgr':
        return to_bgr(tens)
    if cspace == 'gray':
        return to_gray(tens)
    if cspace == 'hsv':
        return to_hsv(tens)
    if cspace == 'hls':
        return to_hls(tens)
    if cspace == 'lab':
        return to_lab(tens)


def _generate_shadow_coordinates(num_shadows, rectangular_roi, shadow_dimension):
    vertices_list=[]
    x1 = rectangular_roi[0]
    y1 = rectangular_roi[1]
    x2 = rectangular_roi[2]
    y2 = rectangular_roi[3]

    for index in range(num_shadows):
        vertex = []

        for dimensions in range(shadow_dimension): ## Dimensionality of the shadow polygon
            vertex.append((random.randint(x1, x2), random.randint(y1, y2)))

        vertices = np.array([vertex], dtype=np.int32) ## single shadow vertices 
        vertices_list.append(vertices)
    return vertices_list ## List of shadow vertices


def _shadow_process(tens, num_shadows, x1, y1, x2, y2, shadow_dimension) -> Tensor:
    tens = to_tensor(tens, enforce_tensor=True)
    tens = _hls(tens) ## Conversion to hls
    cspace = tens.cspace 

    mask = np.zeros_like(tens) 
    imshape = tens.shape

    # Get the list of shadow vertices
    vertices_list = _generate_shadow_coordinates(num_shadows, (x1,y1,x2,y2), shadow_dimension) 

    for vertices in vertices_list: 
        cv.fillPoly(mask, vertices, 255) ## adding all shadow polygons on empty mask, single 255 denotes only red channel

    # If red channel is hot, the Tensor's "Lightness" channel's brightness value is lowered 
    tens[:,:,1][mask[:,:,0]==255] = tens[:,:,1][mask[:,:,0]==255]*0.5   

    return to_tensor(tens, cspace=cspace)
