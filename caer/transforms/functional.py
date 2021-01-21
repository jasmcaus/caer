#    _____           ______  _____ 
#  / ____/    /\    |  ____ |  __ \
# | |        /  \   | |__   | |__) | Caer - Modern Computer Vision
# | |       / /\ \  |  __|  |  _  /  Languages: Python, C, C++
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

from ..adorad import Tensor 

from ..color import (
    to_hls, 
    to_rgb,
    to_bgr
)

def _hls(img, rgb=True) -> Tensor:
    if rgb:
        return to_hls(img)
    else:
        return to_hls(img)


def _bgr(img, rgb=True) -> Tensor:
    if rgb:
        return to_bgr(img)
    else:
        return img


def _rgb(img, rgb=True) -> Tensor:
    if rgb:
        return img
    else:
        return to_rgb(img)


def _hue(img, rgb=True) -> Tensor:
    return _hls(img, rgb=rgb)[:,:,0]


def _get_img_size(img) -> Tensor:
    r"""
        Returns image size as (width, height)
    """
    h, w = img.shape[:2]

    return (w, h)


def _get_num_channels(img) -> int:
    r"""
        We assume only images of 1 and 3 channels
    """
    if len(img.shape) == 3 and img.shape[2] == 3:
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


def _snow_process(img, snow_coeff, rgb=True) -> Tensor:
    img = _hls(img, rgb=rgb)

    img = np.array(img, dtype=np.float64) 

    brightness_coefficient = 2.5 

    img[:,:,1][img[:,:,1]<snow_coeff] = img[:,:,1][img[:,:,1]<snow_coeff]*brightness_coefficient ## scale pixel values up for channel 1 (lightness)
    img[:,:,1][img[:,:,1]>255]  = 255 ##Sets all values above 255 to 255

    img = np.array(img, dtype=np.uint8)

    if rgb:
        return to_rgb(img)
    else:
        return to_bgr(img)


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


def _rain_process(img, slant, drop_length, drop_color, drop_width, rain_drops, rgb=True) -> Tensor:
    img_t = img.copy()

    for rain_drop in rain_drops:
        cv.line(img_t, (rain_drop[0],rain_drop[1]), (rain_drop[0]+slant,rain_drop[1]+drop_length), drop_color, drop_width)

    img = cv.blur(img_t,(7,7)) ## Rainy views are blurred
    brightness_coefficient = 0.7 ## Rainy days are usually shady 

    img = _hls(img) ## Conversion to hls
    img[:,:,1] = img[:,:,1]*brightness_coefficient ## scale pixel values down for channel 1(Lightness)

    if rgb:
        return to_rgb(img)
    else:
        return to_bgr(img)


def _add_blur(img, x, y, hw, fog_coeff) -> Tensor:
    overlay= img.copy()
    output= img.copy()
    alpha= 0.08*fog_coeff
    rad= hw//2
    point=(x+hw//2, y+hw//2)
    cv.circle(overlay,point, int(rad), (255,255,255), -1)
    cv.addWeighted(overlay, alpha, output, 1 -alpha ,0, output)

    return output


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


def _gravel_process(img, x1, x2, y1, y2, num_patches, rgb=True):
    x = img.shape[1]
    y = img.shape[0]
    rectangular_roi_default = []

    for i in range(num_patches):
        xx1 = random.randint(x1, x2)
        xx2 = random.randint(x1, xx1)
        yy1 = random.randint(y1, y2)
        yy2 = random.randint(y1, yy1)
        rectangular_roi_default.append((xx2, yy2, min(xx1, xx2+200), min(yy1, yy2+30)))

    img = _hls(img, rgb=rgb)

    for roi in rectangular_roi_default:
        gravels = _generate_gravel_patch(roi)
        for gravel in gravels:
            x = gravel[0]
            y = gravel[1]
            r = random.randint(1, 4)
            r1 = random.randint(0, 255)
            img[max(y-r,0):min(y+r,y), max(x-r,0):min(x+r,x), 1] = r1

    if rgb:
        return to_rgb(img)
    else:
        return to_bgr(img)


def flare_source(img, point, radius, src_color) -> Tensor:
    r"""
        Add a source of light (flare) on an specific region of an image.

    Args:
        img (Tensor) : Any regular BGR/RGB image.
        point (int): Starting point of the flare.
        radius (int): Intended radius (in pixels) of the flare.
        src_color (tuple): Color of the flare. Must be in the format ``(R,G,B)``
        rectangular_roi (tuple): Rectanglar co-ordinates of the intended region of interest. Default: (-1,-1,-1,-1).
        num_patches (int): Number of patches to operate on.
        rgb (bool): Operate on RGB images. Default: True.
    
    Returns:
        Tensor of shape ``(height, width, channels)``.

    Examples::

        >> img = caer.data.sunrise(rgb=True)
        >> filtered = caer.filters.add_gravel(img, rgb=True)
        >> filtered
        (427, 640, 3)

    """
    if not isinstance(src_color, tuple):
        raise ValueError('`src_color` needs to be a tuple in the format `(R,G,B)`')
    
    # By default, assume a RGB-format
    # We reverse the tuple as OpenCV expects a BGR layout
    src_color = src_color[::-1]

    overlay = img.copy()
    output = img.copy()
    num_times = radius // 10
    alpha = np.linspace(0.0, 1, num=num_times)
    rad = np.linspace(1, radius, num=num_times)

    for i in range(num_times):
        cv.circle(overlay, point, int(rad[i]),  color=src_color, thickness=-1)
        alp = alpha[num_times-i-1]*alpha[num_times-i-1]*alpha[num_times-i-1]
        cv.addWeighted(overlay, alp, output, 1 - alp , 0, output)

    return output


def _add_sun_flare_line(flare_center, angle, imshape):
    x = []
    y = []
    i = 0

    for rand_x in range(0, imshape[1], 10):
        rand_y = math.tan(angle)*(rand_x-flare_center[0])+flare_center[1]
        x.append(rand_x)
        y.append(2*flare_center[1]-rand_y)

    return x, y


def _add_sun_process(img, num_flare_circles, flare_center, src_radius, x, y, src_color) -> Tensor:
    overlay = img.copy()
    output = img.copy()
    imshape = img.shape

    for i in range(num_flare_circles):
        alpha = random.uniform(0.05,0.2)
        r = random.randint(0, len(x)-1)
        rad = random.randint(1, imshape[0]//100-2)
        cv.circle(overlay, (int(x[r]),int(y[r])), rad*rad*rad, (random.randint(max(src_color[0]-50,0), src_color[0]),random.randint(max(src_color[1]-50,0), src_color[1]), random.randint(max(src_color[2]-50,0), src_color[2])), -1)
        cv.addWeighted(overlay, alpha, output, 1 - alpha,0, output)            

    return flare_source(output, (int(flare_center[0]),int(flare_center[1])), src_radius, src_color)


def _apply_motion_blur(img, count):
    img_t = img.copy()
    imshape = img_t.shape
    size = 15
    kernel_motion_blur = np.zeros((size, size))
    kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
    kernel_motion_blur = kernel_motion_blur / size

    i= imshape[1]*3//4 - 10 * count

    while i <= imshape[1]:
        img_t[:,i:,:] = cv.filter2D(img_t[:,i:,:], -1, kernel_motion_blur)
        img_t[:,:imshape[1]-i,:] = cv.filter2D(img_t[:,:imshape[1]-i,:], -1, kernel_motion_blur)
        i += imshape[1]//25-count
        count+=1

    return img_t


def _autumn_process(img, rgb=True) -> Tensor:
    img_t = img.copy()
    imshape = img_t.shape
    img_t = _hls(img_t, rgb=rgb)
    step = 8
    aut_colors=[1, 5, 9, 11]
    col= aut_colors[random.randint(0, 3)]

    for i in range(0, imshape[1], step):
        for j in range(0, imshape[0], step):
            avg = np.average(img_t[j:j+step,i:i+step,0])

            if avg > 20 and avg < 100 and np.average(img[j:j+step,i:i+step,1]) < 100:
                img_t[j:j+step,i:i+step,0] = col
                img_t[j:j+step,i:i+step,2] =255

    if rgb:
        return to_rgb(img)
    else:
        return to_bgr(img)


def _exposure_process(img, rgb=True) -> Tensor:
    img = np.copy(img)
    img_yuv = cv.cvtColor(img, cv.COLOR_BGR2YUV)

    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
    ones = np.ones(img_yuv[:,:,0].shape)
    ones[img_yuv[:,:,0]>150] = 0.85
    img_yuv[:,:,0] = img_yuv[:,:,0]*ones

    img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
    img_yuv[:,:,0] = cv.equalizeHist(img_yuv[:,:,0])
    img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])

    if rgb:
        img_res = cv.cvtColor(img_yuv, cv.COLOR_YUV2RGB)
    else:
        img_res = cv.cvtColor(img_yuv, cv.COLOR_YUV2BGR)

    return cv.fastNlMeansDenoisingColored(img_res, None, 3, 3, 7, 21)


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


def _shadow_process(img, num_shadows, x1, y1, x2, y2, shadow_dimension, rgb=True) -> Tensor:
    img = _hls(img, rgb=rgb) ## Conversion to hls

    mask = np.zeros_like(img) 
    imshape = img.shape

    vertices_list= _generate_shadow_coordinates(num_shadows, (x1,y1,x2,y2), shadow_dimension) #3 getting list of shadow vertices

    for vertices in vertices_list: 
        cv.fillPoly(mask, vertices, 255) ## adding all shadow polygons on empty mask, single 255 denotes only red channel

    img[:,:,1][mask[:,:,0]==255] = img[:,:,1][mask[:,:,0]==255]*0.5   ## if red channel is hot, img's "Lightness" channel's brightness is lowered 

    if rgb:
        return to_rgb(img)
    else:
        return to_bgr(img)
