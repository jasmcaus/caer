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

from .functional import is_list, is_numeric, is_numeric_list_or_tuple, is_tuple, _is_numpy_array
from ..color import (
    rgb_to_hls, 
    bgr_to_hls, 
    bgr_to_rgb,
    rgb_to_bgr,
    hls_to_bgr,
    hls_to_rgb
)

BGR2HLS = 52
RGB2HLS = 53


def _hls(img, rgb=True):
    if rgb:
        return rgb_to_hls(img)
    else:
        return bgr_to_hls(img)


def _bgr(img, rgb=True):
    if rgb:
        return rgb_to_bgr(img)
    else:
        return img


def _rgb(img, rgb=True):
    if rgb:
        return img
    else:
        return bgr_to_rgb(img)


def _hue(img, rgb=True):
    return _hls(img, rgb=rgb)[:,:,0]


def lightness(img, rgb=True):
    return _hls(img, rgb=rgb)[:,:,1]


def saturation(img, rgb=True):
    return _hls(img, rgb=rgb)[:,:,2]


def change_light(img, coeff, rgb=True):
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
    if coeff !=-1:
        if coeff < 0.0 or coeff > 1.0:
            raise ValueError('Brightness coefficient can only be between 0.0 and 1.0')

    if coeff == -1:
        coeff_t = 1 + random.uniform(0, 1) # coeff between 1.0 and 1.5
    else:
        coeff_t = 1 + coeff  # coeff between 1.0 and 2.0

    return change_light(img, coeff_t, rgb=rgb)


def darken(img, darkness_coeff = -1, rgb=True):
    if darkness_coeff != -1:
        if darkness_coeff < 0.0 or darkness_coeff > 1.0:
            raise ValueError('Darkness coeff must only be between 0.0 and 1.0') 

    if darkness_coeff == -1:
        darkness_coeff_t = 1 - random.uniform(0, 1)
    else:
        darkness_coeff_t = 1 - darkness_coeff  

    return change_light(img, darkness_coeff_t, rgb=True)


def random_brightness(img, rgb=True):
    rand_br_coeff = 2 * np.random.uniform(0, 1) # Generates a value between 0.0 and 2.0
    return change_light(img, rand_br_coeff, rgb=rgb)


err_snow_coeff="Snow coeff can only be between 0 and 1"

def snow_process(img, snow_coeff, rgb=True):
    img = _hls(img, rgb=rgb)

    img = np.array(img, dtype=np.float64) 

    brightness_coefficient = 2.5 

    img[:,:,1][img[:,:,1]<snow_coeff] = img[:,:,1][img[:,:,1]<snow_coeff]*brightness_coefficient ## scale pixel values up for channel 1 (lightness)
    img[:,:,1][img[:,:,1]>255]  = 255 ##Sets all values above 255 to 255

    img = np.array(img, dtype=np.uint8)

    if rgb:
        return hls_to_rgb(img)
    else:
        return hls_to_bgr(img)


def add_snow(img, snow_coeff=-1, rgb=True):
    if snow_coeff != -1:
        if snow_coeff < 0.0 or snow_coeff > 1.0:
            raise ValueError('Snow coeff must only be between 0 and 1')
    else:
        snow_coeff=random.uniform(0, 1)

    snow_coeff*=255/2
    snow_coeff+=255/3

    return snow_process(img, snow_coeff, rgb=rgb)


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


def rain_process(img, slant, drop_length, drop_color, drop_width, rain_drops, rgb=True):
    img_t= img.copy()

    for rain_drop in rain_drops:
        cv.line(img_t, (rain_drop[0],rain_drop[1]), (rain_drop[0]+slant,rain_drop[1]+drop_length), drop_color, drop_width)

    img= cv.blur(img_t,(7,7)) ## Rainy views are blurred
    brightness_coefficient = 0.7 ## Rainy days are usually shady 

    img = _hls(img) ## Conversion to hls
    img[:,:,1] = img[:,:,1]*brightness_coefficient ## scale pixel values down for channel 1(Lightness)

    if rgb:
        return hls_to_rgb(img)
    else:
        return hls_to_bgr(img)


# Rain_type = 'drizzle', 'heavy', 'torrential'
def add_rain(img, slant=-1, drop_length=20, drop_width=1, drop_color=(200,200,200), rain_type='None', rgb=True): ## (200,200,200) is a shade of gray
    slant_extreme = slant
    if not(is_numeric(slant_extreme) and (slant_extreme >=-20 and slant_extreme <= 20) or slant_extreme==-1):
        raise ValueError('Numeric value must be between -20 and 20')

    if not(is_numeric(drop_width) and drop_width>=1 and drop_width<=5):
        raise ValueError('Width must be between 1 and 5')

    if not(is_numeric(drop_length) and drop_length>=0 and drop_length<=100):
        raise ValueError('Length must be between 0 and 100')

    imshape = img.shape
    if slant_extreme == -1:
        slant= np.random.randint(-10,10) ##generate random slant if no slant value is given

    rain_drops, drop_length= _generate_random_lines(imshape, slant, drop_length, rain_type)
    return rain_process(img, slant_extreme, drop_length, drop_color, drop_width, rain_drops)


def add_blur(img, x, y, hw, fog_coeff):
    overlay= img.copy()
    output= img.copy()
    alpha= 0.08*fog_coeff
    rad= hw//2
    point=(x+hw//2, y+hw//2)
    cv.circle(overlay,point, int(rad), (255,255,255), -1)
    cv.addWeighted(overlay, alpha, output, 1 -alpha ,0, output)

    return output


def generate_random_blur_coordinates(imshape, hw):
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


def add_fog(img, fog_coeff=-1, rgb=True):
    if fog_coeff != -1:
        if fog_coeff < 0.0 or fog_coeff > 1.0:
            raise ValueError('Fog coefficient must be between 0 and 1')

    imshape = img.shape

    if fog_coeff == -1:
        fog_coeff_t = random.uniform(0.3,1)
    else:
        fog_coeff_t = fog_coeff

    hw = int(imshape[1]//3*fog_coeff_t)
    haze_list = generate_random_blur_coordinates(imshape,hw)
    for haze_points in haze_list: 
        img = add_blur(img, haze_points[0], haze_points[1], hw, fog_coeff_t) 

    img = cv.blur(img, (hw//10, hw//10))
    
    return _rgb(img, rgb=rgb)


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


def gravel_process(img, x1, x2, y1, y2, num_patches, rgb=True):
    x=img.shape[1]
    y=img.shape[0]
    rectangular_roi_default=[]

    for i in range(num_patches):
        xx1=random.randint(x1, x2)
        xx2=random.randint(x1, xx1)
        yy1=random.randint(y1, y2)
        yy2=random.randint(y1, yy1)
        rectangular_roi_default.append((xx2, yy2, min(xx1,xx2+200), min(yy1,yy2+30)))

    img = _hls(img, rgb=rgb)

    for roi in rectangular_roi_default:
        gravels = _generate_gravel_patch(roi)
        for gravel in gravels:
            x = gravel[0]
            y = gravel[1]
            r = random.randint(1, 4)
            r1 = random.randint(0, 255)
            img[max(y-r,0):min(y+r,y),max(x-r,0):min(x+r,x),1] = r1

    if rgb:
        return hls_to_rgb(img)
    else:
        return hls_to_bgr(img)


def add_gravel(img, rectangular_roi=(-1,-1,-1,-1), num_patches=8, rgb=True):
    if is_tuple(rectangular_roi) and is_numeric_list_or_tuple(rectangular_roi) and len(rectangular_roi)==4:
        x1 = rectangular_roi[0]
        y1 = rectangular_roi[1]
        x2 = rectangular_roi[2]
        y2 = rectangular_roi[3]
    else:
        raise ValueError('Rectangular ROI dimensions are invalid.')

    if rectangular_roi==(-1,-1,-1,-1):
        if(_is_numpy_array(img)):
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

    return gravel_process(img, x1, x2, y1, y2, num_patches, rgb=rgb)


def flare_source(img, point, radius, src_color):
    overlay = img.copy()
    output = img.copy()
    num_times = radius//10
    alpha = np.linspace(0.0, 1, num=num_times)
    rad = np.linspace(1, radius, num=num_times)

    for i in range(num_times):
        cv.circle(overlay, point, int(rad[i]),  color=src_color, thickness=-1)
        alp = alpha[num_times-i-1]*alpha[num_times-i-1]*alpha[num_times-i-1]
        cv.addWeighted(overlay, alp, output, 1 -alp , 0, output)

    return output


def add_sun_flare_line(flare_center, angle, imshape):
    x=[]
    y=[]
    i=0

    for rand_x in range(0, imshape[1], 10):
        rand_y = math.tan(angle)*(rand_x-flare_center[0])+flare_center[1]
        x.append(rand_x)
        y.append(2*flare_center[1]-rand_y)

    return x, y


def add_sun_process(img, num_flare_circles, flare_center, src_radius, x, y, src_color):
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


def add_sun_flare(img, flare_center=-1, angle=-1, num_flare_circles=8, src_radius=400, src_color=(255,255,255)):
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

    x, y = add_sun_flare_line(flare_center_t, angle_t, imshape)

    return add_sun_process(img, num_flare_circles, flare_center_t, src_radius, x, y, src_color)


def apply_motion_blur(img, count):
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


def add_speed(img, speed_coeff=-1):
    if speed_coeff != -1:
        if speed_coeff < 0.0 or speed_coeff > 1.0:
            raise ValueError('Speed coefficient must be between 0 and 1')

    if speed_coeff == -1:
        count_t = int(15 * random.uniform(0, 1))
    else:
        count_t = int(15 * speed_coeff)

    return apply_motion_blur(img, count_t)


def autumn_process(img, rgb=True):
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
        return hls_to_rgb(img)
    else:
        return hls_to_bgr(img)


def add_autumn(img, rgb=True):
    return autumn_process(img, rgb=rgb)


def exposure_process(img, rgb=True):
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


def correct_exposure(img, rgb=True):
    return exposure_process(img, rgb=rgb)


def augment_random(img, aug_types='', volume='expand' ):
    aug_types_all = ['random_brightness','add_shadow','add_snow','add_rain','add_fog','add_gravel','add_sun_flare','add_speed','add_autumn','random_flip']

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


def _generate_shadow_coordinates(imshape, num_shadows, rectangular_roi, shadow_dimension):
    vertices_list=[]
    x1=rectangular_roi[0]
    y1=rectangular_roi[1]
    x2=rectangular_roi[2]
    y2=rectangular_roi[3]

    for index in range(num_shadows):
        vertex=[]

        for dimensions in range(shadow_dimension): ## Dimensionality of the shadow polygon
            vertex.append((random.randint(x1, x2), random.randint(y1, y2)))

        vertices = np.array([vertex], dtype=np.int32) ## single shadow vertices 
        vertices_list.append(vertices)
    return vertices_list ## List of shadow vertices


def shadow_process(img, num_shadows, x1, y1, x2, y2, shadow_dimension, rgb=True):
    img = _hls(img, rgb=rgb) ## Conversion to hls

    mask = np.zeros_like(img) 
    imshape = img.shape

    vertices_list= _generate_shadow_coordinates(imshape, num_shadows,(x1,y1,x2,y2), shadow_dimension) #3 getting list of shadow vertices

    for vertices in vertices_list: 
        cv.fillPoly(mask, vertices, 255) ## adding all shadow polygons on empty mask, single 255 denotes only red channel

    img[:,:,1][mask[:,:,0]==255] = img[:,:,1][mask[:,:,0]==255]*0.5   ## if red channel is hot, img's "Lightness" channel's brightness is lowered 

    if rgb:
        return hls_to_rgb(img)
    else:
        return hls_to_bgr(img)


## ROI:(top-left x1,y1, bottom-right x2,y2), shadow_dimension=no. of sides of polygon generated
def add_shadow(img, num_shadows=1, rectangular_roi=(-1,-1,-1,-1), shadow_dimension=5): 
    if not(is_numeric(num_shadows) and num_shadows>=1 and num_shadows<=10):
        raise ValueError('Only 1-10 shadows can be introduced in an image')

    if not(is_numeric(shadow_dimension) and shadow_dimension>=3 and shadow_dimension<=10):
        raise ValueError('Polygons with dimensions < 3 don\'t exist and take time to plot')

    if is_tuple(rectangular_roi) and is_numeric_list_or_tuple(rectangular_roi) and len(rectangular_roi)==4:
        x1=rectangular_roi[0]
        y1=rectangular_roi[1]
        x2=rectangular_roi[2]
        y2=rectangular_roi[3]
    else:
        raise ValueError('Rectangular ROI dimensions are not valid')

    if rectangular_roi==(-1,-1,-1,-1):
        x1=0
        
        if(_is_numpy_array(img)):
            y1=img.shape[0]//2
            x2=img.shape[1]
            y2=img.shape[0]
        else:
            y1=img[0].shape[0]//2
            x2=img[0].shape[1]
            y2=img[0].shape[0]

    elif x1 == -1 or y1 == -1 or x2 == -1 or y2 == -1 or x2 <= x1 or y2 <= y1:
        raise ValueError('Rectangular ROI dimensions are not valid')

    return shadow_process(img,num_shadows, x1, y1, x2, y2, shadow_dimension)