import cv2 as cv 
from PIL import Image as image
import numpy as np 

def cv_to_pill(cv_img):
    cv_img = cv.cvtColor(cv_img, cv.COLOR_BGR2RGB)
    return image.fromarray(cv_img)


def pill_to_cv(pill_img):
    pill_img = np.array(pill_img)
    return cv.cvtColor(pill_img, cv.COLOR_RGB2GRAY)