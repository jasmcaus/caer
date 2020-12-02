import caer 
import os 
import cv2 as cv 
import numpy as np 

here = os.path.dirname(__file__)


def test_imread():
    test_img = os.path.join(here, 'tests', 'data', 'beverages.jpg')

    img = caer.imread(test_img)
    test_against = cv.imread(test_img) 

    assert np.all(img == test_against)


def test_gray():
    test_img = os.path.join(here, 'tests', 'data', 'green_fish.jpg')

    img = caer.imread(test_img, channels=1)

    assert len(img.shape) == 2