# Author: Jason Dsouza
# Github: http://www.github.com/jasmcaus

# Importing the necessary packages
import cv2 as cv
import numpy as np
from urllib.request import urlopen


def get_opencv_version():
    return cv.__version__[0]


def translate(image, x, y):
    """
        Translates a given image across the x-axis and the y-axis
        :param x: shifts the image right (positive) or left (negative)
        :param y: shifts the image down (positive) or up (negative)
    """
    transMat = np.float32([[1, 0, x], [0, 1, y]])
    return cv.warpAffine(image, transMat, (image.shape[1], image.shape[0]))


# def rotate(image, angle, rotPoint=None):
#     """
#         Rotates an given image by an angle around a particular rotation point (if provided) or centre otherwise.
#     """
#     height, width = image.shape[:2]

#     # If no rotPoint is specified, we assume the rotation point to be around the centre
#     if rotPoint is None:
#         centre = (width//2, height//2)

#     rotMat = cv.getRotationMatrix2D(centre, angle, scale=1.0)
#     return cv.warpAffine(image, rotMat, (width, height))


def _cv2_resize(image, width, height, interpolation=None):
    """
    ONLY TO BE USED INTERNALLY. NOT AVAILABLE FOR EXTERNAL USAGE. 
    Resizes the image ignoring the aspect ratio of the original image
    """
    if interpolation is None:
        interpolation = cv.INTER_AREA
    dimensions = (width,height)
    return cv.resize(image, dimensions, interpolation=interpolation)


def rotate(img, angle):
    h, w = img.shape[:2]
    (cX, cY) = (w/2, h/2)

    # Computing the sine and cosine (rotation components of the matrix)
    transMat = cv.getRotationMatrix2D((cX, cY), angle, scale=1.0)
    cos = np.abs(transMat[0, 0])
    sin = np.abs(transMat[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h*sin) + (w*cos))
    nH = int((h*cos) + (w*sin))

    # Adjusts the rotation matrix to take into account translation
    transMat[0, 2] += (nW/2) - cX
    transMat[1, 2] += (nH/2) - cY

    # Performs the actual rotation and returns the image
    return cv.warpAffine(img, transMat, (nW, nH))


def canny(img, threshold1, threshold2, use_median=True, sigma=None):
    if use_median and sigma is None:
        sigma = .3
    
    if use_median:
        # computes the median of the single channel pixel intensities
        med = np.median(img)

        # Canny edge detection using the computed mean
        low = int(max(0, (1.0-sigma) * med))
        up = int(min(255, (1.0+sigma) * med))
        edges = cv.Canny(img, low, up)
    
    else:
        edges = cv.Canny(img, threshold1, threshold2)

    return edges


def to_rgb(img):
    """
        Converts an image from the BGR image format to its RGB version
    """
    if img.shape != 3:
        raise ValueError(f'[ERROR] Image of shape 3 expected. Found shape {img.shape}')

    return cv.cvtColor(img, cv.COLOR_BGR2RGB)


def to_gray(img):
    """
        Converts an image from the BGR image format to its Grayscale version
    """
    if img.shape != 3:
        raise ValueError(f'[ERROR] Image of shape 3 expected. Found shape {img.shape}')

    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)


def to_hsv(img):
    """
        Converts an image from the BGR image format to its HSV version
    """
    if img.shape != 3:
        raise ValueError(f'[ERROR] Image of shape 3 expected. Found shape {img.shape}')

    return cv.cvtColor(img, cv.COLOR_BGR2HSV)


def to_lab(img):
    """
        Converts an image from the BGR image format to its HSV version
    """
    if img.shape != 3:
        raise ValueError(f'[ERROR] Image of shape 3 expected. Found shape {img.shape}')

    return cv.cvtColor(img, cv.COLOR_BGR2LAB)


def url_to_image(url):
    # Converts the image to a Numpy array and reads it in OpenCV
    response = urlopen(url)
    image = np.asarray(bytearray(response.read()), dtype='uint8')
    image = cv.imdecode(image, cv.IMREAD_COLOR)

    return image