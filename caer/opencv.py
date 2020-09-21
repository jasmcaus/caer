# Author: Jason Dsouza
# Github: http://www.github.com/jasmcaus

# Importing the necessary packages
import cv2 as cv
import numpy as np
from urllib.request import urlopen


def get_opencv_version():
    return cv.__version__[0]

def translate(image, x, y):
    # Defines the translation matrix and performs the translation
    matrix = np.float32([[1, 0, x], [0, 1, y]])
    shifted = cv.warpAffine(image, matrix, (image.shape[1], image.shape[0]))

    return shifted

def rotate(image, angle, centre=None, scale=1.0):
    # Grabs the dimensions of the image
    (height, width) = image.shape[:2]

    # If no centre is specified, we grab the centre coordinates of the image
    if centre is None:
        centre = (width // 2, height // 2)

    # Rotates the image
    M = cv.getRotationMatrix2D(centre, angle, scale)
    rotated = cv.warpAffine(image, M, (width, height))

    return rotated

def rotate_bound(image, angle):
    # Grabs the dimensions of the image and then determines the centre
    (h, w) = image.shape[:2]
    (cX, cY) = (w / 2, h / 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # Adjusts the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # Performs the actual rotation and returns the image
    return cv.warpAffine(image, M, (nW, nH))

def resize(image, width=None, height=None, interpolation=cv.INTER_AREA):
    """
    Resizes the image while maintaing the aspect ratio of the original image
    """
    # Grabs the image dimensions 
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, we return the original image
    if width is None and height is None:
        return image

    # If  width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # If height is None
    else:
        # Calculates the ratio of the width and constructs the dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # Resizes the image
    resized = cv.resize(image, dim, interpolation=interpolation)

    return resized

def toMatplotlib(image):
    """
    Converts BGR image ordering to RGB ordering
    """
    return cv.cvtColor(image, cv.COLOR_BGR2RGB)

def url_to_image(url):
    # Converts the image to a Numpy array and reads it in OpenCV
    response = urlopen(url)
    image = np.asarray(bytearray(response.read()), dtype="uint8")
    image = cv.imdecode(image, cv.IMREAD_COLOR)

    return image

def canny(image, sigma=0.33):
    # computes the median of the single channel pixel intensities
    med = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * med))
    upper = int(min(255, (1.0 + sigma) * med))
    edges = cv.Canny(image, lower, upper)

    return edges