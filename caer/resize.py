# Author: Jason Dsouza
# Github: http://www.github.com/jasmcaus

#pylint: disable=bare-except

# Importing the necessary packages
import cv2 as cv
from .opencv import _cv2_resize

def resize(img, IMG_SIZE, keep_aspect_ratio=False):
    """
        Resizes an image using advanced algorithms
        :param IMG_SIZE: Tuple of size 2 in the format (width,height)
        :param keep_aspect_ratio: Boolean to keep/ignore aspect ratio when resizing
    """
    if type(IMG_SIZE) is not tuple or (type(IMG_SIZE) is tuple and len(IMG_SIZE) != 2):
        raise ValueError("[ERROR] IMG_SIZE needs to be a tuple of size 2")
    
    if type(keep_aspect_ratio) is not bool:
        raise ValueError("[ERROR] keep_aspect_ratio must be a boolean")

    new_w, new_h = IMG_SIZE

    if not keep_aspect_ratio: # If False, use OpenCV's resize method
        return _cv2_resize(img, width=new_w, height=new_h)
    else:
        org_height, org_width = img.shape[:2]
        # Computing minimal resize
        min_height = _compute_minimal_resize(org_height, new_h)
        min_width = _compute_minimal_resize(org_width, new_w)

        # Resizing minimally
        img = cv.resize(img, dsize=(min_width,min_height))

        # Resizing to new dimensions
        img = cv.resize(img, dsize=(new_w,new_h))
        return img

def _compute_minimal_resize(org_dim,dim):
    for i in range(10):
        i += 1
        d = dim*i
        if org_dim >= d and org_dim < dim*(i+1):
            return d