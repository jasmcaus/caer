# Caer Documentation
Caer is a set of utility functions based off OpenCV, designed to help speed up your Computer Vision workflow. Functions inside `caer` will help reduce the number of lines of code you add, while still maintaining the overall purpose. 

## Translation
Image translation can be performed by simply calling `caer.translate` 
```python
# Shifts an image 50 pixels to the right and 100 pixels up
translated = caer.translate(image, 50, -100)
```

## Rotation
Image rotate can be performed by calling `caer.rotate`. 
If rotation point `rotPoint` is not specified, the image will be rotated around the centre. 
```python
# Rotates an image around the centre counter-clockwise by 45 degrees
rotated = caer.rotate(image, 45, rotPoint=None)
```
