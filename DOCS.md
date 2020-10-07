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

## Edge Cascades
`caer.edges` computes the edges in an image either using 2 threshold values or the median of the image (if `use_median` = True). Note: Median is given priority is 2 threshold values are passed and `use_median` is True
```python
# Creating an edge cascade using the computed median 
median_edges = caer.edges(image, use_median=True, sigma=0.4)

# Creating an edge cascade using 2 threshold values
threshold_edges = caer.edges(image, 125, 180)
```