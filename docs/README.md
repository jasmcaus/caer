# Caer Documentation
Caer is a set of utility functions designed to help speed up your Computer Vision workflow. Functions inside `caer` will help reduce the number of calculation calls your code makes, ultimately making it neat, concise and readable.

## Overview
- [Caer Version](#get-caer-version)
- [Get Caer Functions](#get-all-caer-functions)
- [Load Images](#load-images)
- [List Images in Directory](#list-image-files-from-directory)
- [List Videos in Directory](#list-video-files-from-directory)
- [List Media in Directory](#list-media-files-from-directory)
- [Translation](#translation)
- [Rotation](#rotation)
- [Resizing](#resizing)
- [Edge Cascades](#edge-cascades)
- [Switch between Colour Spaces](#switch-between-colour-spaces)
- [Image From URL](#image-from-url)
- [Save Lists to disk](#save-lists-to-disk)
- [Train & Validation Split](#train-and-validation-split)



## Get Caer Version
Get the current version number of your `caer` installation.

For versions 1.7.7 above, use either `caer.get_caer_version()` or `caer.get_version()`.
For versions below 1.7.7, use `caer.__version__`.

## Get all Caer functions
`caer.get_caer_methods()` or `caer.get_caer_functions()` will return a tuple of all the available functions in your current installation of `caer`. 

## Load Images
`caer.load_img` reads in an image from a specified filepath. 

**Arguments**
- `image_path`: Path to an image (filepath or URL)
- `target_size`: Final destination size of the image. Tuple of size 2 (width, height). Specify `None` to retain original image dimensions. 
- `channels`: 1 (convert to grayscale) or 3 (BGR/RGB). Default: 3
- `rgb`: Boolean to decide if keep RGB (True) or BGR (False) formatting. Default: True
- `resize_factor`: Resizes image using a specific integer/float. Default: None
- `keep_aspect_ratio`: Resizes an image to `target_size` keeping the aspect ratio. Some parts of the image may not be included (such as margins of the image). Default: False
```python
# BGR Image
>> image = caer.load_img(path, target_size=None, channels=3, rgb=False)

# RGB Image
>> image = caer.load_img(path, target_size=None, channels=3, rgb=True)
```

## List Image Files from Directory
`caer.path.list_images()` lists all image files in the immediate directory (if `include_subdirs = False`)  or all sub-directories, otherwise. 
```python
>> image_list = caer.list_images(DIR='Photos', include_subdirs=True, use_fullpath=False, show_size=False)
>> print(image_list)
```

## List Video Files from Directory
`caer.path.list_videos()` lists all image files in the immediate directory (if `include_subdirs = False`)  or all sub-directories, otherwise. 
```python
>> video_list = caer.list_videos(DIR='Videos', include_subdirs=True, use_fullpath=False, show_size=False)
>> print(video_list)
```

## List Media Files from Directory
`caer.path.list_media()` lists all media files (images and videos) in the immediate directory (if `include_subdirs = False`)  or all sub-directories, otherwise. 
```python
>> media_list = caer.list_media(DIR='Media', include_subdirs=True, use_fullpath=False, show_size=False)
>> print(media_list)
```

## Translation
Image translation can be performed by simply calling `caer.translate()` 
```python
# Shifts an image 50 pixels to the right and 100 pixels up
>> translated = caer.translate(image, 50, -100)
```

## Rotation
Image rotate can be performed by calling `caer.rotate()`. 
If rotation point `rotPoint` is not specified, the image will be rotated around the centre. 
```python
# Rotates an image around the centre counter-clockwise by 45 degrees
>> rotated = caer.rotate(image, 45, rotPoint=None)
```

## Resizing
`caer.resize()` resizes an image either by using a scale factor (keeps aspect ratio) or to a strict image size (original aspect ratio may not be kept)
```python
# Resizes the image to half its original dimensions
>> half_img = caer.resize(image, scale_factor=.5)
# Resizes the image to a fixed size of (500, 500)
>> img_500 = caer.resize(image, dimensions=(500,500))
```

## Edge Cascades
Available v1.7.7 onwards, `caer.edges()` computes the edges in an image either using 2 threshold values or the median of the image (if `use_median` = True). 

Note: Median is given priority if 2 threshold values are passed and `use_median` is True
```python
# Creating an edge cascade using the computed median 
>> median_edges = caer.edges(image, use_median=True, sigma=0.4)

# Creating an edge cascade using 2 threshold values
>> threshold_edges = caer.edges(image, 125, 180)
```

## Switch between Colour Spaces
Currently, `caer` supports converting an image from BGR to the RGB, Grayscale, HSV and LAB colour spaces. More colour spaces will be supported in future updates. 
```python
# BGR to RGB (Useful if you use Matplotlib to display images)
>> rgb = caer.to_rgb(image)
# BGR to Grayscale
>> gray = caer.to_gray(image)
# BGR to HSV
>> hsv = caer.to_hsv(image)
# BGR to LAB
>> lab = caer.to_lab(image)
```

## Image from URL
***Note: `caer.load_img()` can read in images from URLs as well. This is the recommended method.***

`caer.url_from_image()` reads in an image from a URL and returns it as an RGB image (if `rgb = True`) or BGR (if `rgb=False`)
```python
# Returns an RGB image
>> img_from_url_rgb = caer.url_from_image(url, rgb=True)
# Returns a BGR image
>> img_from_url_bgr = caer.url_from_image(url, rgb=False)
```

## Save lists to disk
`caer.saveNumpy()` saves Python lists or Numpy arrays as .npy or .npz files (extension inferred from the `base_name`)
```python
>> py_list = [1,2,3,4]
# Saving as a .npy file
>> caer.saveNumpy(base_name='py_list.npy', data=py_list)
# Saving as a .npz file
>> caer.saveNumpy(base_name='py_list_compressed.npz', data=py_list)
```

## Train and Validation Split
`caer.train_val_split()` splits the training set (features, labels) into actual training and validation sets
```python
>> X_train, y_train, X_val, y_val = caer.train_val_split(features, labels, val_ratio=.2)
```