# Caer GUI
A simple TkInter GUI app example, designed to showcase some of Caer's features.
This should only be used as a base to create a new GUI.

Requirements: 
* Python3
* Caer
* Matplotlib

Running it is as simple as 
```python
python caer_gui.py
```

Note: Tested as working in Windows 10 with python v3.6.8 and Kubuntu Linux with python v3.6.8. 
- You can select one of 14 built-in images to display (startup has `caer.data.island` selected as default)
- You can also select one of your own images (use `Open File >>` option to browse locally or enter a URL to the image), either of PNG / JPG / BMP file types is available and was tested as working
- Selecting any of the images, at any point in time, will always start with a fresh original image and reset 
controls (with the exception of 'Open File >>' which will allow you to browse and select a different image)
- 'Reload Image' button will reload the original version of currently selected image, including the image file loaded by the user

All function controls are set to manipulate the currently displayed image within the adjust_ghsps() function.

- Edges and Emboss effects are mutually exclusive (you can only have one applied at the time)
- Histogram will not be available when Edges are enabled
- Only one Histogram window can be displayed at any time
- The 'Rotation' button is currently set to keep on rotating the image with every tap (while preserving the whole image and only showing its rotated version)
