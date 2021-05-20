# Caer GUI
A simple TkInter GUI app example, designed to showcase some of Caer's features
This should only be used as a base to create a new GUI.

Requirements: 
* Python
* Caer
* Matplotlib

Running it is as simple as 
```python
python caer_gui.py
```

Note: Tested as working in Windows 10 with python v3.6.8 and Kubuntu Linux with python v3.6.8. 
You can select one of 9 built-in images to display (startup has `caer.data.sland` selected as default)
Selecting any of the images, at any point in time, will always start with a fresh original image and reset 
controls.

Replace with or add your own image(s) by [following the instructions here](https://caer.readthedocs.io/en/latest/api/io.html).
This will require that you modify `main()` and `show_original_image()` functions.

All function controls are set to manipulate the currently displayed image.
Edges and Emboss effects are mutually exclusive (you can only have one applied at the time).
Gamma, Hue, Saturation, Sharpness, Posterize, Solarize, Edges and Emboss effects are currently somewhat unique 
and, when applied to the image, will follow the following rule:
  - Applying 'Resize', 'Rotate' and/or any of the 'Flip' functions to transformed image will preserve that image 
    and have all those effects reset

The above mentioned could possibly be corrected by converting all those buttons to checkboxes and applying all the effects within a single function (just use the current adjust_ghsps() function)
The 'Rotation' button is currently set to keep on rotating the image with every tap