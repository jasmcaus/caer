# Caer GUI
A simple TkInter GUI app example, designed to showcase some of Caer's features
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
You can select one of 9 built-in images to display (startup has `caer.data.island` selected as default)
Selecting any of the images, at any point in time, will always start with a fresh original image and reset 
controls ('Reload Image' button will do the same).

Replace with or add your own image(s) by [following the instructions here](https://caer.readthedocs.io/en/latest/api/io.html).
This will require that you modify `main()` and `show_original_image()` functions.

All function controls are set to manipulate the currently displayed image:
- Edges and Emboss effects are mutually exclusive (you can only have one applied at the time)
- Gamma, Hue, Saturation, Sharpness, Posterize, Solarize, Edges and Emboss effects are all applied at the same time within the adjust_ghsps() function
- Applying 'Resize', 'Rotate' and/or any of the 'Flip' functions to transformed image will preserve that image and have all the above effects sliders reset

The last point above could possibly be corrected by doing the following:
- Remove FlipHV button and its function all together (since FlipH and FlipV provide the same functionality when used together)
- Convert the rest of the buttons to checkboxes and conditionally apply their effects within the current adjust_ghsps() function (probably after all other transformations are applied).

The 'Rotation' button is currently set to keep on rotating the image with every tap (while preserving the whole image and only showing its rotated version).
