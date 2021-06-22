# Caer GUI
A simple Tkinter GUI app example, designed to showcase some of Caer's features for manipulating images.
This should only be used as a base to create a new GUI.

App Requirements:
* Python3
* Caer
* Matplotlib

Running it is as simple as 
```python
python caer_gui.py
```

Note: Tested as working in Windows 10 with python v3.6.8 and Kubuntu Linux with python v3.6.8. 

- You can select one of 14 built-in images to display (startup has `caer.data.island` selected as default)
- You can also select one of your own images (use the `Open File >>` option to browse locally or enter a URL to the image)
- Either of PNG / JPG / BMP / TIFF file types is available and was tested as working while some others might work as well
- Selecting any of the images, at any point in time, will always start with a fresh original image and reset controls
- The exception to the above is the 'Open File >>' option which will allow you to browse and select a different image
- 'Reload Image' button will reload the original version of currently selected image, including the image file loaded by the user

All function controls are set to manipulate the currently displayed image within the adjust_ghsps() function.

- Edges and Emboss effects are mutually exclusive (you can only have one applied at the time)
- Only one Histogram window can be displayed at any time
- The 'Rotation' button is currently set to keep on rotating the image with every tap (while preserving the whole image and only showing its rotated version)

# Caer GUI Video
A rather simple Tkinter GUI app example, designed to showcase some of Caer's features for playing videos.
This should only be used as a base to create a new GUI.

App Requirements:
* Python3
* Caer

Running it is as simple as 
```python
python caer_gui_video.py
```

Note: Tested as working in Windows 10 with python v3.6.8.

- You can select to display video from one of your cameras (0 is usually a default for laptop's built-in camera)
- You can also select to play one of your own video files (use the `Open File >>` option to browse locally or enter a URL to the file), either of AVI / MKV / MP4 / MPG / WMV file types is available and was tested as working while some others might work as well
- You can loop the playback of the video file
- You can take a screenshot of the current video frame (PNG file will be saved in the app's folder)
- There is no audio playback

# Useful Resources
Check the other GUI examples [here](https://github.com/GitHubDragonFly/CAER_Video_GUI).
