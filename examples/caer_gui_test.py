# Simple tkinter GUI app example, designed to showcase some caer features
# Should only be used as a base to create a new GUI
# It can be re-designed, controls re-grouped and code improved

# Requirements: python3, caer, matplotlib

# Run it either via IDLE or from command prompt / terminal with one of these commands:
# - 'python caer_gui_test.py'
# - 'python -m caer_gui_test'
# - 'python3 caer_gui_test.py'
# - 'python3 -m caer_gui_test'

# Tested as working in Windows 10 with python v3.6.8 and Kubuntu Linux with python v3.6.8
# You can select one of 9 built-in images to display (startup has "Island" selected as default)
# Selecting any of the images, at any point in time, will always start with a fresh original image and reset controls.
# Replace with or add your own image(s) by following the instructions here: https://caer.readthedocs.io/en/latest/api/io.html
# The above will require that you modify main() and show_original_image() functions
# All function controls are set to manipulate the currently displayed image
# Gamma, Hue, Saturation, Posterize and Solarize effects are currently somewhat unique and, when applied to the image, will follow the following rule:
# - Applying 'Resize', 'Rotate' and/or any of the 'Flip' functions to transformed image will preserve that image and have all the sliders reset
# The above mentioned could be corrected by changing all those buttons to checkboxes and applying all the effects within a single function (similar to the current adjust_ghsps() function)
# The 'Rotation' button is currently set to keep on rotating the image with every tap

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from tkinter import *
import platform
import math
import caer

pythonVersion = platform.python_version()

def show_original_image(*args):
    global currentImage
    global rotationApplied
    global lblCurrentAngle
    global currentAngle
    global resizedImgBtn
    global flipHImgBtn
    global flipVImgBtn
    global flipHVImgBtn
    global rotateImgBtn

    if resizedImgBtn['bg'] == 'lightblue':
        resizedImgBtn['bg'] = 'lightgrey'
    elif flipHImgBtn['bg'] == 'lightblue':
        flipHImgBtn['bg'] = 'lightgrey'
    elif flipVImgBtn['bg'] == 'lightblue':
        flipVImgBtn['bg'] = 'lightgrey'
    elif flipHVImgBtn['bg'] == 'lightblue':
        flipHVImgBtn['bg'] = 'lightgrey'
    else:
        rotateImgBtn['bg'] = 'lightgrey'

    selectedImage = imageSelection.get()

    if selectedImage == 'Mountain':
        currentImage = caer.data.mountain(rgb=True)
    elif selectedImage == 'Sunrise':
        currentImage = caer.data.sunrise(rgb=True)
    elif selectedImage == 'Island':
        currentImage = caer.data.island(rgb=True)
    elif selectedImage == 'Puppies':
        currentImage = caer.data.puppies(rgb=True)
    elif selectedImage == 'Black Cat':
        currentImage = caer.data.black_cat(rgb=True)
    elif selectedImage == 'Gold Fish':
        currentImage = caer.data.gold_fish(rgb=True)
    elif selectedImage == 'Bear':
        currentImage = caer.data.bear(rgb=True)
    elif selectedImage == 'Camera':
        currentImage = caer.data.camera(rgb=True)
    else:
        currentImage = caer.data.guitar(rgb=True)

    rotationApplied = False
    currentAngle = 0.0
    lblCurrentAngle['text'] = str(currentAngle)
    selectedAngle.set('0.0')
    reset_ghsps()

    image_show(currentImage)

def show_resized_image():
    global currentImage
    global resizedImgBtn
    global flipHImgBtn
    global flipVImgBtn
    global flipHVImgBtn
    global rotateImgBtn

    tempSize = selectedSize.get()

    if 'x' in tempSize:
        size = tempSize.replace(' ', '').split('x')

        try:
            if resizedImgBtn['bg'] == 'lightgrey':
                resizedImgBtn['bg'] = 'lightblue'

                if flipHImgBtn['bg'] == 'lightblue':
                    flipHImgBtn['bg'] = 'lightgrey'
                elif flipVImgBtn['bg'] == 'lightblue':
                    flipVImgBtn['bg'] = 'lightgrey'
                elif flipHVImgBtn['bg'] == 'lightblue':
                    flipHVImgBtn['bg'] = 'lightgrey'
                else:
                    rotateImgBtn['bg'] = 'lightgrey'

            if not transformedImage is None:
                currentImage = transformedImage
                reset_ghsps()

            # Resize the image without preserving aspect ratio
            currentImage = caer.resize(currentImage, target_size=(int(size[0]),int(size[1])), preserve_aspect_ratio=False)
            currentImage.cspace = 'rgb'

            if rotationApplied:
                show_rotated_image(True)
            else:
                image_show(currentImage)
        except Exception as e:
            print(str(e))

def show_h_flipped_image():
    global currentImage
    global resizedImgBtn
    global flipHImgBtn
    global flipVImgBtn
    global flipHVImgBtn
    global rotateImgBtn

    if flipHImgBtn['bg'] == 'lightgrey':
        flipHImgBtn['bg'] = 'lightblue'

        if resizedImgBtn['bg'] == 'lightblue':
            resizedImgBtn['bg'] = 'lightgrey'
        elif flipVImgBtn['bg'] == 'lightblue':
            flipVImgBtn['bg'] = 'lightgrey'
        elif flipHVImgBtn['bg'] == 'lightblue':
            flipHVImgBtn['bg'] = 'lightgrey'
        else:
            rotateImgBtn['bg'] = 'lightgrey'

    if not transformedImage is None:
        currentImage = transformedImage
        reset_ghsps()

    currentImage = caer.transforms.hflip(currentImage)
    currentImage.cspace = 'rgb'

    if rotationApplied:
        show_rotated_image(True)
    else:
        image_show(currentImage)

def show_v_flipped_image():
    global currentImage
    global resizedImgBtn
    global flipHImgBtn
    global flipVImgBtn
    global flipHVImgBtn
    global rotateImgBtn

    if flipVImgBtn['bg'] == 'lightgrey':
        flipVImgBtn['bg'] = 'lightblue'

        if resizedImgBtn['bg'] == 'lightblue':
            resizedImgBtn['bg'] = 'lightgrey'
        elif flipHImgBtn['bg'] == 'lightblue':
            flipHImgBtn['bg'] = 'lightgrey'
        elif flipHVImgBtn['bg'] == 'lightblue':
            flipHVImgBtn['bg'] = 'lightgrey'
        else:
            rotateImgBtn['bg'] = 'lightgrey'

    if not transformedImage is None:
        currentImage = transformedImage
        reset_ghsps()

    currentImage = caer.transforms.vflip(currentImage)
    currentImage.cspace = 'rgb'

    if rotationApplied:
        show_rotated_image(True)
    else:
        image_show(currentImage)

def show_hv_flipped_image():
    global currentImage
    global resizedImgBtn
    global flipHImgBtn
    global flipVImgBtn
    global flipHVImgBtn
    global rotateImgBtn

    if flipHVImgBtn['bg'] == 'lightgrey':
        flipHVImgBtn['bg'] = 'lightblue'

        if resizedImgBtn['bg'] == 'lightblue':
            resizedImgBtn['bg'] = 'lightgrey'
        elif flipHImgBtn['bg'] == 'lightblue':
            flipHImgBtn['bg'] = 'lightgrey'
        elif flipVImgBtn['bg'] == 'lightblue':
            flipVImgBtn['bg'] = 'lightgrey'
        else:
            rotateImgBtn['bg'] = 'lightgrey'

    if not transformedImage is None:
        currentImage = transformedImage
        reset_ghsps()

    currentImage = caer.transforms.hvflip(currentImage)
    currentImage.cspace = 'rgb'

    if rotationApplied:
        show_rotated_image(True)
    else:
        image_show(currentImage)

def show_rotated_image(external = False):
    global currentImage
    global rotationApplied
    global lblCurrentAngle
    global currentAngle
    global resizedImgBtn
    global flipHImgBtn
    global flipVImgBtn
    global flipHVImgBtn
    global rotateImgBtn

    angle = selectedAngle.get()

    if angle == '':
        angle = '0.0'
        currentAngle = 0.0
        rotationApplied = False
    elif angle == '0.0' or ((float(angle) > 0 or float(angle) < 0) and math.fmod(float(angle), 360) == 0):
        currentAngle = 0.0
        rotationApplied = False
    else:
        if not external:
            currentAngle += float(angle)

        mod = math.fmod(currentAngle, 360)

        if currentAngle > 360 or currentAngle < -360:
            currentAngle = mod
            rotationApplied = True
        elif mod == 0:
            currentAngle = 0.0
            rotationApplied = False
        else:
            rotationApplied = True

    lblCurrentAngle['text'] = str(currentAngle)

    tempAnchorPoint = anchorSelection.get()

    if tempAnchorPoint == 'Center':
        anchor = None
    elif tempAnchorPoint == 'TopLeft':
        anchor = (0, 0)
    elif tempAnchorPoint == 'TopMiddle':
        anchor = ((currentImage.width() // 2), 0)
    elif tempAnchorPoint == 'TopRight':
        anchor = (currentImage.width(), 0)
    elif tempAnchorPoint == 'MiddleLeft':
        anchor = (0, (currentImage.height() // 2))
    elif tempAnchorPoint == 'MiddleRight':
        anchor = (currentImage.width(), (currentImage.height() // 2))
    elif tempAnchorPoint == 'BottomLeft':
        anchor = (0, currentImage.height())
    elif tempAnchorPoint == 'BottomMiddle':
        anchor = ((currentImage.width() // 2), currentImage.height())
    elif tempAnchorPoint == 'BottomRight':
        anchor = (currentImage.width(), currentImage.height())

    try:
        if rotateImgBtn['bg'] == 'lightgrey':
            rotateImgBtn['bg'] = 'lightblue'

            if resizedImgBtn['bg'] == 'lightblue':
                resizedImgBtn['bg'] = 'lightgrey'
            elif flipHImgBtn['bg'] == 'lightblue':
                flipHImgBtn['bg'] = 'lightgrey'
            elif flipVImgBtn['bg'] == 'lightblue':
                flipVImgBtn['bg'] = 'lightgrey'
            else:
                flipHVImgBtn['bg'] = 'lightgrey'

        # preserve current image and only display its rotated version

        if not transformedImage is None:
            rot = caer.transforms.rotate(transformedImage, float(currentAngle), rotPoint=anchor)
            if not rotationApplied:
                currentImage = transformedImage
                reset_ghsps()
        else:
            rot = caer.transforms.rotate(currentImage, float(currentAngle), rotPoint=anchor)

        rot.cspace = 'rgb'

        image_show(rot)
    except Exception as e:
        print(str(e))

def image_show(tens):
    subplot.clear()
    subplot.imshow(tens)
    canvas.draw()

def refresh_axis():
    global showAxis

    # Hide / Show the graph x / y axis
    if not showAxis:
        subplot.xaxis.set_visible(True), subplot.yaxis.set_visible(True)
        showAxis = True
    else:
        subplot.xaxis.set_visible(False), subplot.yaxis.set_visible(False)
        showAxis = False

    fig.canvas.draw()

def adjust_ghsps(*args):
    global transformedImage

    # apply all transformations to currently displayed image
    transformedImage = caer.transforms.adjust_hue(currentImage, hue.get())
    transformedImage = caer.transforms.adjust_saturation(transformedImage, saturation.get())
    transformedImage = caer.transforms.adjust_gamma(transformedImage, imgGamma.get())

    if posterize.get() < 6:
        transformedImage = caer.transforms.posterize(transformedImage, posterize.get())

    if solarize.get() < 255:
        transformedImage = caer.transforms.solarize(transformedImage, solarize.get())

    transformedImage.cspace = 'rgb'
    
    if rotationApplied:
        show_rotated_image(True)
    else:
        image_show(transformedImage)

def reset_ghsps():
    global transformedImage
    global imgGamma
    global hue
    global saturation
    global posterize
    global solarize

    transformedImage = None

    # reset all sliders
    imgGamma.set(1.0)
    hue.set(0.0)
    saturation.set(1.0)
    posterize.set(6)
    solarize.set(255)

def main():
    global root
    global canvas
    global fig
    global subplot
    global currentImage
    global transformedImage
    global imageSelection
    global showAxis
    global sliderSolarize
    global resizedImgBtn
    global flipHImgBtn
    global flipVImgBtn
    global flipHVImgBtn
    global rotateImgBtn
    global selectedSize
    global selectedAngle
    global resizedImgSize
    global rotationAngle
    global anchorSelection
    global rotationApplied
    global lblCurrentAngle
    global currentAngle
    global imgGamma
    global hue
    global saturation
    global posterize
    global solarize

    root = Tk()
    root.config(background='white')
    root.title('CAER GUI Test - Python v' + pythonVersion)
    root.geometry('1024x768')

    currentImage = None
    transformedImage = None
    rotationApplied = False
    showAxis = False
    currentAngle = 0.0

    # bind the 'q' keyboard key to quit
    root.bind('q', lambda event:root.destroy())

    # add a frame to hold top controls
    frame1 = Frame(root, background='black')
    frame1.pack(side=TOP, fill=X)

    # create the built-in image selection variable and choices
    imageSelection = StringVar()
    imageChoices = { 'Mountain', 'Sunrise', 'Island', 'Puppies', 'Black Cat', 'Gold Fish', 'Bear', 'Camera', 'Guitar'}
    imageSelection.set('Island')
    imageSelection.trace('w', show_original_image)

    # create the built-in image selection popup menu
    popup_menu_image = OptionMenu(frame1, imageSelection, *imageChoices)
    popup_menu_image['width'] = 10
    popup_menu_image['bg'] = 'lightgreen'
    popup_menu_image.pack(side=LEFT, padx=2)

    # create a button to re-size the image
    resizedImgBtn = Button(frame1, text='Resize', width=6, bg='lightgrey', relief=RAISED, command=show_resized_image)
    resizedImgBtn.pack(side=LEFT, padx=2, pady=2)

    # create an entry box for re-size dimensions
    selectedSize = StringVar()
    resizedImgSize = Entry(frame1, justify=CENTER, textvariable=selectedSize, font='Helvetica 10', width=10, bg='white', relief=RAISED)
    resizedImgSize.pack(side=LEFT, padx=2, pady=2)
    selectedSize.set('400x400')

    # create a button to flip the image horizontally
    flipHImgBtn = Button(frame1, text='FlipH', width=6, bg='lightgrey', relief=RAISED, command=show_h_flipped_image)
    flipHImgBtn.pack(side=LEFT, padx=2, pady=2)

    # create a button to flip the image vertically
    flipVImgBtn = Button(frame1, text='FlipV', width=6, bg='lightgrey', relief=RAISED, command=show_v_flipped_image)
    flipVImgBtn.pack(side=LEFT, padx=2, pady=2)

    # create a button to flip the image horizontally and vertically
    flipHVImgBtn = Button(frame1, text='FlipHV', width=6, bg='lightgrey', relief=RAISED, command=show_hv_flipped_image)
    flipHVImgBtn.pack(side=LEFT, padx=2, pady=2)

    # create a button to rotate the image
    rotateImgBtn = Button(frame1, text='Rotate', width=6, bg='lightgrey', relief=RAISED, command=show_rotated_image)
    rotateImgBtn.pack(side=LEFT, padx=2, pady=2)

    # create a label for the rotation angle
    lblAngle = Label(frame1, text='Angle', fg='yellow', bg='black', font='Helvetica 8')
    lblAngle.pack(side=LEFT, padx=2, pady=2)

    # create the rotation angle selection variable and an entry box
    selectedAngle = StringVar()
    rotationAngle = Entry(frame1, justify=CENTER, textvariable=selectedAngle, font='Helvetica 10', width=5, bg='white', relief=RAISED)
    rotationAngle.pack(side=LEFT, padx=2, pady=2)
    selectedAngle.set('0.0')

    # create a read-only label for the current angle
    lblCurrentAngle = Label(frame1, text='0.0', state='disabled', fg='lightgrey', bg='white', font='Helvetica 8', width=5)
    lblCurrentAngle.pack(side=LEFT, padx=2, pady=2)

    # create a label for the rotation anchor
    lblAnchor = Label(frame1, text='Anchor', fg='yellow', bg='black', font='Helvetica 8')
    lblAnchor.pack(side=LEFT, padx=2, pady=2)

    # create the rotation anchor selection variable and choices
    anchorSelection = StringVar()
    anchorChoices = { 'BottomLeft', 'BottomMiddle', 'BottomRight', 'Center', 'MiddleLeft', 'MiddleRight', 'TopLeft', 'TopMiddle', 'TopRight'}
    anchorSelection.set('Center')

    # create the anchor selection popup menu
    popup_menu_anchor = OptionMenu(frame1, anchorSelection, *anchorChoices)
    popup_menu_anchor['width'] = 12
    popup_menu_anchor.pack(side=LEFT, padx=2)

    #-----------------------------------------------------------------------

    # add a frame to hold side controls
    frame2 = Frame(root, background='black')
    frame2.pack(side=RIGHT, fill=Y)

    # create the image gamma slider control
    imgGamma = DoubleVar()
    sliderGamma = Scale(frame2, label='Gamma', variable=imgGamma, troughcolor='blue', from_=0.0, to=2.0, resolution=0.1, sliderlength=15, showvalue=False, orient=HORIZONTAL, command=adjust_ghsps)
    sliderGamma.pack(side=TOP, anchor=E, padx=2, pady=2)
    imgGamma.set(1.0)

    # create the image hue slider control
    hue = DoubleVar()
    sliderHue = Scale(frame2, label='Hue', variable=hue, troughcolor='blue', from_=-0.5, to=0.5, resolution=0.05, sliderlength=15, showvalue=False, orient=HORIZONTAL, command=adjust_ghsps)
    sliderHue.pack(side=TOP, anchor=E, padx=2, pady=2)
    hue.set(0.0)

    # create the image saturation slider control
    saturation = DoubleVar()
    sliderSaturation = Scale(frame2, label='Saturation', variable=saturation, troughcolor='blue', from_=0.0, to=2.0, resolution=0.1, sliderlength=15, showvalue=False, orient=HORIZONTAL, command=adjust_ghsps)
    sliderSaturation.pack(side=TOP, anchor=E, padx=2, pady=2)
    saturation.set(1.0)

    # create the image posterize slider control
    posterize = IntVar()
    sliderPosterize = Scale(frame2, label='Posterize', variable=posterize, troughcolor='blue', from_=6, to=1, resolution=1, sliderlength=15, showvalue=False, orient=HORIZONTAL, command=adjust_ghsps)
    sliderPosterize.pack(side=TOP, padx=2, pady=10)
    posterize.set(6)

    # create the image solarize slider control
    solarize = IntVar()
    sliderSolarize = Scale(frame2, label='Solarize', variable=solarize, troughcolor='blue', from_=255, to=0, resolution=1, sliderlength=15, showvalue=False, orient=HORIZONTAL, command=adjust_ghsps)
    sliderSolarize.pack(side=TOP, padx=2, pady=5)
    solarize.set(255)

    # add exit button
    exitBtn = Button(frame2, text='Exit', width=6, bg='lightgrey', relief=RAISED, command=root.destroy)
    exitBtn.pack(side=BOTTOM, anchor=CENTER, pady=4)

    # create matplotlib figure, subplot, canvas and toolbar
    fig = Figure(figsize=(6.4, 4.3), dpi=100)
    subplot = fig.add_subplot(111)
    subplot.xaxis.set_visible(False), subplot.yaxis.set_visible(False)

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()

    toolbar = NavigationToolbar2Tk(canvas, root)
    toolbar._Spacer()
    toolbar._Button('Reload Image', None, toggle=False, command=show_original_image)
    toolbar._Spacer()
    toolbar._Button('Show / Hide Axis', None, toggle=True, command=refresh_axis)
    toolbar.update()

    canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)

    # set the minimum window size to the current size
    root.update()
    root.minsize(root.winfo_width(), root.winfo_height())

    show_original_image()

    root.mainloop()

if __name__=='__main__':
    main()
