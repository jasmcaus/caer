# Simple tkinter GUI app designed to showcase some caer features
# Requirements: python3, caer, matplotlib

# Run it either via IDLE or from command prompt / terminal with one of these commands:
# - 'python caer_gui_test.py'
# - 'python -m caer_gui_test'
# - 'python3 caer_gui_test.py'
# - 'python3 -m caer_gui_test'

# Tested as working in Windows 10 with python v3.6.8 and Kubuntu Linux with python v3.6.8
# The "Original" button will display the original Sunrise image
# Replace the image with your own by following the instructions here: https://caer.readthedocs.io/en/latest/api/io.html
# All function controls are set to manipulate the currently displayed image

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from tkinter import *
import platform
import caer

# Standard 640x427 test image that ships out-of-the-box with caer
sunrise = caer.data.sunrise(rgb=True)

pythonVersion = platform.python_version()

def show_original_image():
    global currentImage
    global originalImgBtn
    global resizedImgBtn
    global flipHImgBtn
    global flipVImgBtn
    global flipHVImgBtn
    global rotateImgBtn

    if originalImgBtn['bg'] == 'lightgrey':
        originalImgBtn['bg'] = 'lightblue'

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

    currentImage = sunrise
    reset_ghs()

    image_show(currentImage)

def show_resized_image():
    global currentImage
    global originalImgBtn
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

                if originalImgBtn['bg'] == 'lightblue':
                    originalImgBtn['bg'] = 'lightgrey'
                elif flipHImgBtn['bg'] == 'lightblue':
                    flipHImgBtn['bg'] = 'lightgrey'
                elif flipVImgBtn['bg'] == 'lightblue':
                    flipVImgBtn['bg'] = 'lightgrey'
                elif flipHVImgBtn['bg'] == 'lightblue':
                    flipHVImgBtn['bg'] = 'lightgrey'
                else:
                    rotateImgBtn['bg'] = 'lightgrey'

            if not transformedImage is None:
                currentImage = transformedImage
                reset_ghs()

            # Resize the image without preserving aspect ratio
            currentImage = caer.resize(currentImage, target_size=(int(size[0]),int(size[1])), preserve_aspect_ratio=False)

            currentImage.cspace = 'rgb'

            if rotationApplied:
                show_rotated_image()
            else:
                image_show(currentImage)
        except Exception as e:
            print(str(e))

def show_h_flipped_image():
    global currentImage
    global originalImgBtn
    global resizedImgBtn
    global flipHImgBtn
    global flipVImgBtn
    global flipHVImgBtn
    global rotateImgBtn

    if flipHImgBtn['bg'] == 'lightgrey':
        flipHImgBtn['bg'] = 'lightblue'

        if originalImgBtn['bg'] == 'lightblue':
            originalImgBtn['bg'] = 'lightgrey'
        elif resizedImgBtn['bg'] == 'lightblue':
            resizedImgBtn['bg'] = 'lightgrey'
        elif flipVImgBtn['bg'] == 'lightblue':
            flipVImgBtn['bg'] = 'lightgrey'
        elif flipHVImgBtn['bg'] == 'lightblue':
            flipHVImgBtn['bg'] = 'lightgrey'
        else:
            rotateImgBtn['bg'] = 'lightgrey'

    if not transformedImage is None:
        currentImage = transformedImage
        reset_ghs()

    currentImage = caer.transforms.hflip(currentImage)

    currentImage.cspace = 'rgb'

    if rotationApplied:
        show_rotated_image()
    else:
        image_show(currentImage)

def show_v_flipped_image():
    global currentImage
    global originalImgBtn
    global resizedImgBtn
    global flipHImgBtn
    global flipVImgBtn
    global flipHVImgBtn
    global rotateImgBtn

    if flipVImgBtn['bg'] == 'lightgrey':
        flipVImgBtn['bg'] = 'lightblue'

        if originalImgBtn['bg'] == 'lightblue':
            originalImgBtn['bg'] = 'lightgrey'
        elif resizedImgBtn['bg'] == 'lightblue':
            resizedImgBtn['bg'] = 'lightgrey'
        elif flipHImgBtn['bg'] == 'lightblue':
            flipHImgBtn['bg'] = 'lightgrey'
        elif flipHVImgBtn['bg'] == 'lightblue':
            flipHVImgBtn['bg'] = 'lightgrey'
        else:
            rotateImgBtn['bg'] = 'lightgrey'

    if not transformedImage is None:
        currentImage = transformedImage
        reset_ghs()

    currentImage = caer.transforms.vflip(currentImage)

    currentImage.cspace = 'rgb'

    if rotationApplied:
        show_rotated_image()
    else:
        image_show(currentImage)

def show_hv_flipped_image():
    global currentImage
    global originalImgBtn
    global resizedImgBtn
    global flipHImgBtn
    global flipVImgBtn
    global flipHVImgBtn
    global rotateImgBtn

    if flipHVImgBtn['bg'] == 'lightgrey':
        flipHVImgBtn['bg'] = 'lightblue'

        if originalImgBtn['bg'] == 'lightblue':
            originalImgBtn['bg'] = 'lightgrey'
        elif resizedImgBtn['bg'] == 'lightblue':
            resizedImgBtn['bg'] = 'lightgrey'
        elif flipHImgBtn['bg'] == 'lightblue':
            flipHImgBtn['bg'] = 'lightgrey'
        elif flipVImgBtn['bg'] == 'lightblue':
            flipVImgBtn['bg'] = 'lightgrey'
        else:
            rotateImgBtn['bg'] = 'lightgrey'

    if not transformedImage is None:
        currentImage = transformedImage
        reset_ghs()

    currentImage = caer.transforms.hvflip(currentImage)

    currentImage.cspace = 'rgb'

    if rotationApplied:
        show_rotated_image()
    else:
        image_show(currentImage)

def show_rotated_image():
    global currentImage
    global rotationApplied
    global originalImgBtn
    global resizedImgBtn
    global flipHImgBtn
    global flipVImgBtn
    global flipHVImgBtn
    global rotateImgBtn

    angle = selectedAngle.get()

    if angle == '':
        angle = '0'
        rotationApplied = False
    elif angle == '0' or ((float(angle) > 0 or float(angle) < 0) and float(angle) % 360 == 0):
        rotationApplied = False
    else:
        rotationApplied = True

    anchor = None # Center point

    tempAnchorPoint = anchorSelection.get()

    if tempAnchorPoint == 'Top Left':
        anchor = (0, 0)
    elif tempAnchorPoint == 'Top Middle':
        anchor = ((currentImage.width() // 2), 0)
    elif tempAnchorPoint == 'Top Right':
        anchor = (currentImage.width(), 0)
    elif tempAnchorPoint == 'Middle Left':
        anchor = (0, (currentImage.height() // 2))
    elif tempAnchorPoint == 'Middle Right':
        anchor = (currentImage.width(), (currentImage.height() // 2))
    elif tempAnchorPoint == 'Bottom Left':
        anchor = (0, currentImage.height())
    elif tempAnchorPoint == 'Bottom Middle':
        anchor = ((currentImage.width() // 2), currentImage.height())
    elif tempAnchorPoint == 'Bottom Right':
        anchor = (currentImage.width(), currentImage.height())

    try:
        if rotateImgBtn['bg'] == 'lightgrey':
            rotateImgBtn['bg'] = 'lightblue'

            if originalImgBtn['bg'] == 'lightblue':
                originalImgBtn['bg'] = 'lightgrey'
            elif resizedImgBtn['bg'] == 'lightblue':
                resizedImgBtn['bg'] = 'lightgrey'
            elif flipHImgBtn['bg'] == 'lightblue':
                flipHImgBtn['bg'] = 'lightgrey'
            elif flipVImgBtn['bg'] == 'lightblue':
                flipVImgBtn['bg'] = 'lightgrey'
            else:
                flipHVImgBtn['bg'] = 'lightgrey'

        # preserve current image and only display its rotated version

        if not transformedImage is None:
            rot = caer.transforms.rotate(transformedImage, float(angle), rotPoint=anchor)
            if not rotationApplied:
                currentImage = transformedImage
                reset_ghs()
        else:
            rot = caer.transforms.rotate(currentImage, float(angle), rotPoint=anchor)

        rot.cspace = 'rgb'

        image_show(rot)
    except Exception as e:
        print(str(e))

def image_show(tens):
    global canvas
    global subplot

    subplot.clear()
    subplot.xaxis.set_ticks([]), subplot.yaxis.set_ticks([])  # Hides the graph ticks and x / y axis
    subplot.imshow(tens)
    canvas.draw()

def adjust_gamma_hue_saturation(*args):
    global transformedImage

    # apply all transformations to currently displayed image
    transformedImage = caer.transforms.adjust_hue(currentImage, hue_factor=hue.get())
    transformedImage = caer.transforms.adjust_saturation(transformedImage, saturation_factor=saturation.get())
    transformedImage = caer.transforms.adjust_gamma(transformedImage, gamma=imgGamma.get())

    if rotationApplied:
        show_rotated_image()
    else:
        image_show(transformedImage)

def reset_ghs():
    global transformedImage
    global imgGamma
    global hue
    global saturation

    transformedImage = None

    # reset gamma, hue and saturation sliders
    imgGamma.set(1.0)
    hue.set(0.0)
    saturation.set(1.0)

def main():
    global root
    global canvas
    global subplot
    global currentImage
    global transformedImage
    global originalImgBtn
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
    global imgGamma
    global hue
    global saturation

    root = Tk()
    root.config(background='white')
    root.title('CAER Sunrise GUI Test - Python v' + pythonVersion)
    root.geometry('1200x768')

    currentImage = None
    transformedImage = None
    rotationApplied = False

    # bind the 'q' keyboard key to quit
    root.bind('q', lambda event:root.destroy())

    # add a frame to hold all controls
    frame1 = Frame(root, background='black')
    frame1.pack(side='top', fill=X)

    # create all buttons and an entry box for the re-size dimensions
    originalImgBtn = Button(frame1, text='Original', width=6, bg='lightgrey', relief=RAISED, command=show_original_image)
    originalImgBtn.pack(side=LEFT, padx=2, pady=2)

    resizedImgBtn = Button(frame1, text='Resize', width=6, bg='lightgrey', relief=RAISED, command=show_resized_image)
    resizedImgBtn.pack(side=LEFT, padx=2, pady=2)

    selectedSize = StringVar()
    resizedImgSize = Entry(frame1, justify=CENTER, textvariable=selectedSize, font='Helvetica 10', width=9, bg='white', relief=RAISED)
    resizedImgSize.pack(side=LEFT, padx=2, pady=2)
    selectedSize.set('400x400')

    flipHImgBtn = Button(frame1, text='FlipH', width=6, bg='lightgrey', relief=RAISED, command=show_h_flipped_image)
    flipHImgBtn.pack(side=LEFT, padx=2, pady=2)

    flipVImgBtn = Button(frame1, text='FlipV', width=6, bg='lightgrey', relief=RAISED, command=show_v_flipped_image)
    flipVImgBtn.pack(side=LEFT, padx=2, pady=2)

    flipHVImgBtn = Button(frame1, text='FlipHV', width=6, bg='lightgrey', relief=RAISED, command=show_hv_flipped_image)
    flipHVImgBtn.pack(side=LEFT, padx=2, pady=2)

    rotateImgBtn = Button(frame1, text='Rotate', width=6, bg='lightgrey', relief=RAISED, command=show_rotated_image)
    rotateImgBtn.pack(side=LEFT, padx=2, pady=2)

    # create a label for the rotation angle
    lblAngle = Label(frame1, text='Angle', fg='yellow', bg='black', font='Helvetica 8')
    lblAngle.pack(side=LEFT, padx=2, pady=2)

    # create the rotation angle selection variable and an entry box
    selectedAngle = StringVar()
    rotationAngle = Entry(frame1, justify=CENTER, textvariable=selectedAngle, font='Helvetica 10', width=4, bg='white', relief=RAISED)
    rotationAngle.pack(side=LEFT, padx=2, pady=2)
    selectedAngle.set('90')

    # create a label for the rotation anchor
    lblAnchor = Label(frame1, text='Anchor', fg='yellow', bg='black', font='Helvetica 8')
    lblAnchor.pack(side=LEFT, padx=2, pady=2)

    # create the rotation anchor selection variable and choices
    anchorSelection = StringVar()
    anchorChoices = { 'BottomLeft', 'BottomMiddle', 'BottomRight', 'Center', 'MiddleLeft', 'MiddleRight', 'TopLeft', 'TopMiddle', 'TopRight'}
    anchorSelection.set('Center')

    # create the anchor selection popup menu
    popup_menu_anchor = OptionMenu(frame1, anchorSelection, *anchorChoices)
    popup_menu_anchor.pack(side=LEFT, padx=2)

    # create the image gamma slider control
    imgGamma = DoubleVar()
    sliderGamma = Scale(frame1, label='Gamma', variable=imgGamma, troughcolor='blue', from_=0.0, to=2.0, resolution=0.1, sliderlength=15, showvalue=False, orient=HORIZONTAL, command=adjust_gamma_hue_saturation)
    sliderGamma.pack(side=LEFT, padx=5, pady=2)
    imgGamma.set(1.0)

    # create the image hue slider control
    hue = DoubleVar()
    sliderHue = Scale(frame1, label='Hue', variable=hue, troughcolor='blue', from_=-0.5, to=0.5, resolution=0.05, sliderlength=15, showvalue=False, orient=HORIZONTAL, command=adjust_gamma_hue_saturation)
    sliderHue.pack(side=LEFT, padx=5, pady=2)
    hue.set(0.0)

    # create the image saturation slider control
    saturation = DoubleVar()
    sliderSaturation = Scale(frame1, label='Saturation', variable=saturation, troughcolor='blue', from_=0.0, to=2.0, resolution=0.1, sliderlength=15, showvalue=False, orient=HORIZONTAL, command=adjust_gamma_hue_saturation)
    sliderSaturation.pack(side=LEFT, padx=5, pady=2)
    saturation.set(1.0)

    # add exit button
    exitBtn = Button(frame1, text='Exit', width=6, bg='lightgrey', relief=RAISED, command=root.destroy)
    exitBtn.pack(side=RIGHT, padx=4, pady=2)

    # create matplotlib figure, subplot, canvas and toolbar
    fig = Figure(figsize=(5, 4), dpi=400)
    subplot = fig.add_subplot(111)
    subplot.xaxis.set_ticks([]), subplot.yaxis.set_ticks([])  # Hides the graph ticks and x / y axis

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()

    toolbar = NavigationToolbar2Tk(canvas, root)
    toolbar.update()

    canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)

    # set the minimum window size to the current size
    root.update()
    root.minsize(root.winfo_width(), root.winfo_height())

    show_original_image()

    root.mainloop()

if __name__=='__main__':
    main()
