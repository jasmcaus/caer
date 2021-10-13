# Simple tkinter GUI app example, designed to showcase some caer features for manipulating images
# Should only be used as a base to create a new GUI
# It can be re-designed, controls re-grouped and code improved

# Requirements: python3, caer, matplotlib

# Run it either via IDLE or from command prompt / terminal with one of these commands:
# - 'python caer_gui.py'
# - 'python -m caer_gui'
# - 'python3 caer_gui.py'
# - 'python3 -m caer_gui'

# Tested as working in Windows 10 with python v3.6.8 and Kubuntu Linux with python v3.6.8
# You can select one of 14 built-in images to display (startup has "Island" selected as default)
# You can also browse and select one of your images (use the 'Open File >>' option and either browse locally or enter a URL)
# Either of PNG / JPG / BMP / TIFF file types is available and was tested as working
# Selecting any of the images, at any point in time, will always start with a fresh original image and reset controls
# The exception to the above is 'Open File >>' which will allow you to select a different image
# The 'Reload Image' button will reload the original version of currently selected image, including the user opened file

# All function controls are set to manipulate the currently displayed image
# Edges and Emboss effects are mutually exclusive (you can only have one applied at the time)
# Only one Histogram window can be displayed at any time
# The 'Rotation' button is currently set to keep on rotating the image with every tap

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from tkinter import *
from tkinter import filedialog as fd

import platform
import math
import caer

pythonVersion = platform.python_version()

def reload_image():
    global reload_local_file

    if imageSelection.get() == 'Open File >>':
        reload_local_file = True
    
    show_original_image()

def show_original_image(*args):
    global currentImage
    global previous_image
    global reload_local_file
    global popup_menu_image
    global image_size
    global resizedImgBtn
    global rotateImgBtn

    user_cancelled_or_error = False

    if resizedImgBtn['bg'] == 'lightblue':
        resizedImgBtn['bg'] = 'lightgrey'
    else:
        rotateImgBtn['bg'] = 'lightgrey'

    selectedImage = imageSelection.get()

    previous = lblFileName['text']
    lblFileName['text'] = ''
    lblError['text'] = ''

    if selectedImage == 'Open File >>':
        if not reload_local_file:
            all_ext = ('All files', '*.*')
            png_ext = ('PNG files', '.png')
            jpg_ext = ('JPG files', '.jpg .jpeg')
            bmp_ext = ('BMP files', '.bmp')
            tiff_ext = ('TIFF files', '.tif .tiff')

            try:
                img_filename = fd.askopenfilename(filetypes=(all_ext, png_ext, jpg_ext, bmp_ext, tiff_ext))

                if img_filename != '':
                    lblFileName['text'] = img_filename
                    currentImage = caer.imread(img_filename)
                else:
                    # user clicked 'Cancel' button
                    user_cancelled_or_error = True
            except Exception as e:
                lblError['text'] = 'Error'
                user_cancelled_or_error = True
                print(str(e))
        else:
            reload_local_file = False
            lblFileName['text'] = previous
    elif selectedImage == 'Mountain':
        currentImage = caer.data.mountain(rgb=True)
    elif selectedImage == 'Snow':
        currentImage = caer.data.snow(rgb=True)
    elif selectedImage == 'Sunrise':
        currentImage = caer.data.sunrise(rgb=True)
    elif selectedImage == 'Night':
        currentImage = caer.data.night(rgb=True)
    elif selectedImage == 'Island':
        currentImage = caer.data.island(rgb=True)
    elif selectedImage == 'Puppies':
        currentImage = caer.data.puppies(rgb=True)
    elif selectedImage == 'Black Cat':
        currentImage = caer.data.black_cat(rgb=True)
    elif selectedImage == 'Sea Turtle':
        currentImage = caer.data.sea_turtle(rgb=True)
    elif selectedImage == 'Gold Fish':
        currentImage = caer.data.gold_fish(rgb=True)
    elif selectedImage == 'Bear':
        currentImage = caer.data.bear(rgb=True)
    elif selectedImage == 'Beverages':
        currentImage = caer.data.beverages(rgb=True)
    elif selectedImage == 'Tent':
        currentImage = caer.data.tent(rgb=True)
    elif selectedImage == 'Camera':
        currentImage = caer.data.camera(rgb=True)
    else:
        currentImage = caer.data.guitar(rgb=True)

    if not user_cancelled_or_error:
        image_size = [str(int(currentImage.width())), str(int(currentImage.height()))]
        selectedSize.set(image_size[0] + 'x' + image_size[1])

        reset_ghsps()

        image_show(currentImage)
    else:
        lblFileName['text'] = previous
        imageSelection.set(previous_image)
        popup_menu_image['bg'] = 'green'
        popup_menu_image['bg'] = 'lightgreen'
        selectedImage = previous_image

    previous_image = selectedImage

def resize_image():
    global resizedImgBtn
    global rotateImgBtn
    global image_resized
    global image_size

    # reset the error label's text
    if lblError['text'] == 'Error':
        lblError['text'] = ''

    tempSize = selectedSize.get()

    if 'x' in tempSize:
        try:
            findex = tempSize.index('x')
            lindex = tempSize.rindex('x')

            if findex == lindex:
                image_size = tempSize.replace(' ', '').split('x')

                # test if all values can be represented with an integer
                int(image_size[0])
                int(image_size[1])

                if resizedImgBtn['bg'] == 'lightgrey':
                    resizedImgBtn['bg'] = 'lightblue'
                    rotateImgBtn['bg'] = 'lightgrey'

                image_resized = True

                adjust_ghsps()
            else:
                lblError['text'] = 'Error'
                print("Invalid size specified!")
        except Exception as e:
            lblError['text'] = 'Error'
            print(str(e))
    else:
        lblError['text'] = 'Error'
        print("Invalid size specified!")

def show_rotated_image(external = False):
    global rotationApplied
    global currentAngle
    global anchor
    global resizedImgBtn
    global rotateImgBtn

    # reset the error label's text
    if lblError['text'] == 'Error':
        lblError['text'] = ''

    angle = selectedAngle.get()

    try:
        # test if the angle value can be represented with a float
        float(angle)

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

                if rotateImgBtn['bg'] == 'lightgrey':
                    rotateImgBtn['bg'] = 'lightblue'
                    resizedImgBtn['bg'] = 'lightgrey'

        # only display the rotated version of the image
        if not transformedImage is None:
            rot = caer.transforms.rotate(transformedImage, float(currentAngle), rotPoint=anchor)
        else:
            rot = caer.transforms.rotate(currentImage, float(currentAngle), rotPoint=anchor)

        image_show(rot)
    except Exception as e:
        lblError['text'] = 'Error'
        print(str(e))

def image_show(tens):
    subplot.clear()
    subplot.imshow(tens) # optionally add aspect='auto' to switch to automatic aspect mode
    canvas.draw()

def refresh_axis():
    global showAxis

    # reset the error label's text
    if lblError['text'] == 'Error':
        lblError['text'] = ''

    # Hide / Show the graph's x / y axis
    if not showAxis:
        subplot.xaxis.set_visible(True), subplot.yaxis.set_visible(True)
        showAxis = True
    else:
        subplot.xaxis.set_visible(False), subplot.yaxis.set_visible(False)
        showAxis = False

    fig.canvas.draw()

def flip_image_horizontally():
    global flip_H

    flip_H = not flip_H
    adjust_ghsps()

def flip_image_vertically():
    global flip_V

    flip_V = not flip_V
    adjust_ghsps()

def set_edges():
    global show_emboss

    if show_edges.get() == 1:
        show_emboss.set(0)
        sliderEmboss['state'] = 'disabled'
        sliderLowThreshold['state'] = 'normal'
    else:
        sliderLowThreshold['state'] = 'disabled'

    adjust_ghsps()

def set_emboss():
    global show_edges

    if show_emboss.get() == 1:
        show_edges.set(0)
        sliderLowThreshold['state'] = 'disabled'
        sliderEmboss['state'] = 'normal'
    else:
        sliderEmboss['state'] = 'disabled'
    
    adjust_ghsps()

def set_sharpen_kernel(*args):
    global sharpenKernel

    sharpenKernel = caer.data.np.array([[-1, -1, -1], [-1, sharpen.get(), -1], [-1, -1, -1]])

    adjust_ghsps()

def adjust_ghsps(*args):
    global transformedImage

    if not currentImage is None:
        # reset the error label's text
        if lblError['text'] == 'Error':
            lblError['text'] = ''

        transformedImage = caer.to_tensor(currentImage, cspace = "rgb")

        # apply all transformations to currently displayed image

        if image_resized:
            transformedImage = caer.resize(transformedImage, target_size=(int(image_size[0]),int(image_size[1])), preserve_aspect_ratio=False)
        
        if hue.get() != 0.0:
            transformedImage = caer.transforms.adjust_hue(transformedImage, hue.get())

        if saturation.get() != 1.0:
            transformedImage = caer.transforms.adjust_saturation(transformedImage, saturation.get())

        if imgGamma.get() != 1.05:
            transformedImage = caer.transforms.adjust_gamma(transformedImage, imgGamma.get())

        if sharpen.get() != 8.9:
            transformedImage = caer.core.cv.filter2D(transformedImage, -1, sharpenKernel)

        gb = gaussian_blur.get()

        if gb > 1:
            transformedImage = caer.core.cv.GaussianBlur(transformedImage, (gb + 1, gb + 1), caer.core.cv.BORDER_DEFAULT)

        if posterize.get() < 6:
            transformedImage = caer.transforms.posterize(transformedImage, posterize.get())

        if solarize.get() < 255:
            transformedImage = caer.transforms.solarize(transformedImage, solarize.get())

        if sobel_threshold.get() > 0:
            transformedImage = caer.core.cv.cvtColor(transformedImage, caer.core.cv.COLOR_RGB2GRAY)
            sobelKernel = sobel_threshold.get() if sobel_threshold.get() % 2 != 0 else sobel_threshold.get() + 1 # values 1, 3 and 5
            dx = dy = sobel_threshold.get() - 2 if sobel_threshold.get() > 2 else sobel_threshold.get()
            sobelx = caer.core.cv.Sobel(transformedImage, caer.core.cv.IMREAD_GRAYSCALE, dx, 0, ksize=sobelKernel)
            sobely = caer.core.cv.Sobel(transformedImage, caer.core.cv.IMREAD_GRAYSCALE, 0, dy, ksize=sobelKernel)
            transformedImage = caer.core.cv.bitwise_or(sobelx, sobely)
            transformedImage = caer.core.cv.cvtColor(transformedImage, caer.core.cv.COLOR_GRAY2RGB)

        if show_edges.get() == 1:
            transformedImage = caer.core.cv.cvtColor(transformedImage, caer.core.cv.COLOR_RGB2GRAY)
            transformedImage = caer.core.cv.Canny(transformedImage, low_threshold.get(), low_threshold.get() * 2)
            transformedImage = caer.core.cv.cvtColor(transformedImage, caer.core.cv.COLOR_GRAY2RGB)

        if show_emboss.get() == 1:
            transformedImage = caer.core.cv.filter2D(transformedImage, -1, embossKernel) + emboss.get()

        if flip_H:
            transformedImage = caer.transforms.hflip(transformedImage)

        if flip_V:
            transformedImage = caer.transforms.vflip(transformedImage)

        if rotationApplied:
            show_rotated_image(True)
        else:
            image_show(transformedImage)

def reset_ghsps():
    global rotationApplied
    global currentAngle
    global anchor
    global image_resized
    global transformedImage
    global imgGamma
    global hue
    global saturation
    global gaussian_blur
    global posterize
    global solarize
    global show_edges
    global low_threshold
    global sobel_threshold
    global sharpen
    global show_emboss
    global emboss
    global flip_H
    global flip_V

    transformedImage = None

    # reset flags and variables
    image_resized = False
    rotationApplied = False
    anchor = None
    selectedAngle.set('0.0')
    currentAngle = 0.0
    lblCurrentAngle['text'] = str(currentAngle)

    # reset flip buttons
    btnFlip_H.deselect()
    flip_H = False
    btnFlip_V.deselect()
    flip_V = False

    # reset all sliders
    imgGamma.set(1.05)
    hue.set(0.0)
    saturation.set(1.0)
    gaussian_blur.set(0)
    posterize.set(6)
    solarize.set(255)
    show_edges.set(0)
    low_threshold.set(50)
    sliderLowThreshold['state'] = 'disabled'
    sobel_threshold.set(0)
    sharpen.set(8.9)
    show_emboss.set(0)
    emboss.set(114)
    sliderEmboss['state'] = 'disabled'

    # close any open histogram window
    plt.close()

def show_histogram_window():
    # reset the error label's text
    if lblError['text'] == 'Error':
        lblError['text'] = ''

    plt.close()

    plt.figure()
    plt.title('Colour Histogram')
    plt.xlabel('Bins')
    plt.ylabel('Number of pixels')

    colors = ('r', 'g', 'b')

    for i,col in enumerate(colors):
        if not transformedImage is None:
            img = transformedImage
        else:
            img = currentImage

        hist = caer.core.cv.calcHist([img], [i], None, [256], [0,256])

        plt.plot(hist, color=col)
        plt.xlim([0,256])

    plt.show()

def main():
    global root
    global canvas
    global fig
    global subplot
    global lblError
    global lblFileName
    global currentImage
    global previous_image
    global transformedImage
    global imageSelection
    global popup_menu_image
    global reload_local_file
    global showAxis
    global resizedImgBtn
    global flip_H
    global flip_V
    global btnFlip_H
    global btnFlip_V
    global rotateImgBtn
    global selectedSize
    global selectedAngle
    global rotationAngle
    global anchor
    global anchorSelection
    global rotationApplied
    global lblCurrentAngle
    global currentAngle
    global imgGamma
    global hue
    global saturation
    global sharpen
    global gaussian_blur
    global posterize
    global solarize
    global show_edges
    global low_threshold
    global sliderLowThreshold
    global show_emboss
    global emboss
    global embossKernel
    global sliderEmboss
    global sobel_threshold

    # create our main window
    root = Tk()
    root.config(background='white')
    root.title('CAER Image GUI - Python v' + pythonVersion)
    root.geometry('1024x768')
    root.bind('<Destroy>', on_exit)

    # the following works for a single screen setup
    # if using a multi-screen setup then see the following link:
    # Ref: https://stackoverflow.com/questions/3129322/how-do-i-get-monitor-resolution-in-python/56913005#56913005
    screenDPI = root.winfo_fpixels('1i')
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    currentImage = None
    previous_image = 'Island'
    transformedImage = None
    reload_local_file = False
    rotationApplied = False
    anchor = None
    showAxis = False
    flip_H, flip_V = False, False
    currentAngle = 0.0

    embossKernel = caer.data.np.array([[0, 1, 0], [0, 0, 0], [0, -1, 0]])

    # bind the 'q' keyboard key to quit
    root.bind('q', lambda event:root.destroy())

    #-----------------------------------------------------------------------

    # add a frame to hold top controls
    frame1 = Frame(root, background='black')
    frame1.pack(side=TOP, fill=X)

    # create the built-in image selection variable and choices
    imageSelection = StringVar()
    imageChoices = ['Open File >>', 'Bear', 'Beverages', 'Black Cat', 'Camera', 'Gold Fish', 'Guitar', 'Island', 'Mountain', 'Night', 'Puppies', 'Snow', 'Sunrise', 'Sea Turtle', 'Tent']
    imageSelection.set('Island')
    imageSelection.trace('w', show_original_image)

    # create the built-in image selection popup menu
    popup_menu_image = OptionMenu(frame1, imageSelection, *imageChoices)
    popup_menu_image['width'] = 10
    popup_menu_image['bg'] = 'lightgreen'
    popup_menu_image.pack(side=LEFT, padx=2)

    # create a button to re-size the image
    resizedImgBtn = Button(frame1, text='Resize', width=6, bg='lightgrey', relief=RAISED, command=resize_image)
    resizedImgBtn.pack(side=LEFT, padx=2, pady=2)

    # create an entry box for re-size dimensions
    selectedSize = StringVar()
    resizedImgSize = Entry(frame1, justify=CENTER, textvariable=selectedSize, font='Helvetica 10', width=10, bg='white', relief=RAISED)
    resizedImgSize.pack(side=LEFT, padx=2, pady=2)
    selectedSize.set('')

    # create a button to rotate the image
    rotateImgBtn = Button(frame1, text='Rotate', width=6, bg='lightgrey', relief=RAISED, command=show_rotated_image)
    rotateImgBtn.pack(side=LEFT, padx=2, pady=2)

    # create a label for the rotation angle
    lblAngle = Label(frame1, text=u'\u2220', fg='yellow', bg='black', font='Helvetica 10')
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
    lblAnchor = Label(frame1, text=u'\u2693', fg='yellow', bg='black', font='Helvetica 10')
    lblAnchor.pack(side=LEFT, padx=2, pady=2)

    # create the rotation anchor selection variable and choices
    anchorSelection = StringVar()
    anchorChoices = ['BottomLeft', 'BottomMiddle', 'BottomRight', 'Center', 'MiddleLeft', 'MiddleRight', 'TopLeft', 'TopMiddle', 'TopRight']
    anchorSelection.set('Center')

    # create the anchor selection popup menu
    popup_menu_anchor = OptionMenu(frame1, anchorSelection, *anchorChoices)
    popup_menu_anchor['width'] = 12
    popup_menu_anchor.pack(side=LEFT, padx=2)

    # create a label to show the name of the local image file opened by user
    lblFileName = Label(frame1, text='', fg='yellow', bg='black', font='Helvetica 10')
    lblFileName.pack(side=RIGHT, padx=10, pady=2)

    #-----------------------------------------------------------------------

    # add a frame to hold side controls, screen attributes and the Error labels
    frame2 = Frame(root, background='black')
    frame2.pack(side=RIGHT, fill=Y)

    # create the image gamma slider control
    imgGamma = DoubleVar()
    sliderGamma = Scale(frame2, label='Gamma', variable=imgGamma, troughcolor='blue', from_=0.1, to=2.0, resolution=0.05, sliderlength=15, showvalue=False, orient=HORIZONTAL, command=adjust_ghsps)
    sliderGamma.pack(side=TOP, anchor=E, padx=2)
    imgGamma.set(1.05)

    # create the image hue slider control
    hue = DoubleVar()
    sliderHue = Scale(frame2, label='Hue', variable=hue, troughcolor='blue', from_=-0.5, to=0.5, resolution=0.05, sliderlength=15, showvalue=False, orient=HORIZONTAL, command=adjust_ghsps)
    sliderHue.pack(side=TOP, anchor=E, padx=2, pady=3)
    hue.set(0.0)

    # create the image saturation slider control
    saturation = DoubleVar()
    sliderSaturation = Scale(frame2, label='Saturation', variable=saturation, troughcolor='blue', from_=0.0, to=2.0, resolution=0.1, sliderlength=15, showvalue=False, orient=HORIZONTAL, command=adjust_ghsps)
    sliderSaturation.pack(side=TOP, anchor=E, padx=2)
    saturation.set(1.0)

    # create the image sharpen slider control
    sharpen = DoubleVar()
    sliderSharpen = Scale(frame2, label='Sharpen', variable=sharpen, troughcolor='blue', from_=7.9, to=9.9, resolution=0.05, sliderlength=15, showvalue=False, orient=HORIZONTAL, command=set_sharpen_kernel)
    sliderSharpen.pack(side=TOP, padx=2, pady=3)
    sharpen.set(8.9)

    # create the image Gaussian Blur slider control
    gaussian_blur = IntVar()
    sliderGaussianBlur = Scale(frame2, label='Gaussian Blur', variable=gaussian_blur, troughcolor='blue', from_=0, to=10, resolution=2, sliderlength=15, showvalue=False, orient=HORIZONTAL, command=adjust_ghsps)
    sliderGaussianBlur.pack(side=TOP, padx=2)
    gaussian_blur.set(0)

    # create the image posterize slider control
    posterize = IntVar()
    sliderPosterize = Scale(frame2, label='Posterize', variable=posterize, troughcolor='blue', from_=6, to=1, resolution=1, sliderlength=15, showvalue=False, orient=HORIZONTAL, command=adjust_ghsps)
    sliderPosterize.pack(side=TOP, padx=2, pady=3)
    posterize.set(6)

    # create the image solarize slider control
    solarize = IntVar()
    sliderSolarize = Scale(frame2, label='Solarize', variable=solarize, troughcolor='blue', from_=255, to=0, resolution=1, sliderlength=15, showvalue=False, orient=HORIZONTAL, command=adjust_ghsps)
    sliderSolarize.pack(side=TOP, padx=2)
    solarize.set(255)

    # create the image sobel threshold slider control
    sobel_threshold = IntVar()
    sliderSobelThreshold = Scale(frame2, label='Sobel Gradient', variable=sobel_threshold, troughcolor='blue', from_=0, to=4, resolution=1, sliderlength=15, showvalue=False, orient=HORIZONTAL, command=adjust_ghsps)
    sliderSobelThreshold.pack(side=TOP, padx=2, pady=3)
    sobel_threshold.set(0)

    # add 'Edges' checkbox
    show_edges = IntVar()
    chbShowEdges = Checkbutton(frame2, text='Edges', variable=show_edges, width=7, command=set_edges)
    chbShowEdges.pack(side=TOP, padx=2)
    show_edges.set(0)

    # create the image edges low threshold slider control
    low_threshold = IntVar()
    sliderLowThreshold = Scale(frame2, label='Edges Threshold', variable=low_threshold, troughcolor='blue', from_=100, to=0, resolution=1, sliderlength=15, showvalue=False, orient=HORIZONTAL, command=adjust_ghsps)
    sliderLowThreshold.pack(side=TOP, padx=2, pady=3)
    low_threshold.set(50)

    # add 'Emboss' checkbox
    show_emboss = IntVar()
    chbShowEmboss = Checkbutton(frame2, text='Emboss', variable=show_emboss, width=7, command=set_emboss)
    chbShowEmboss.pack(side=TOP, padx=2)
    show_emboss.set(0)

    # create the image emboss slider control
    emboss = IntVar()
    sliderEmboss = Scale(frame2, label='Emboss Threshold', variable=emboss, troughcolor='blue', from_=128, to=99, resolution=1, sliderlength=15, showvalue=False, orient=HORIZONTAL, command=adjust_ghsps)
    sliderEmboss.pack(side=TOP, padx=2, pady=3)
    emboss.set(114)

    lblScreen = Label(frame2, text='Screen', fg='grey', bg='black', font='Helvetica 9')
    lblScreen.pack(side=TOP, anchor=CENTER, pady=7)

    lblResolution = Label(frame2, text='res: ' + str(screen_width) + ' x ' + str(screen_height), fg='grey', bg='black', font='Helvetica 9')
    lblResolution.pack(side=TOP, anchor=CENTER)

    lblDPI = Label(frame2, text='dpi: ' + str(int(screenDPI)), fg='grey', bg='black', font='Helvetica 9')
    lblDPI.pack(side=TOP, anchor=CENTER)

    # add exit button
    exitBtn = Button(frame2, text='Exit', width=7, fg='red', bg='lightgrey', relief=RAISED, command=root.destroy)
    exitBtn.pack(side=BOTTOM, anchor=CENTER, pady=5)

    lblError = Label(frame2, text='', fg='red', bg='black', font='Helvetica 12')
    lblError.pack(side=BOTTOM, anchor=CENTER, pady=10)

    #-----------------------------------------------------------------------

    # create matplotlib figure, subplot, canvas and toolbar
    fig = Figure(figsize=(640//screenDPI, 427//screenDPI), dpi=int(screenDPI))
    subplot = fig.add_subplot(111)
    subplot.xaxis.set_visible(False), subplot.yaxis.set_visible(False)
    fig.set_tight_layout(True)

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()

    toolbar = NavigationToolbar2Tk(canvas, root)
    toolbar._Spacer()
    toolbar._Button('Show Axis', None, toggle=True, command=refresh_axis)
    toolbar._Spacer()
    toolbar._Button('Reload Image', None, toggle=False, command=reload_image)
    toolbar._Spacer()
    btnFlip_H = toolbar._Button('FlipH', None, toggle=True, command=flip_image_horizontally)
    toolbar._Spacer()
    btnFlip_V = toolbar._Button('FlipV', None, toggle=True, command=flip_image_vertically)
    toolbar._Spacer()
    toolbar._Button('Histogram', None, toggle=False, command=show_histogram_window)
    toolbar.update()

    canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)

    #-----------------------------------------------------------------------

    # set the minimum window size to the current size
    root.update()
    root.minsize(root.winfo_width(), root.winfo_height())

    show_original_image()

    root.mainloop()

def on_exit(*args):
    plt.close()

if __name__=='__main__':
    main()
    caer.core.cv.destroyAllWindows()