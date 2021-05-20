# A simple TkInter GUI app example, designed to showcase some of Caer's features
# This should only be used as a base to create a new GUI.

# Requirements: Python, Caer, Matplotlib

# Run it either via IDLE or from command prompt / terminal with one of these commands:
# - 'python caer_gui.py'
# - 'python -m caer_gui'
# - 'python3 caer_gui.py'
# - 'python3 -m caer_gui'

# Tested as working in Windows 10 with python v3.6.8 and Kubuntu Linux with python v3.6.8. 
# You can select one of 9 built-in images to display (startup has `caer.data.sland` selected as default)
# Selecting any of the images, at any point in time, will always start with a fresh original image and reset 
# controls.
# Replace with or add your own image(s) by following the instructions here: https://caer.readthedocs.io/en/latest/api/io.html
# The above will require that you modify main() and show_original_image() functions.
# All function controls are set to manipulate the currently displayed image.
# Edges and Emboss effects are mutually exclusive (you can only have one applied at the time).
# Gamma, Hue, Saturation, Sharpness, Posterize, Solarize, Edges and Emboss effects are currently somewhat unique 
# and, when applied to the image, will follow the following rule:
#   - Applying 'Resize', 'Rotate' and/or any of the 'Flip' functions to transformed image will preserve that image 
#     and have all those effects reset
#
# The above mentioned could possibly be corrected by converting all those buttons to checkboxes and applying all the effects within a single function (just use the current adjust_ghsps() function)
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
                currentImage = caer.to_tensor(transformedImage, cspace = 'rgb')
                reset_ghsps()

            # Resize the image without preserving aspect ratio
            currentImage = caer.to_tensor(caer.resize(currentImage, target_size=(int(size[0]),int(size[1])), preserve_aspect_ratio=False), cspace = 'rgb')

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
        currentImage = caer.to_tensor(transformedImage, cspace = 'rgb')
        reset_ghsps()

    currentImage = caer.to_tensor(caer.transforms.hflip(currentImage), cspace = 'rgb')

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
        currentImage = caer.to_tensor(transformedImage, cspace = 'rgb')
        reset_ghsps()

    currentImage = caer.to_tensor(caer.transforms.vflip(currentImage), cspace = 'rgb')

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
        currentImage = caer.to_tensor(transformedImage, cspace = 'rgb')
        reset_ghsps()

    currentImage = caer.to_tensor(caer.transforms.hvflip(currentImage), cspace = 'rgb')

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
            rot = caer.to_tensor(caer.transforms.rotate(transformedImage, float(currentAngle), rotPoint=anchor), cspace = 'rgb')
            if not rotationApplied:
                currentImage = caer.to_tensor(transformedImage, cspace = 'rgb')
                reset_ghsps()
        else:
            rot = caer.to_tensor(caer.transforms.rotate(currentImage, float(currentAngle), rotPoint=anchor), cspace = 'rgb')

        image_show(rot)
    except Exception as e:
        print(str(e))

def image_show(tens):
    subplot.clear()
    subplot.imshow(tens) # optionally add aspect='auto' to switch to automatic aspect mode
    canvas.draw()

def refresh_axis():
    global showAxis

    # Hide / Show the graph's x / y axis
    if not showAxis:
        subplot.xaxis.set_visible(True), subplot.yaxis.set_visible(True)
        showAxis = True
    else:
        subplot.xaxis.set_visible(False), subplot.yaxis.set_visible(False)
        showAxis = False

    fig.canvas.draw()

def set_edges():
    global show_emboss

    if show_edges.get() == 1:
        show_emboss.set(0)

    adjust_ghsps()

def set_emboss():
    global show_edges

    if show_emboss.get() == 1:
        show_edges.set(0)
    
    adjust_ghsps()

def adjust_ghsps(*args):
    global transformedImage

    if not currentImage is None:
        # apply all transformations to currently displayed image
        transformedImage = caer.to_tensor(caer.transforms.adjust_hue(currentImage, hue.get()), cspace = 'rgb')
        transformedImage = caer.to_tensor(caer.transforms.adjust_saturation(transformedImage, saturation.get()), cspace = 'rgb')
        transformedImage = caer.to_tensor(caer.transforms.adjust_gamma(transformedImage, imgGamma.get()), cspace = 'rgb')

        if sharpen.get() != 8.9:
            kernel = caer.data.np.array([[-1, -1, -1], [-1, sharpen.get(), -1], [-1, -1, -1]])
            transformedImage = caer.to_tensor(caer.core.cv.filter2D(transformedImage, -1, kernel), cspace = 'rgb')

        gb = gaussian_blur.get()

        if gb > 1:
            transformedImage = caer.to_tensor(caer.core.cv.GaussianBlur(transformedImage, (gb + 1, gb + 1), caer.core.cv.BORDER_DEFAULT), cspace = 'rgb')

        if posterize.get() < 6:
            transformedImage = caer.to_tensor(caer.transforms.posterize(transformedImage, posterize.get()), cspace = 'rgb')

        if solarize.get() < 255:
            transformedImage = caer.to_tensor(caer.transforms.solarize(transformedImage, solarize.get()), cspace = 'rgb')

        if show_edges.get() == 1:
            transformedImage = caer.to_tensor(caer.core.cv.Canny(transformedImage, low_threshold.get(), low_threshold.get() * 2), cspace = 'rgb')

        if show_emboss.get() == 1:
            kernel = caer.data.np.array([[0, 1, 0], [0, 0, 0], [0, -1, 0]])
            transformedImage = caer.to_tensor(caer.core.cv.filter2D(transformedImage, -1, kernel) + emboss.get(), cspace = 'rgb')

        if rotationApplied:
            show_rotated_image(True)
        else:
            image_show(transformedImage)

def reset_ghsps():
    global transformedImage
    global imgGamma
    global hue
    global saturation
    global gaussian_blur
    global posterize
    global solarize
    global show_edges
    global low_threshold
    global sharpen
    global show_emboss
    global emboss

    transformedImage = None

    # reset all sliders
    imgGamma.set(1.0)
    hue.set(0.0)
    saturation.set(1.0)
    gaussian_blur.set(0)
    posterize.set(6)
    solarize.set(255)
    show_edges.set(0)
    low_threshold.set(50)
    sharpen.set(8.9)
    show_emboss.set(0)
    emboss.set(114)

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
    global gaussian_blur
    global posterize
    global solarize
    global show_edges
    global low_threshold
    global sharpen
    global show_emboss
    global emboss

    # create our window
    root = Tk()
    root.config(background='white')
    root.title('CAER GUI Test - Python v' + pythonVersion)
    root.geometry('1024x768')

    # the following works for a single screen setup
    # if using a multi-screen setup then see the following link:
    # Ref: https://stackoverflow.com/questions/3129322/how-do-i-get-monitor-resolution-in-python/56913005#56913005
    screenDPI = root.winfo_fpixels('1i')
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

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
    selectedSize.set('1280x854')

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

    # add a frame to hold side controls and screen attributes labels
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

    # create the image sharpen slider control
    sharpen = DoubleVar()
    sliderSharpen = Scale(frame2, label='Sharpen', variable=sharpen, troughcolor='blue', from_=7.9, to=9.9, resolution=0.05, sliderlength=15, showvalue=False, orient=HORIZONTAL, command=adjust_ghsps)
    sliderSharpen.pack(side=TOP, padx=2, pady=5)
    sharpen.set(8.9)

    # create the image Gaussian Blur slider control
    gaussian_blur = IntVar()
    sliderGaussianBlur = Scale(frame2, label='Gaussian Blur', variable=gaussian_blur, troughcolor='blue', from_=0, to=10, resolution=2, sliderlength=15, showvalue=False, orient=HORIZONTAL, command=adjust_ghsps)
    sliderGaussianBlur.pack(side=TOP, padx=2, pady=5)
    gaussian_blur.set(0)

    # create the image posterize slider control
    posterize = IntVar()
    sliderPosterize = Scale(frame2, label='Posterize', variable=posterize, troughcolor='blue', from_=6, to=1, resolution=1, sliderlength=15, showvalue=False, orient=HORIZONTAL, command=adjust_ghsps)
    sliderPosterize.pack(side=TOP, padx=2, pady=5)
    posterize.set(6)

    # create the image solarize slider control
    solarize = IntVar()
    sliderSolarize = Scale(frame2, label='Solarize', variable=solarize, troughcolor='blue', from_=255, to=0, resolution=1, sliderlength=15, showvalue=False, orient=HORIZONTAL, command=adjust_ghsps)
    sliderSolarize.pack(side=TOP, padx=2, pady=5)
    solarize.set(255)

    # add 'Edges' checkbox
    show_edges = IntVar()
    chbShowEdges = Checkbutton(frame2, text='Edges', variable=show_edges, width=7, command=set_edges)
    chbShowEdges.pack(side=TOP, padx=2, pady=5)
    show_edges.set(0)

    # create the image edges low threshold slider control
    low_threshold = IntVar()
    sliderLowThreshold = Scale(frame2, label='Edges Threshold', variable=low_threshold, troughcolor='blue', from_=100, to=0, resolution=1, sliderlength=15, showvalue=False, orient=HORIZONTAL, command=adjust_ghsps)
    sliderLowThreshold.pack(side=TOP, padx=2, pady=5)
    low_threshold.set(50)

    # add 'Emboss' checkbox
    show_emboss = IntVar()
    chbShowEmboss = Checkbutton(frame2, text='Emboss', variable=show_emboss, width=7, command=set_emboss)
    chbShowEmboss.pack(side=TOP, padx=2, pady=5)
    show_emboss.set(0)

    # create the image emboss slider control
    emboss = IntVar()
    sliderEmboss = Scale(frame2, label='Emboss Threshold', variable=emboss, troughcolor='blue', from_=128, to=99, resolution=1, sliderlength=15, showvalue=False, orient=HORIZONTAL, command=adjust_ghsps)
    sliderEmboss.pack(side=TOP, padx=2, pady=5)
    emboss.set(114)

    lblScreen = Label(frame2, text='Screen', fg='grey', bg='black', font='Helvetica 9')
    lblScreen.pack(side=TOP, anchor=CENTER, pady=15)

    lblResolution = Label(frame2, text='res: ' + str(screen_width) + 'x' + str(screen_height), fg='grey', bg='black', font='Helvetica 9')
    lblResolution.pack(side=TOP, anchor=CENTER)

    lblDPI = Label(frame2, text='dpi: ' + str(int(screenDPI)), fg='grey', bg='black', font='Helvetica 9')
    lblDPI.pack(side=TOP, anchor=CENTER)

    # add exit button
    exitBtn = Button(frame2, text='Exit', width=7, fg='red', bg='lightgrey', relief=RAISED, command=root.destroy)
    exitBtn.pack(side=BOTTOM, anchor=CENTER, pady=4)

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
    toolbar._Button('Reload Image', None, toggle=False, command=show_original_image)
    toolbar._Spacer()
    toolbar._Button('Show / Hide Axis', None, toggle=True, command=refresh_axis)
    toolbar.update()

    canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)

    #-----------------------------------------------------------------------

    # set the minimum window size to the current size
    root.update()
    root.minsize(root.winfo_width(), root.winfo_height())

    show_original_image()

    root.mainloop()

if __name__=='__main__':
    main()
    caer.core.cv.destroyAllWindows()
