# Tested as working in Windows 10 with python v3.6.8
# Resizing, flipping and rotating controls are set to manipulate the currently displayed image
# Hue, Saturation and Motion Blur slider controls are only enabled for the original Sunrise image

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure

from tkinter import *
import platform
import caer

pythonVersion = platform.python_version()

# Standard 640x427 test image that ships out-of-the-box with caer
sunrise = caer.data.sunrise(rgb=True)

def show_original_image():
    global currentImage
    global originalImgBtn
    global resizedImgBtn
    global flipHImgBtn
    global flipVImgBtn
    global flipHVImgBtn
    global rotateImgBtn
    global sliderHue
    global hue
    global sliderSaturation
    global saturation
    global sliderMotionBlur
    global motionBlur

    if originalImgBtn['bg'] == 'lightgrey':
        originalImgBtn['bg'] = 'lightblue'
        sliderHue['state'] = 'normal'
        hue.set(0)
        sliderSaturation['state'] = 'normal'
        saturation.set(1)
        sliderMotionBlur['state'] = 'normal'
        motionBlur.set(0)

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
    currentImage.cspace = 'rgb'
    image_show(currentImage)

def show_resized_image():
    global currentImage
    global originalImgBtn
    global resizedImgBtn
    global flipHImgBtn
    global flipVImgBtn
    global flipHVImgBtn
    global rotateImgBtn
    global sliderHue
    global sliderSaturation
    global sliderMotionBlur

    tempSize = selectedSize.get()

    if 'x' in tempSize:
        size = tempSize.replace(' ', '').split('x')

        try:
            if resizedImgBtn['bg'] == 'lightgrey':
                resizedImgBtn['bg'] = 'lightblue'
                sliderHue['state'] = 'disabled'
                sliderSaturation['state'] = 'disabled'
                sliderMotionBlur['state'] = 'disabled'

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

            # Resize the image without preserving aspect ratio
            resized = caer.resize(currentImage, target_size=(int(size[0]),int(size[1])), preserve_aspect_ratio=False)
            currentImage = resized
            currentImage.cspace = 'rgb'
            image_show(currentImage)
        except Exception as e:
            print(str(e)) # pass

def show_h_flipped_image():
    global currentImage
    global originalImgBtn
    global resizedImgBtn
    global flipHImgBtn
    global flipVImgBtn
    global flipHVImgBtn
    global rotateImgBtn
    global sliderHue
    global sliderSaturation
    global sliderMotionBlur

    if flipHImgBtn['bg'] == 'lightgrey':
        flipHImgBtn['bg'] = 'lightblue'
        sliderHue['state'] = 'disabled'
        sliderSaturation['state'] = 'disabled'
        sliderMotionBlur['state'] = 'disabled'

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

    hf = caer.transforms.hflip(currentImage)
    currentImage = hf
    currentImage.cspace = 'rgb'
    image_show(currentImage)

def show_v_flipped_image():
    global currentImage
    global originalImgBtn
    global resizedImgBtn
    global flipHImgBtn
    global flipVImgBtn
    global flipHVImgBtn
    global rotateImgBtn
    global sliderHue
    global sliderSaturation
    global sliderMotionBlur

    if flipVImgBtn['bg'] == 'lightgrey':
        flipVImgBtn['bg'] = 'lightblue'
        sliderHue['state'] = 'disabled'
        sliderSaturation['state'] = 'disabled'
        sliderMotionBlur['state'] = 'disabled'

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

    vf = caer.transforms.vflip(currentImage)
    currentImage = vf
    currentImage.cspace = 'rgb'
    image_show(currentImage)

def show_hv_flipped_image():
    global currentImage
    global originalImgBtn
    global resizedImgBtn
    global flipHImgBtn
    global flipVImgBtn
    global flipHVImgBtn
    global rotateImgBtn
    global sliderHue
    global sliderSaturation
    global sliderMotionBlur

    if flipHVImgBtn['bg'] == 'lightgrey':
        flipHVImgBtn['bg'] = 'lightblue'
        sliderHue['state'] = 'disabled'
        sliderSaturation['state'] = 'disabled'
        sliderMotionBlur['state'] = 'disabled'

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

    hvf = caer.transforms.hvflip(currentImage)
    currentImage = hvf
    currentImage.cspace = 'rgb'
    image_show(currentImage)

def show_rotated_image():
    global currentImage
    global originalImgBtn
    global resizedImgBtn
    global flipHImgBtn
    global flipVImgBtn
    global flipHVImgBtn
    global rotateImgBtn
    global sliderHue
    global sliderSaturation
    global sliderMotionBlur

    angle = selectedAngle.get()

    if angle == '':
        angle = '0'

    anchor = None # Center point

    tempAnchorPoint = anchorSelection.get()

    if tempAnchorPoint == 'Top Left':
        anchor = (0, 0)
    elif tempAnchorPoint == 'Top Middle':
        anchor = (int(currentImage.width / 2), 0)
    elif tempAnchorPoint == 'Top Right':
        anchor = (currentImage.width, 0)
    elif tempAnchorPoint == 'Middle Left':
        anchor = (0, int(currentImage.height / 2))
    elif tempAnchorPoint == 'Middle Right':
        anchor = (currentImage.width, int(currentImage.height / 2))
    elif tempAnchorPoint == 'Bottom Left':
        anchor = (0, currentImage.height)
    elif tempAnchorPoint == 'Bottom Middle':
        anchor = (int(currentImage.width / 2), currentImage.height)
    elif tempAnchorPoint == 'Bottom Right':
        anchor = (currentImage.width, currentImage.height)

    try:
        if rotateImgBtn['bg'] == 'lightgrey':
            rotateImgBtn['bg'] = 'lightblue'
            sliderHue['state'] = 'disabled'
            sliderSaturation['state'] = 'disabled'
            sliderMotionBlur['state'] = 'disabled'

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

        rot = caer.transforms.rotate(currentImage, int(angle), rotPoint=anchor)
        currentImage = rot
        currentImage.cspace = 'rgb'
        image_show(currentImage)
    except:
        pass

def image_show(tens):
    global canvas
    global subplot

    subplot.clear()
    subplot.xaxis.set_ticks([]), subplot.yaxis.set_ticks([])  # Hides the graph ticks and x / y axis
    subplot.imshow(tens)
    canvas.draw()

def adjust_hue(*args):
    global currentImage

    currentImage = caer.transforms.adjust_hue(sunrise, hue_factor=float(args[0]))
    image_show(currentImage)

def adjust_saturation(*args):
    global currentImage

    currentImage = caer.transforms.adjust_saturation(sunrise, saturation_factor=float(args[0]))
    image_show(currentImage)

def adjust_motion_blur(*args):
    global currentImage

    currentImage = caer.transforms.sim_motion_blur(sunrise, speed_coeff=float(args[0]))
    currentImage.cspace = 'rgb'
    image_show(currentImage)

def main():
    global root
    global canvas
    global subplot
    global currentImage
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
    global sliderHue
    global hue
    global sliderSaturation
    global saturation
    global sliderMotionBlur
    global motionBlur

    root = Tk()
    root.config(background='white')
    root.title('CAER Sunrise GUI Test - Python v' + pythonVersion)
    root.geometry('1024x768')

    currentImage = None

    # bind the 'q' keyboard key to quit
    root.bind('q', lambda event:root.destroy())

    # add a frame to hold buttons
    frame1 = Frame(root, background='black')
    frame1.pack(side='top', fill=X)

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

    selectedAngle = StringVar()
    rotationAngle = Entry(frame1, justify=CENTER, textvariable=selectedAngle, font='Helvetica 10', width=4, bg='white', relief=RAISED)
    rotationAngle.pack(side=LEFT, padx=2, pady=2)
    selectedAngle.set('45')

    # create a label for the rotation anchor
    lblAnchor = Label(frame1, text='Anchor', fg='yellow', bg='black', font='Helvetica 8')
    lblAnchor.pack(side=LEFT, padx=2, pady=2)

    # create the rotation anchor selection variable
    anchorSelection = StringVar()
    anchorChoices = { 'Bottom Left', 'Bottom Middle', 'Bottom Right', 'Center', 'Middle Left', 'Middle Right', 'Top Left', 'Top Middle', 'Top Right'}
    anchorSelection.set('Center')

    # create the anchor selection popup menu
    popup_menu_anchor = OptionMenu(frame1, anchorSelection, *anchorChoices)
    # popup_menu_anchor.config(bg = 'lightgreen')
    popup_menu_anchor.pack(side=LEFT, padx=2)

    # create the image hue slider control
    hue = DoubleVar()
    sliderHue = Scale(frame1, label='Hue', variable=hue, troughcolor='blue', from_=-0.5, to=0.5, resolution=0.05, sliderlength=15, showvalue=False, orient=HORIZONTAL, command=adjust_hue)
    sliderHue.pack(side=LEFT, padx=10, pady=2)
    hue.set(0)

    # create the image saturation slider control
    saturation = DoubleVar()
    sliderSaturation = Scale(frame1, label='Saturation', variable=saturation, troughcolor='blue', from_=0, to=2.0, resolution=0.1, sliderlength=15, showvalue=False, orient=HORIZONTAL, command=adjust_saturation)
    sliderSaturation.pack(side=LEFT, padx=6, pady=2)
    saturation.set(1)

    # create the image motion blur slider control
    motionBlur = DoubleVar()
    sliderMotionBlur = Scale(frame1, label='Motion Blur', variable=motionBlur, troughcolor='blue', from_=0, to=0.2, resolution=0.05, sliderlength=15, showvalue=False, orient=HORIZONTAL, command=adjust_motion_blur)
    sliderMotionBlur.pack(side=LEFT, padx=6, pady=2)
    motionBlur.set(0)

    exitBtn = Button(frame1, text='Exit', width=6, bg='lightgrey', relief=RAISED, command=root.destroy)
    exitBtn.pack(side=RIGHT, padx=4, pady=2)

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
