# Simple tkinter GUI app example, designed to showcase some caer features for playing video
# Should only be used as a base to create a new GUI
# It can be re-designed, controls re-grouped and code improved

# Requirements: python3, caer

# Run it either via IDLE or from command prompt / terminal with one of these commands:
# - 'python caer_gui_video.py'
# - 'python -m caer_gui_video'
# - 'python3 caer_gui_video.py'
# - 'python3 -m caer_gui_video'

# You can select the camera source to capture the video from (0 is usually default)
# You can also open and play a video file as well as loop it
# You can take a screenshot of the current video frame

# Tested as working in Windows 10 with python v3.6.8

from tkinter import *
from tkinter import filedialog as fd

import threading
import platform
import caer

class play_file_video_thread(threading.Thread):
   def __init__(self):
      threading.Thread.__init__(self)
   def run(self):
      play_file_video()

class play_camera_video_thread(threading.Thread):
   def __init__(self):
      threading.Thread.__init__(self)
   def run(self):
      play_camera_video()

app_closing = False
pythonVersion = platform.python_version()
caerVersion = caer.__version__

def select_video_source(*args):
    global video_file
    global video_cam
    global sourceSelection

    selectedSource = sourceSelection.get()

    if selectedSource == 'Open File >>':
            try:
                video_file = fd.askopenfilename(filetypes=(('AVI files', '*.avi'),('MKV files', '*.mkv'),('MP4 files', '*.mp4'),('MPG files', '*.mpg'),('WMV files', '*.wmv'),('All files', '*.*')))

                if video_file != '':
                    start_playing_file_video()
                else:
                    sourceSelection.set('None')
                    popup_menu_source['bg'] = 'green'
                    popup_menu_source['bg'] = 'lightgreen'
            except Exception as e:
                print(str(e))
    elif selectedSource != 'None':
        # [-1:] is functional for 0 to 9 indexes, use [7:] instead to cover any index (provided all names still start with 'Camera_')
        video_cam = int(selectedSource[-1:])
        start_playing_camera_video()

def start_playing_file_video():
    try:
        thread1 = play_file_video_thread()
        thread1.setDaemon(True)
        thread1.start()
    except Exception as e:
        print("unable to start play_file_video_thread, " + str(e))

def play_file_video():
    global close_video_window
    global sourceSelection
    global take_a_screenshot
    global checkVarLoop

    if not video_file is None:
        capture1 = None
        close_video_window = False
        popup_menu_source['state'] = 'disabled'
        popup_menu_scale['state'] = 'disabled'
        closeBtn['state'] = 'normal'
        screenshotBtn['state'] = 'normal'
        chbLoop['state'] = 'normal'

        try:
            capture1 = caer.core.cv.VideoCapture(video_file)

            while True:
                isTrue, frame = capture1.read()

                if isTrue:
                    if scaleSelection.get() != '1.00':
                        width = int(frame.shape[1] * float(scaleSelection.get()))
                        height = int(frame.shape[0] * float(scaleSelection.get()))

                        dimensions = (width, height)

                        frame = caer.core.cv.resize(frame, dimensions, interpolation = caer.core.cv.INTER_AREA)

                        caer.core.cv.imshow(video_file, frame)
                    else:
                        caer.core.cv.imshow(video_file, frame)

                    if take_a_screenshot:
                        caer.imsave('./Screenshot_' + str(screenshot_count) + '.png', caer.to_tensor(frame, cspace="bgr"))
                        take_a_screenshot = False
                else:
                    if checkVarLoop.get() == 1:
                        capture1.release()
                        capture1 = caer.core.cv.VideoCapture(video_file)
                    else:
                        break

                if caer.core.cv.waitKey(20) & 0xFF == ord('d') or app_closing or close_video_window:
                    break
        except Exception as e:
            print(str(e))

        if not app_closing:
            popup_menu_source['state'] = 'normal'
            popup_menu_scale['state'] = 'normal'
            closeBtn['state'] = 'disabled'
            screenshotBtn['state'] = 'disabled'
            checkVarLoop.set(0)
            chbLoop['state'] = 'disabled'
            sourceSelection.set('None')

        capture1.release()
        caer.core.cv.destroyAllWindows()

def start_playing_camera_video():
    try:
        thread2 = play_camera_video_thread()
        thread2.setDaemon(True)
        thread2.start()
    except Exception as e:
        print("unable to start play_camera_video_thread, ' + str(e)")

def play_camera_video():
    global close_video_window
    global sourceSelection
    global take_a_screenshot

    if not video_cam is None:
        capture2 = None
        close_video_window = False
        popup_menu_source['state'] = 'disabled'
        popup_menu_scale['state'] = 'disabled'
        closeBtn['state'] = 'normal'
        screenshotBtn['state'] = 'normal'

        try:
            capture2 = caer.core.cv.VideoCapture(video_cam)

            while True:
                isTrue, frame = capture2.read()

                if isTrue:
                    if scaleSelection.get() != '1.00':
                        width = int(frame.shape[1] * float(scaleSelection.get()))
                        height = int(frame.shape[0] * float(scaleSelection.get()))

                        dimensions = (width, height)

                        frame = caer.core.cv.resize(frame, dimensions, interpolation = caer.core.cv.INTER_AREA)

                        caer.core.cv.imshow('Camera_' + str(video_cam), frame)
                    else:
                        caer.core.cv.imshow('Camera_' + str(video_cam), frame)

                    if take_a_screenshot:
                        caer.imsave('./Screenshot_' + str(screenshot_count) + '.png', caer.to_tensor(frame, cspace="bgr"))
                        take_a_screenshot = False
                else:
                    break

                if caer.core.cv.waitKey(20) & 0xFF == ord('d') or app_closing or close_video_window:
                    break
        except Exception as e:
            print(str(e))

        if not app_closing:
            popup_menu_source['state'] = 'normal'
            popup_menu_scale['state'] = 'normal'
            closeBtn['state'] = 'disabled'
            screenshotBtn['state'] = 'disabled'
            sourceSelection.set('None')

        capture2.release()
        caer.core.cv.destroyAllWindows()

def close_video():
    global close_video_window

    close_video_window = True

def take_screenshot():
    global take_a_screenshot
    global screenshot_count

    take_a_screenshot = True
    screenshot_count += 1

def main():
    global root
    global video_file
    global video_cam
    global closeBtn
    global screenshotBtn
    global screenshot_count
    global take_a_screenshot
    global close_video_window
    global sourceSelection
    global scaleSelection
    global popup_menu_source
    global popup_menu_scale
    global checkVarLoop
    global chbLoop

    # create our window
    root = Tk()
    root.config(background='navy')
    root.title('CAER Video GUI - Python v' + pythonVersion)
    root.geometry('390x140')
    root.resizable(0,0)

    # the following works for a single screen setup
    # if using a multi-screen setup then see the following link:
    # Ref: https://stackoverflow.com/questions/3129322/how-do-i-get-monitor-resolution-in-python/56913005#56913005
    screenDPI = root.winfo_fpixels('1i')
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    video_file, video_cam = None, None

    take_a_screenshot = False
    screenshot_count = 0

    # bind the 'q' keyboard key to quit
    root.bind('q', lambda event:root.destroy())

    close_video_window = False

    #-----------------------------------------------------------------------

    # add a frame to hold top controls
    frame1 = Frame(root, background='navy')
    frame1.pack(side=TOP, fill=X)

    # create a label for the video source
    lblSource = Label(frame1, text='Video Source', fg='yellow', bg='navy', font='Helvetica 10')
    lblSource.pack(side=LEFT, padx=10, pady=10)

    # create the video source selection variable and choices
    sourceSelection = StringVar()
    sourceChoices = ['None', 'Open File >>', 'Camera_0', 'Camera_1', 'Camera_2', 'Camera_3']
    sourceSelection.set('None')
    sourceSelection.trace('w', select_video_source)

    # create the video source selection popup menu
    popup_menu_source = OptionMenu(frame1, sourceSelection, *sourceChoices)
    popup_menu_source['width'] = 10
    popup_menu_source['font'] = 'Helvetica 10'
    popup_menu_source['bg'] = 'lightgreen'
    popup_menu_source.pack(side=LEFT, pady=10)

    # create the video scale selection variable and choices
    scaleSelection = StringVar()
    scaleChoices = ['2.00', '1.75', '1.50', '1.00', '0.75', '0.50', '0.25']
    scaleSelection.set('1.00')

    # create the built-in image selection popup menu
    popup_menu_scale = OptionMenu(frame1, scaleSelection, *scaleChoices)
    popup_menu_scale['width'] = 3
    popup_menu_scale['font'] = 'Helvetica 10'
    popup_menu_scale['bg'] = 'lightgreen'
    popup_menu_scale.pack(side=RIGHT, padx=10, pady=10)

    # create a label for the video scaling
    lblScale = Label(frame1, text='Video Scale', fg='yellow', bg='navy', font='Helvetica 10')
    lblScale.pack(side=RIGHT, pady=10)

    #-----------------------------------------------------------------------

    # add a frame to hold caer version and screen attributes labels
    frame2 = Frame(root, background='navy')
    frame2.pack(side=TOP, fill=X)

    lblScreen = Label(frame2, text='Screen : ' + str(screen_width) + ' x ' + str(screen_height) + '   dpi: ' + str(int(screenDPI)), fg='lightgrey', bg='navy', font='Helvetica 9')
    lblScreen.pack(side=LEFT, padx=10, pady=10)

    lblVersion = Label(frame2, text='caer  v' + caerVersion, fg='lightgrey', bg='navy', font='Helvetica 9')
    lblVersion.pack(side=RIGHT, padx=10, pady=10)

    #-----------------------------------------------------------------------

    # add a frame to hold Loop checkbox and Close Video, Screenshot and Exit buttons
    frame3 = Frame(root, background='navy')
    frame3.pack(side=TOP, fill=BOTH)

    # add Close Video button
    closeBtn = Button(frame3, text='Close Video', width=10, fg='blue', bg='lightgrey', state='disabled', relief=RAISED, command=close_video)
    closeBtn.pack(side=LEFT, padx=10, pady=10)

    # add Screenshot button
    screenshotBtn = Button(frame3, text='Screenshot', width=10, fg='blue', bg='lightgrey', state='disabled', relief=RAISED, command=take_screenshot)
    screenshotBtn.pack(side=LEFT, padx=1, pady=10)

    # add Loop checkbox
    checkVarLoop = IntVar()
    chbLoop = Checkbutton(frame3, text='Loop', variable=checkVarLoop, bg='lightgrey', fg='blue', font='Helvetica 8', state='disabled')
    checkVarLoop.set(0)
    chbLoop.pack(side=LEFT, padx=10, pady=8)

    # add Exit button
    exitBtn = Button(frame3, text='Exit', width=10, fg='red', bg='lightgrey', relief=RAISED, command=root.destroy)
    exitBtn.pack(side=RIGHT, anchor=CENTER, padx=10, pady=10)

    #-----------------------------------------------------------------------

    # set the minimum window size to the current size
    root.update()
    root.minsize(root.winfo_width(), root.winfo_height())

    root.mainloop()

if __name__=='__main__':
    main()
    app_closing = True
