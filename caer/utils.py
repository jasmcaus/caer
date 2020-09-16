# Author: Jason Dsouza
# Github: http://www.github.com/jasmcaus

#pylint: disable=bare-except

# Importing the necessary packages
# import os
from urllib.request import urlopen
import os
import time
import math
import cv2 as cv
import numpy as np

# # For Python 2.7
# import sys
# if sys.version_info.major == 2:
#     from urllib2 import urlopen
# For Python

def readImg(image_path, resized_img_size=None, channels=1):
    if not os.path.exists(image_path):
        raise FileNotFoundError('[ERROR] The image file was not found')

    image_array = cv.imread(image_path)

    # [INFO] Using the following piece of code results in a 'None' in the training set
    # if image_array == None:
    #     pass
    if channels == 1:
        image_array = cv.cvtColor(image_array, cv.COLOR_BGR2GRAY)
    if resized_img_size is not None:
        image_array = cv.resize(image_array, (resized_img_size,resized_img_size))
    return image_array

def get_classes_from_dir(DIR):
    if len(os.listdir(DIR)) == 0:
        raise ValueError('[ERROR] The specified directory does not seem to have any folders in it')
    else:
        classes = [i for i in os.listdir(DIR)]
        return classes

def compute_mean(DIR, channels):
    if channels == 3:
        rMean, gMean, bMean = 0,0,0
    if channels == 1:
        mean = 0
    count = 0

    if os.path.exists(DIR) is False:
        raise ValueError('The specified directory does not exist', DIR)

    for root, _, files in os.walk(DIR):
        for file in files:
            if file.endswith('.png') or file.endswith('.jpg'):
                count += 1
                filepath = root + os.sep + file
                img = cv.imread(filepath)
                if channels == 3:
                    b,g,r = cv.mean(img.astype('float32'))[:3]
                    rMean += r
                    bMean += b
                    gMean += g
                if channels == 1:
                    mean += cv.mean(img.astype('float32'))[0]

    # Computing average mean
    if channels == 3:
        rMean /= count
        bMean /= count 
        gMean /= count
        return rMean, bMean, gMean

    if channels == 1:
        mean /= count
        return mean

def saveNumpy(name, x):
    """
    Saves an array to a .npy file
    Converts to Numpy (if not already)
    """

    x = np.array(x)
    if '.npy' in name:
        np.save(name, x)
    elif '.npz' in name:
        np.savez_compressed(name, x)

def train_val_split(X,y,val_ratio=.2):
    """
    Returns X_train, X_val, y_train, y_val
    """
    try:
        from sklearn.model_selection import train_test_split    
        X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=val_ratio,random_state = 2)
        return X_train, X_val, y_train, y_val
    except ModuleNotFoundError:
        print('[ERROR] The Sklearn Python package needs to be installed')

def extract_frames(input_folder, output_folder, IMG_SIZE=None, label_counter = None, video_count=None, frames_per_sec=None, dest_filetype='jpg'):
    """ Function to extract frames from videos within a directory
    and save them as separate frames in an output directory.
    Args:
        input_folder: Input video directory.
        output_folder: Output directory to save the frames.
        IMG_SIZE: Destination Image Size
        label_counter: Starting label counter
        video_count: Number of videos to process.
        frames_per_sec: Number of frames to process per second. 
        dest_filetype: Processed image filetype (png, jpg) --> Default: png
    Returns:
        label_counter (after processing)
    """

    dest_filetype.replace('.','')
    processed_videos = 0
    vid_count = 0 # to check if < video_count

    if os.path.exists(input_folder) is False:
        raise ValueError('Input folder does not exist', input_folder)

    if label_counter is None:
        label_counter = 0

    if video_count is None:
        video_count = len(os.listdir(input_folder))

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    start = time.time()

    for root, _, files in os.walk(input_folder):
        for file in files:
            if vid_count < video_count:
                if file.endswith('.mp4') or file.endswith('.avi'): # if a video file
                    vid_filepath = root + os.sep + file
                    capture = cv.VideoCapture(vid_filepath)
                    video_frame_counter = 0

                    # Find the number of frames and FPS
                    video_frame_count = int(capture.get(cv.CAP_PROP_FRAME_COUNT)) - 1
                    video_fps = math.ceil(capture.get(cv.CAP_PROP_FPS))

                    print(f'{vid_count+1}. Reading \'{file}\'. Number of frames: {video_frame_count}. FPS: {video_fps}')

                    if frames_per_sec is not None:
                        interval= determine_interval(video_fps/frames_per_sec) # eg: 30//15
                        print('Interval: ', interval)
                    # if frames_per_sec is None, we assume that each frame should be processed
                    else:
                        interval = 1
                    
                    # Start converting the video
                    while capture.isOpened():
                        _, frame = capture.read()

                        if IMG_SIZE is not None:                    
                            frame = cv.resize(frame, (IMG_SIZE,IMG_SIZE))
                        
                        # Write the results back to output location as per specified frames per second
                        if video_frame_counter % interval == 0:
                            cv.imwrite(f'{output_folder}/{label_counter}.{dest_filetype}', frame)
                            video_frame_counter += 1
                            label_counter += 1
                            print('Frame counter: ', video_frame_counter)
                        
                        video_frame_counter += 1

                        # If there are no more frames left
                        if video_frame_counter > (video_frame_count-1):
                            capture.release()
                            processed_videos += 1
                            break

    end = time.time()
    # Printing stats
    taken = end-start
    print('[INFO] {} videos extracted in {:.2f} seconds'.format(processed_videos, taken ))

    return label_counter

def determine_interval(x):
    y = '{:.1f}'.format(x)
    inde = y.find('.') + 1
    if inde == -1: # if no '.' (if an integer)
        return x
    if int(y[inde]) < 5:
        return math.floor(x)
    else:
        return math.ceil(x)

def sort_dict(unsorted_dict, descending=False):
    """
    Sorts a dictionary in ascending order (if descending = False) or descending order (if descending = True)
    """
    return sorted(unsorted_dict.items(), key=lambda x:x[1], reverse=descending)

def plotAcc(histories):
    """
    Plots the model accuracies as 2 graphs
    """
    import matplotlib.pyplot as plt 
    acc = histories.history['acc']
    val_acc = histories.history['val_acc']
    loss = histories.history['loss']
    val_loss = histories.history['val_loss']

    epochs = range(1, len(acc)+1)

    # Plotting Accuracy
    plt.plot(epochs, acc, 'b', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    # Plotting Loss
    plt.plot(epochs, loss, 'b', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.show()

# -------------------- OPENCV IMAGE-SPECIFIC METHODS --------------------- 

def translate(image, x, y):
    # Defines the translation matrix and performs the translation
    matrix = np.float32([[1, 0, x], [0, 1, y]])
    shifted = cv.warpAffine(image, matrix, (image.shape[1], image.shape[0]))

    return shifted

def rotate(image, angle, centre=None, scale=1.0):
    # Grabs the dimensions of the image
    (height, width) = image.shape[:2]

    # If no centre is specified, we grab the centre coordinates of the image
    if centre is None:
        centre = (width // 2, height // 2)

    # Rotates the image
    M = cv.getRotationMatrix2D(centre, angle, scale)
    rotated = cv.warpAffine(image, M, (width, height))

    return rotated

def rotate_bound(image, angle):
    # Grabs the dimensions of the image and then determines the centre
    (h, w) = image.shape[:2]
    (cX, cY) = (w / 2, h / 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # Adjusts the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # Performs the actual rotation and returns the image
    return cv.warpAffine(image, M, (nW, nH))

def resize(image, width=None, height=None, interpolation=cv.INTER_AREA):
    """
    Resizes the image while maintaing the aspect ratio of the original image
    """
    # Grabs the image dimensions 
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, we return the original image
    if width is None and height is None:
        return image

    # If  width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # If height is None
    else:
        # Calculates the ratio of the width and constructs the dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # Resizes the image
    resized = cv.resize(image, dim, interpolation=interpolation)

    return resized

def toMatplotlib(image):
    """
    Converts BGR image ordering to RGB ordering
    """
    return cv.cvtColor(image, cv.COLOR_BGR2RGB)

def url_to_image(url, flag=cv.IMREAD_COLOR):
    # Converts the image to a Numpy array and reads it in OpenCV
    response = urlopen(url)
    image = np.asarray(bytearray(response.read()), dtype="uint8")
    image = cv.imdecode(image, flag)

    return image

def canny(image, sigma=0.33):
    # computes the median of the single channel pixel intensities
    med = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * med))
    upper = int(min(255, (1.0 + sigma) * med))
    edges = cv.Canny(image, lower, upper)

    return edges