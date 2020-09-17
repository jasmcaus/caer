# Author: Jason Dsouza
# Github: http://www.github.com/jasmcaus

# Importing the necessary packages
import os
import math
import time
import cv2 as cv


def extract_frames(input_folder, 
                   output_folder, 
                   IMG_SIZE=None, 
                   label_counter = None, 
                   video_count=None, 
                   frames_per_sec=None, 
                   dest_filetype='jpg'):
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