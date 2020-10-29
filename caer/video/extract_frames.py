# Copyright 2020 The Caer Authors. All Rights Reserved.
#
# Licensed under the MIT License (see LICENSE);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at <https://opensource.org/licenses/MIT>
#
# ==============================================================================

import os
import math
import time
import cv2 as cv

from .._checks import _check_target_size
from ..path import listdir, _acceptable_video_formats, exists


def extract_frames(input_folder, 
                   output_folder, 
                   IMG_SIZE=None, 
                   label_counter = None, 
                   video_count=None, 
                   frames_per_sec=None, 
                   frame_interval=None,
                   dest_filetype='jpg'):
    """ Function to extract frames from videos within a directory
    and save them as separate frames in an output directory.
    Args:
        input_folder: Input video directory.
        output_folder: Output directory to save the frames.
        IMG_SIZE: Destination Image Size (tuple of size 2)
        label_counter: Starting label counter
        video_count: Number of videos to process.
        frames_per_sec: Number of frames to process per second. 
        frame_interval: Interval between the frames to be processed.
        dest_filetype: Processed image filetype (png, jpg) --> Default: png
    Returns:
        label_counter (after processing)
    """

    dest_filetype.replace('.','')
    processed_videos = 0
    vid_count = 0 # to check if < video_count

    if exists(input_folder) is False:
        raise ValueError('Input folder does not exist', input_folder)
    
    if IMG_SIZE is not None:
        _ = _check_target_size(IMG_SIZE)

    if label_counter is None:
        label_counter = 0

    if video_count is None:
        video_count = len(listdir(input_folder))

    if not exists(output_folder):
        os.mkdir(output_folder)

    # Begin Timer
    start = time.time()

    for root, _, files in os.walk(input_folder):
        for file in files:
            if vid_count < video_count:
                if file.endswith(_acceptable_video_formats): # if a video file
                    vid_filepath = root + os.sep + file
                    capture = cv.VideoCapture(vid_filepath)
                    video_frame_counter = 0

                    # Find the number of frames and FPS
                    video_frame_count = int(capture.get(cv.CAP_PROP_FRAME_COUNT)) - 1
                    video_fps = math.ceil(capture.get(cv.CAP_PROP_FPS))

                    print(f'{vid_count+1}. Reading \'{file}\'. Number of frames: {video_frame_count}. FPS: {video_fps}')

                    if frames_per_sec is not None and frame_interval is None:
                        interval= _determine_interval(video_fps/frames_per_sec) # eg: 30//15
                        print('Interval: ', interval)
                    
                    elif frame_interval is not None:
                        interval = frame_interval

                    # if frames_per_sec and frame_interval are both None, we assume that each frame should be processed
                    else:
                        interval = 1
                    
                    # Start converting the video
                    while capture.isOpened():
                        _, frame = capture.read()

                        if IMG_SIZE is not None:                    
                            frame = cv.resize(frame, IMG_SIZE)
                        
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
    # End timer
    end = time.time()
    
    # Printing stats
    taken = end-start
    print('[INFO] {} videos extracted in {:.2f} seconds'.format(processed_videos, taken ))

    return label_counter


def _determine_interval(x):
    y = '{:.1f}'.format(x)
    inde = y.find('.') + 1
    if inde == -1: # if no '.' (if an integer)
        return x
    if int(y[inde]) < 5:
        return math.floor(x)
    else:
        return math.ceil(x)