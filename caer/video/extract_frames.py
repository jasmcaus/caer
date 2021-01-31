#    _____           ______  _____ 
#  / ____/    /\    |  ____ |  __ \
# | |        /  \   | |__   | |__) | Caer - Modern Computer Vision
# | |       / /\ \  |  __|  |  _  /  Languages: Python, C, C++, Cuda
# | |___   / ____ \ | |____ | | \ \  http://github.com/jasmcaus/caer
#  \_____\/_/    \_ \______ |_|  \_\

# Licensed under the MIT License <http://opensource.org/licenses/MIT>
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021 The Caer Authors <http://github.com/jasmcaus>


import math
import time
import cv2 as cv

from .._internal import _check_target_size
from ..path import list_videos, exists, mkdir
from .constants import FRAME_COUNT, FPS
from ..io import imsave, resize

__all__ = [
    'extract_frames'
]

def extract_frames(input_folder, 
                   output_folder, 
                   target_size=None, 
                   recursive=False,
                   label_counter = None, 
                   max_video_count=None, 
                   frames_per_sec=None, 
                   frame_interval=None,
                   dest_filetype='jpg') -> int:
    r"""
        Extract frames from videos within a directory and save them as separate frames in an output directory.

    Args:
        input_folder (str): Input video directory.
        output_folder (str): Output directory to save the frames.
        target_size (tuple): Destination Image Size (tuple of size 2)
        label_counter (int): Starting label counter (optional)
        max_video_count (int): Number of videos to process.
        frames_per_sec (int, float): Number of frames to process per second. 
        frame_interval (int, float): Interval between the frames to be processed.
        dest_filetype (str): Processed image filetype (png, jpg). Default: png

    Returns:
        label_counter (after processing)

    """

    dest_filetype.replace('.', '')
    processed_videos = 0
    vid_count = 0 # to check if < max_video_count

    if not exists(input_folder):
        raise ValueError('Input folder does not exist', input_folder)
    
    if target_size is not None:
        _ = _check_target_size(target_size)
    
    video_list = list_videos(input_folder, recursive=recursive, use_fullpath=True, verbose=0)

    if len(video_list) == 0:
        raise ValueError(f'No videos found at {input_folder}')

    if label_counter is None:
        label_counter = 0

    if max_video_count is None:
        max_video_count = len(video_list)

    if not exists(output_folder):
        mkdir(output_folder)

    # Begin Timer
    start = time.time()

    for vid_filepath in video_list:
        if vid_count < max_video_count:
            capture = cv.VideoCapture(vid_filepath)
            video_frame_counter = 0
            vid_count += 1

            # Find the number of frames and FPS
            video_frame_count = int(capture.get(FRAME_COUNT)) - 1
            video_fps = math.ceil(capture.get(FPS))
            file = vid_filepath[vid_filepath.rindex('/')+1:]
            
            if frames_per_sec is not None:
                if frame_interval is None:
                    interval = _determine_interval(video_fps/frames_per_sec) # eg: 30//15
            
                else:
                    interval = frame_interval

            # if frames_per_sec and frame_interval are both None, we assume that each frame should be processed
            else:
                interval = 1
            
            # processed_frames = (video_frame_count//video_fps) * frames_per_sec

            print(f'{vid_count}. Reading \'{file}\'. Frame Count: {video_frame_count}. FPS: {video_fps}. Processed frames: {video_frame_count//interval}')
            
            # Start converting the video
            while capture.isOpened():
                _, frame = capture.read()

                if target_size is not None:                    
                    frame = resize(frame, target_size=target_size)
                
                # Write the results back to output location as per specified frames per second
                if video_frame_counter % interval == 0:
                    imsave(f'{output_folder}/{file}_{label_counter}.{dest_filetype}', frame)
                    video_frame_counter += 1
                    label_counter += 1
                    # print('Frame counter: ', video_frame_counter)
                
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
    minu = taken // 60
    sec = taken % 60
    if processed_videos > 1:
        print(f'[INFO] {processed_videos} videos extracted in {minu:.0f}m {sec:.0f}s')
    else:
        print(f'[INFO] {processed_videos} video extracted in {minu:.0f}m {sec:.0f}s')

    return label_counter


def _determine_interval(x) -> int:
    y = '{x:.1f}'
    inde = y.find('.') + 1
    if inde == -1: # if no '.' (if an integer)
        return x
    if int(y[inde]) < 5:
        return math.floor(x)
    else:
        return math.ceil(x)
