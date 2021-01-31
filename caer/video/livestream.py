#    _____           ______  _____ 
#  / ____/    /\    |  ____ |  __ \
# | |        /  \   | |__   | |__) | Caer - Modern Computer Vision
# | |       / /\ \  |  __|  |  _  /  Languages: Python, C, C++, Cuda
# | |___   / ____ \ | |____ | | \ \  http://github.com/jasmcaus/caer
#  \_____\/_/    \_ \______ |_|  \_\

# Licensed under the MIT License <http://opensource.org/licenses/MIT>
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021 The Caer Authors <http://github.com/jasmcaus>


#pylint:disable=no-member,pointless-string-statement

from threading import Thread
import math
import cv2 as cv

from .constants import FPS

__all__ = [
    'LiveStream'
]

"""
    Python threading has a specific meaning for daemon. A daemon thread will shut down immediately when the program exits. One way to think about these definitions is to consider the daemon thread a thread that runs in the background without worrying about shutting it down.

    If a program is running Threads that are not daemons, then the program will wait for those threads to complete before it terminates. Threads that are daemons, however, are just killed wherever they are when the program is exiting.
"""

# This class can only handle live video streams. When applied on pre-existing videos, there appears to # be an issue with Threading. As a result, the video plays through with a high (almost x4) speed
# This issue has been marked and will be fixed in a future update. 

class LiveStream:
    r"""
        This is an auxiliary class that enables Live Video Streaming for caer with minimalistic latency, and at the expense
        of little to no additional computational requirements.
        
        The basic idea behind it is to tracks and save the salient feature array for the given number of frames and then uses these anchor point to cancel out all perturbations relative to it for the incoming frames in the queue. This class relies heavily on **Threaded Queue mode** for error-free & ultra-fast frame handling.

    Args:
        source (int): Source path for the video. If ``source=0``, the default camera device is used. For 
            multiple external camera devices, use incremented values. For eg: ``source=1`` represents the second camera device on your system.
    """

    def __init__(self, source=0):
        r"""
            Source must either be an integer (0, 1, 2 etc) or a path to a video file
        """
        if isinstance(source, str):
            raise ValueError('Expected an integer. Got a filepath. LiveVideoStream is for live streams only')
        
        # Initializing the video stream
        self.stream = cv.VideoCapture(source)
        print('Live yes')
        self.ret, self.frame = self.stream.read()
        print('Live not')

        self.width = int(self.stream.get(cv.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.stream.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.res = (self.width, self.height)

        self.fps = math.ceil(self.stream.get(FPS))

        # Initializing the thread name
        self.thread_name = 'DefaultVideoStream'

        # Boolean to check whether stream should be killed
        self.kill_stream = False

    def begin_stream(self):
        # Starting the thread to read frames from the video stream
        thread = Thread(target=self.update, name=self.thread_name, args=())
        thread.daemon = True
        thread.start()
        return self

    def read(self):
        return self.frame
    
    def update(self):
        while not self.kill_stream:
            self.ret, self.frame = self.stream.read()
    
    def release(self):
        # Stops the stream
        # Releases video pointer
        self.kill_stream = True
    
    # Counting frames not applicable for live video
    
    # # Gets frame count
    # def count_frames(self):
    #     if not self.kill_stream:
    #         if get_opencv_version() == '2':
    #             return int(self.stream.get(FRAME_COUNT_DEPR))
    #         else:
    #             return int(self.stream.get(FRAME_COUNT))

    # Gets FPS count
    def get_fps(self):
        if not self.kill_stream:
            return self.fps
            # if get_opencv_version() == '2':
            #     return math.ceil(self.stream.get(FPS_DEPR))
            # else:
            #     return math.ceil(self.stream.get(FPS))

    
    # Get frame dimensions
    def get_res(self):
        return self.res