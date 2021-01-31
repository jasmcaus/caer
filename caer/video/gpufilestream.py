#    _____           ______  _____ 
#  / ____/    /\    |  ____ |  __ \
# | |        /  \   | |__   | |__) | Caer - Modern Computer Vision
# | |       / /\ \  |  __|  |  _  /  Languages: Python, C, C++, Cuda
# | |___   / ____ \ | |____ | | \ \  http://github.com/jasmcaus/caer
#  \_____\/_/    \_ \______ |_|  \_\

# Licensed under the MIT License <http://opensource.org/licenses/MIT>
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021 The Caer Authors <http://github.com/jasmcaus>


from threading import Thread
import time
import math
from queue import Queue
import cv2 as cv

from .constants import FRAME_COUNT, FPS

__all__ = [
    'GPUFileStream'
]


class GPUFileStream:
    r"""
        This is an auxiliary class that enables Video Streaming using the GPU for caer with minimalistic latency, and at the expense of little to no additional computational requirements.
        
        The basic idea behind it is to tracks and save the salient feature array for the given number of frames and then uses these anchor point to cancel out all perturbations relative to it for the incoming frames in the queue. This class relies heavily on **Threaded Queue mode** for error-free & ultra-fast frame handling.

    Args:
        source (int, str): Source path for the video. If ``source=0``, the default camera device is used. For 
            multiple external camera devices, use incremented values. For eg: ``source=1`` represents the second camera device on your system.
        qsize (int): Default queue size for handling the video streams. Default: 128.
    """

    def __init__(self, source, qsize=128):
        """
            Source must be a path to a video file
            Utilizes your system's GPU to process the stream
        """

        if not isinstance(source, str):
            raise ValueError(f'Expected either a filepath. Got {type(source)}. Consider using VideoStream which supports both live video as well as pre-existing videos')

        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        self.stream = cv.VideoCapture(source)
        self.kill_stream = False
        self.count = 0

        # initialize the queue to store frames
        self.Q = Queue(maxsize=qsize)

        self.width = int(self.stream.get(cv.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.stream.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.res = (self.width, self.height)

        self.fps = math.ceil(self.stream.get(FPS))
        self.frames = int(self.stream.get(FRAME_COUNT))
        
        # since we use UMat to store the images to
        # we need to initialize them beforehand
        self.qframes = [0] * qsize
        for ii in range(qsize):
            self.qframes[ii] = cv.UMat(self.height, self.width, cv.CV_8UC3)


    def begin_stream(self):
        # start a thread to read frames from the file video stream
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self


    def update(self):
        # keep looping infinitely
        while True:
            if self.kill_stream:
                return

            # otherwise, ensure the queue has room in it
            if not self.Q.full():
                self.count += 1
                target = (self.count-1) % self.Q.maxsize
                ret = self.stream.grab()

                if not ret:
                    self.release()
                    return 

                self.stream.retrieve(self.qframes[target])

                # add the frame to the queue
                self.Q.put(target)


    def read(self):
        while (not self.more() and self.kill_stream):
            time.sleep(0.1)
        # return next frame in the queue
        return self.qframes[self.Q.get()]


    def more(self):
        # return True if there are still frames in the queue
        return self.Q.qsize() > 0


    def release(self):
        self.kill_stream = True
        # wait until stream resources are released
        self.thread.join()


    # Gets frame count
    def count_frames(self):
        if not self.kill_stream and not self.live_video:
            return self.frames
            # if get_opencv_version() == '2':
            #     return int(self.stream.get(FRAME_COUNT_DEPR))
            # else:
            #     return int(self.stream.get(FRAME_COUNT))
            

        if self.live_video:
            print('[WARNING] Frames cannot be computed on live streams')
            return -1


    # Gets FPS count
    def get_fps(self):
        if not self.kill_stream:
            return self.fps

    # Get frame dimensions
    def get_res(self):
        return self.res