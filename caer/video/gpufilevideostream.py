#
#  _____ _____ _____ _____
# |     |     | ___  | __|  Caer - Modern Computer Vision
# |     | ___ |      | \    Languages: Python, C, C++
# |_____|     | ____ |  \   http://github.com/jasmcaus/caer

# Licensed under the MIT License <http://opensource.org/licenses/MIT>
# SPDX-License-Identifier: MIT
# Copyright (c) 2020 The Caer Authors <http://github.com/jasmcaus>


from threading import Thread
import time
import math
from queue import Queue
import cv2 as cv

from ..globals import FRAME_COUNT, FPS


class GPUFileVideoStream:

    def __init__(self, source, queueSize=128):
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
        self.Q = Queue(maxsize=queueSize)

        self.width = int(self.stream.get(cv.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.stream.get(cv.CAP_PROP_FRAME_HEIGHT))
        
        # since we use UMat to store the images to
        # we need to initialize them beforehand
        self.frames = [0] * queueSize
        for ii in range(queueSize):
            self.frames[ii] = cv.UMat(self.height, self.width, cv.CV_8UC3)


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

                self.stream.retrieve(self.frames[target])

                # add the frame to the queue
                self.Q.put(target)


    def read(self):
        while (not self.more() and self.kill_stream):
            time.sleep(0.1)
        # return next frame in the queue
        return self.frames[self.Q.get()]


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
            return int(self.stream.get(FRAME_COUNT))
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
            return math.ceil(self.stream.get(FPS))

__all__ = [
    'GPUFileVideoStream'
]