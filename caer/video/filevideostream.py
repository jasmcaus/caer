# Copyright 2020 The Caer Authors. All Rights Reserved.
#
# Licensed under the MIT License (see LICENSE);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at <https://opensource.org/licenses/MIT>
#
# ==============================================================================

from threading import Thread
import time
import math
from queue import Queue
import cv2 as cv

from ..opencv import get_opencv_version
from ..configs import FRAME_COUNT, FPS


#pylint:disable=no-member

# ret, jpeg = cv2.imencode('.jpg', image)
# return jpeg.tobytes()

# This class can handle both live as well as pre-existing videos. 
class FileVideoStream:
    def __init__(self, source = 0, queue_size=128):
        """
            Source must either be an integer (0,1,2 etc) or a path to a video file
        """
        self.live_video = False
        
        if isinstance(source, int):
            self.live_video = True
            # raise ValueError('Expected a filepath. Got an integer. FileVideoStream is not for live feed. Use LiveVideoStream instead')

        if not isinstance(source, (int,str)):
            raise ValueError(f'Expected either an integer or filepath. Got {type(source)}')
        
		# initializing the video stream
        self.video_stream = cv.VideoCapture(source)
        self.kill_stream = False
        
        # initialize the queue to store frames 
        self.Q = Queue(maxsize=queue_size)
        # intialize thread
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True

    def begin_stream(self):
        # start a thread to read frames from the video stream
        self.thread.start()
        return self

    def update(self):
        while True:
            if self.kill_stream:
                break

            # otherwise, ensure the queue has room in it
            if not self.Q.full():
                # read the next frame from the file
                ret, frame = self.video_stream.read()

                # If at the end of the video stream
                if not ret:
                    self.kill_stream = True

                # add the frame to the queue
                self.Q.put(frame)
            else:
                time.sleep(0.1)  # Rest for 10ms if we have a full queue

        self.video_stream.release()

    def read(self):
        # return next frame in the queue
        return self.Q.get()

    def __running(self):
        return self.__more() or not self.kill_stream

    def __more(self):
        # returns True if there are still frames in the queue
        tries = 0
        while self.Q.qsize() == 0 and not self.kill_stream and tries < 5:
            time.sleep(0.1)
            tries += 1

        return self.Q.qsize() > 0

    def release(self):
        self.kill_stream = True
        # wait until stream resources are released
        self.thread.join()
    
    # Gets frame count
    def count_frames(self):
        if not self.kill_stream and not self.live_video:
            if get_opencv_version() == '2':
                return int(self.stream.get(cv.FRAME_COUNT))
            else:
                return int(self.stream.get(cv.CAP_PROP_FRAME_COUNT))

        if self.live_video:
            print('[WARNING] Frames cannot be computed on live streams')
            return -1

    # Gets FPS count
    def get_fps(self):
        if not self.kill_stream:
            if get_opencv_version() == '2':
                return math.ceil(self.stream.get(cv.cv.CAP_PROP_FPS))
            else:
                return math.ceil(self.stream.get(cv.CAP_PROP_FPS))


__all__ = [
    'FileVideoStream'
]