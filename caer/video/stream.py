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
import numpy as np 

from .constants import (
    FPS, FRAME_COUNT, FRAME_HEIGHT, FRAME_WIDTH
)
from ..jit.annotations import Tuple
from ..adorad import Tensor

__all__ = [
    'Stream'
]


#pylint:disable=no-member

# ret, jpeg = cv2.imencode('.jpg', image)
# return jpeg.tobytes()

# This class can handle both live as well as pre-existing videos. 
class Stream:
    r"""
        This is an auxiliary class that enables Video Streaming for ``caer`` with minimalistic latency, and at the expense
        of little to no additional computational requirements.
        
        The basic idea behind it is to tracks and save the salient feature array for the given number of frames and then uses these anchor point to cancel out all perturbations relative to it for the incoming frames in the queue. This class relies heavily on **Threaded Queue mode** for error-free & ultra-fast frame handling.

    Args:
        source (int, str): Source path for the video. Uses an external camera device if ``source`` is an integer.
        qsize (int): Default queue size for handling the video streams. Default: 128.
    """

    def __init__(self, source = 0, qsize=128) -> None: # TODO: Add colorspace support
        r"""
            Source must either be an integer (0,1,2 etc) or a path to a video file
        """
        self.live_video = False
        
        if isinstance(source, int):
            self.live_video = True
            # raise ValueError('Expected a filepath. Got an integer. FileVideoStream is not for live feed. Use LiveVideoStream instead')

        if not isinstance(source, (int,str)):
            raise ValueError(f'Expected either an integer or filepath. Got {type(source)}')
        
        # initializing the video stream
        self._video_stream = cv.VideoCapture(source)
        self.kill_stream = False

        self.width = int(self._video_stream.get(FRAME_WIDTH))
        self.height = int(self._video_stream.get(FRAME_HEIGHT))
        self.res = (self.width, self.height)

        self.fps = math.ceil(self._video_stream.get(FPS))
        self.frames = int(self._video_stream.get(FRAME_COUNT))
        
        # initialize the queue to store frames 
        self._Q = Queue(maxsize=qsize)

        # intialize thread
        self._thread = None


    def start(self) -> None:
        # start a thread to read frames from the video stream
        self._thread = Thread(target=self._update, name="caer.video.Stream()", args=())
        self._thread.daemon = True
        self._thread.start()
        return self


    def _update(self) -> Tensor:
        while True:
            if self.kill_stream:
                break

            # otherwise, ensure the queue has room in it
            if not self._Q.full():
                # read the next frame from the file
                ret, frame = self._video_stream.read()

                # If at the end of the video stream
                if not ret:
                    self.release()
                    return 

                # add the frame to the queue
                self._Q.put(frame)
            else:
                time.sleep(0.1)  # Rest for 10ms if we have a full queue

        self._video_stream.release()


    def read(self) -> Tensor:
        """
        Extracts frames synchronously from monitored deque, while maintaining a fixed-length frame buffer in the memory, and blocks the thread if the deque is full.

        **Returns:** A n-dimensional numpy array.
        """

        return self._Q.get()


    def release(self) -> None:
        """
        Safely terminates the thread, and release the Stream resources.
        """
        self.kill_stream = True
        # wait until stream resources are released
        if self._thread is not None:
            self._thread.join()
            self._thread = None 
        self.frames = 0 
        self.fps = 0
    

    # Gets frame count
    def count_frames(self) -> int:
        """
            Returns the number of frames for the current video
            Optional: use the `frames` attribute
        """
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
    def get_fps(self) -> (float, int):
        """
            Returns the fps (frames per second) value for the current video
            Optional: use the `fps` attribute
        """
        if not self.kill_stream:
            return self.fps
            # if get_opencv_version() == '2':
            #     return math.ceil(self.stream.get(FPS_DEPR))
            # else:
            #     return math.ceil(self.stream.get(FPS))
    
    
    # Get frame dimensions
    def get_res(self) -> Tuple[int]:
        return self.res



###########################################################################################
# Old implementation of stream.py when it used to inherit the FileStream() from filestream.py
# Until 5 Dec 2020, this was the implementation
###########################################################################################

# from .filestream import FileStream


# __all__ = [
#     'Stream'
# ]

# # Using the FileStream class as it can handle both live as well as pre-existing videos
# class Stream(FileStream):
#     """
#     Stream() supports a diverse range of video streams which can handle/control video stream almost any IP/USB Cameras, multimedia video file format (tested upto 4k), any network stream URL such as *http(s), rtp, rstp, rtmp, mms, etc.*

#     It provides a flexible, high-level multi-threaded wrapper around OpenCV's VideoCapture() for threaded, error-free and synchronized frame handling.
#     """

#     def __init__(self, source=0):
#         # Initializing the stream from DefaultVideoStream
#         print('Beg of f')
#         self._video_stream = FileStream(source=source)
#         print('End of f')

#         self.width = self._video_stream.width
#         self.height = self._video_stream.height
#         self.res = (self.width, self.height)

#         self.fps = self._video_stream.fps
#         self.frames = self._video_stream.frames

#     def start(self):
#         # Begins the threaded video stream
#         return self._video_stream.start()
    
#     def update(self):
#         self._video_stream.update()
      
#     def read(self):
#         """
#         Extracts frames synchronously from monitored deque, while maintaining a fixed-length frame buffer in the memory, and blocks the thread if the deque is full.

#         **Returns:** A n-dimensional numpy array.
#         """
        
#         return self._video_stream.read()
    
#     def count_frames(self):
#         return self._video_stream.count_frames()
        
#     def release(self):
#         """
#         Safely terminates the thread, and release the VideoStream resources.
#         """
#         self._video_stream.release()
        

#      # Get FPS
#     def get_fps(self):
#         return self._video_stream.get_fps()

#     # Get frame dimensions
#     def get_res(self):
#         return self._video_stream.get_res()
