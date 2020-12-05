#    _____           ______  _____ 
#  / ____/    /\    |  ____ |  __ \
# | |        /  \   | |__   | |__) | Caer - Modern Computer Vision
# | |       / /\ \  |  __|  |  _  /  Languages: Python, C, C++
# | |___   / ____ \ | |____ | | \ \  http://github.com/jasmcaus/caer
#  \_____\/_/    \_ \______ |_|  \_\

# Licensed under the MIT License <http://opensource.org/licenses/MIT>
# SPDX-License-Identifier: MIT
# Copyright (c) 2020 The Caer Authors <http://github.com/jasmcaus>


import math 

from .filestream import FileStream
from .constants import (
    FPS, FRAME_COUNT, FRAME_HEIGHT, FRAME_WIDTH
)


__all__ = [
    'Stream'
]

# Using the FileStream class as it can handle both live as well as pre-existing videos

class Stream():
    """
    Stream() supports a diverse range of video streams which can handle/control video stream almost any IP/USB Cameras, multimedia video file format (tested upto 4k), any network stream URL such as *http(s), rtp, rstp, rtmp, mms, etc.*

    It provides a flexible, high-level multi-threaded wrapper around OpenCV's VideoCapture() for threaded, error-free and synchronized frame handling.
    """

    def __init__(self, source=0):
        # Initializing the stream from DefaultVideoStream
        self._video_stream = FileStream(source=source)

        self.width = int(self._video_stream.get(FRAME_WIDTH))
        self.height = int(self._video_stream.get(FRAME_HEIGHT))
        self.res = (self.width, self.height)

        self.fps = math.ceil(self._video_stream.get(FPS))
        self.frames = int(self._video_stream.get(FRAME_COUNT))

    def start(self):
        # Begins the threaded video stream
        return self.video_stream.begin_stream()
    
    def update(self):
        self.video_stream.update()
      
    def read(self):
        """
        Extracts frames synchronously from monitored deque, while maintaining a fixed-length frame buffer in the memory, and blocks the thread if the deque is full.

        **Returns:** A n-dimensional numpy array.
        """
        
        return self.video_stream.read()
    
    def count_frames(self):
        return self.video_stream.count_frames()
        
    def release(self):
        """
        Safely terminates the thread, and release the VideoStream resources.
        """
        self.video_stream.release()
        

     # Get FPS
    def get_fps(self):
        return self.video_stream.get_fps()

    # Get frame dimensions
    def get_res(self):
        return self.video_stream.get_res()


VideoStream = Stream()