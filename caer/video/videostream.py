#    _____           ______  _____ 
#  / ____/    /\    |  ____ |  __ \
# | |        /  \   | |__   | |__) | Caer - Modern Computer Vision
# | |       / /\ \  |  __|  |  _  /  Languages: Python, C, C++
# | |___   / ____ \ | |____ | | \ \  http://github.com/jasmcaus/caer
#  \_____\/_/    \_ \______ |_|__\_

# Licensed under the MIT License <http://opensource.org/licenses/MIT>
# SPDX-License-Identifier: MIT
# Copyright (c) 2020 The Caer Authors <http://github.com/jasmcaus>


from .filevideostream import FileVideoStream

__all__ = [
    'VideoStream'
]

# Using the FileVideoStream class as it can handle both live as well as pre-existing videos

class VideoStream():
    def __init__(self, source=0):
        # Initializing the stream from DefaultVideoStream
        self.video_stream = FileVideoStream(source=source)

    def begin_stream(self):
        # Begins the threaded video stream
        return self.video_stream.begin_stream()
    
    def update(self):
        self.video_stream.update()
      
    def read(self):
        # Returns the current frame
        return self.video_stream.read()
    
    def count_frames(self):
        return self.video_stream.count_frames()
    
    def get_fps(self):
        return self.video_stream.get_fps()
        
    def release(self):
        self.video_stream.release()
