# Copyright 2020 The Caer Authors. All Rights Reserved.
#
# Licensed under the MIT License (see LICENSE);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at <https://opensource.org/licenses/MIT>
#
# ==============================================================================


from .filevideostream import FileVideoStream


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


__all__ = [
    'VideoStream'
]