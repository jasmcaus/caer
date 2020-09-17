# Author: Jason Dsouza
# Github: http://www.github.com/jasmcaus

# Importing the necessary packages
from .default_videostream import DefaultVideoStream

class VideoStream():
    def __init__(self, source=0):
        # Initializing the stream from DefaultVideoStream
        self.video_stream = DefaultVideoStream(source=source)

    def start_stream(self):
        # Begins the threaded video stream
        return self.video_stream.start_stream()
    
    def update_frame(self):
        self.video_stream.update_frame()
    
    def read_frame(self):
        # Returns the current frame
        self.video_stream.read_frame()
    
    def stop_stream(self):
        self.video_stream.stop_stream()