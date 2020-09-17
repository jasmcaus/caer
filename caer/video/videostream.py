# Author: Jason Dsouza
# Github: http://www.github.com/jasmcaus

# Importing the necessary packages
from .default_videostream import DefaultVideoStream

class VideoStream():
    def __init__(self, source=0):
        # Initializing the stream from DefaultVideoStream
        self.video_stream = DefaultVideoStream(source=source)

    def start(self):
        # Begins the threaded video stream
        return self.video_stream.start()
    
    def update_frame(self):
        self.video_stream.update_frame()
      
    def read(self):
        # Returns the current frame
        return self.video_stream.read()
    
    def count_frames(self):
        return self.video_stream.count_frames()
    
    def get_fps(self):
        return self.video_stream.get_fps()
        
    def release(self):
        self.video_stream.release()