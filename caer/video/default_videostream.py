# Author: Jason Dsouza
# Github: http://www.github.com/jasmcaus

#pylint:disable=pointless-string-statement

# Importing the necessary packages
import cv2 as cv
from threading import Thread
import os
from ..utils import get_opencv_version

class DefaultVideoStream:
    def __init__(self, source=0, thread_name='DefaultVideoStream'):
        """
            Source must either be an integer (0,1,2 etc) or a path to a video file
        """
        if type(source) is str:
            if os.path.isdir(source):
                raise ValueError('[ERROR] Expected path to a media file. Got path to a directory')
            if not os.path.exists(source):
                raise ValueError('[ERROR] The specified filepath does not exist')
        
        # Initializing the video stream
        self.stream = cv.VideoCapture(source)
        self.ret, self.frame = self.stream.read()

        # Initializing the thread name
        self.thread_name = thread_name

        # Boolean to check whether stream should be killed
        self.kill_stream = False

    def start_stream(self):
        """
            Python threading has a more specific meaning for daemon. A daemon thread will shut down immediately when the program exits. One way to think about these definitions is to consider the daemon thread a thread that runs in the background without worrying about shutting it down.

            If a program is running Threads that are not daemons, then the program will wait for those threads to complete before it terminates. Threads that are daemons, however, are just killed wherever they are when the program is exiting.
        """
        # Starting the thread to read frames from the video stream
        thread = Thread(target=self.update_frame, name=self.thread_name, args=())
        thread.daemon = True
        thread.start()
        return self

    def read_frame(self):
        return self.frame
    
    def update_frame(self):
        # while True:
        #     if self.kill_stream:
        #         return
        while not self.kill_stream:
            self.ret, self.frame = self.stream.read()
    
    def release(self):
        # Stops the stream
        self.kill_stream = True
    
    def count_frames(self):
        if not self.kill_stream:
            if get_opencv_version() == '2':
                return int(self.stream.get(cv.cv.CAP_PROP_FRAME_COUNT))
            else:
                return int(self.stream.get(cv.CAP_PROP_FRAME_COUNT))