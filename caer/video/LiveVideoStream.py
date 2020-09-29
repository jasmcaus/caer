# Author: Jason Dsouza
# Github: http://www.github.com/jasmcaus

#pylint:disable=no-member,pointless-string-statement

# Importing the necessary packages
from threading import Thread
import math
import cv2 as cv
from ..utils import get_opencv_version

"""
    Python threading has a specific meaning for daemon. A daemon thread will shut down immediately when the program exits. One way to think about these definitions is to consider the daemon thread a thread that runs in the background without worrying about shutting it down.

    If a program is running Threads that are not daemons, then the program will wait for those threads to complete before it terminates. Threads that are daemons, however, are just killed wherever they are when the program is exiting.
"""

# This class can only handle live video streams. When applied on pre-existing videos, there appears to # be an issue with Threading. As a result, the video plays through with a high (almost x4) speed
# This issue has been marked and will be fixed in a future update. 

class LiveVideoStream:
    def __init__(self, source=0):
        """
            Source must either be an integer (0,1,2 etc) or a path to a video file
        """
        if type(source) is str:
            raise ValueError('[ERROR] Expected an integer. Got a filepath. LiveVideoStream is for live streams only')
        
        # Initializing the video stream
        self.stream = cv.VideoCapture(source)
        self.ret, self.frame = self.stream.read()

        # Initializing the thread name
        self.thread_name = 'DefaultVideoStream'

        # Boolean to check whether stream should be killed
        self.kill_stream = False

    def begin_stream(self):
        # Starting the thread to read frames from the video stream
        thread = Thread(target=self.update, name=self.thread_name, args=())
        thread.daemon = True
        thread.start()
        return self

    def read(self):
        return self.frame
    
    def update(self):
        while not self.kill_stream:
            self.ret, self.frame = self.stream.read()
    
    def release(self):
        # Stops the stream
        # Releases video pointer
        self.kill_stream = True
    
    # Counting frames not applicable for live video
    
    # # Gets frame count
    # def count_frames(self):
    #     if not self.kill_stream:
    #         if get_opencv_version() == '2':
    #             return int(self.stream.get(cv.cv.CAP_PROP_FRAME_COUNT))
    #         else:
    #             return int(self.stream.get(cv.CAP_PROP_FRAME_COUNT))

    # Gets FPS count
    def get_fps(self):
        if not self.kill_stream:
            if get_opencv_version() == '2':
                return math.ceil(self.stream.get(cv.cv.CAP_PROP_FPS))
            else:
                return math.ceil(self.stream.get(cv.CAP_PROP_FPS))