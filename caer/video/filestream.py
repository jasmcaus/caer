"""

Note: 
The very same implementation can be found in stream.py. Until 5 Dec 2020, this was separate, but due to several reasons (such as video lag in the threads, this is now merged into stream.py)
"""

# #    _____           ______  _____ 
# #  / ____/    /\    |  ____ |  __ \
# # | |        /  \   | |__   | |__) | Caer - Modern Computer Vision
# # | |       / /\ \  |  __|  |  _  /  Languages: Python, C, C++
# # | |___   / ____ \ | |____ | | \ \  http://github.com/jasmcaus/caer
# #  \_____\/_/    \_ \______ |_|  \_\

# # Licensed under the MIT License <http://opensource.org/licenses/MIT>
# # SPDX-License-Identifier: MIT
# # Copyright (c) 2020-21 The Caer Authors <http://github.com/jasmcaus>


# from threading import Thread
# import time
# import math
# from queue import Queue
# import cv2 as cv
# import numpy as np 

# from ..adorad import Tensor
# from .constants import (
#     FPS, FRAME_COUNT, FRAME_HEIGHT, FRAME_WIDTH
# )
# from ..jit.annotations import Tuple

# __all__ = [
#     'FileStream'
# ]


# #pylint:disable=no-member

# # ret, jpeg = cv2.imencode('.jpg', image)
# # return jpeg.tobytes()

# # This class can handle both live as well as pre-existing videos. 
# class FileStream:
#     def __init__(self, source = 0, queue_size=128) -> None: # TODO: Add colorspace support
#         """
#             Source must either be an integer (0,1,2 etc) or a path to a video file
#         """
#         self.live_video = False
        
#         if isinstance(source, int):
#             self.live_video = True
#             # raise ValueError('Expected a filepath. Got an integer. FileVideoStream is not for live feed. Use LiveVideoStream instead')

#         if not isinstance(source, (int,str)):
#             raise ValueError(f'Expected either an integer or filepath. Got {type(source)}')
        
# 		# initializing the video stream
#         print('Vid str begin')
#         self._video_stream = cv.VideoCapture(source)
#         print('Vid sr end')
#         self.kill_stream = False

#         self.width = int(self._video_stream.get(FRAME_WIDTH))
#         self.height = int(self._video_stream.get(FRAME_HEIGHT))
#         self.res = (self.width, self.height)

#         self.fps = math.ceil(self._video_stream.get(FPS))
#         self.frames = int(self._video_stream.get(FRAME_COUNT))
        
#         # initialize the queue to store frames 
#         self._Q = Queue(maxsize=queue_size)

#         # intialize thread
#         self._thread = None


#     def begin_stream(self) -> None:
#         # start a thread to read frames from the video stream
#         self._thread = Thread(target=self._update, name="caer.video.Stream()", args=())
#         self._thread.daemon = True
#         self._thread.start()
#         return self


#     def _update(self) -> Tensor:
#         while True:
#             if self.kill_stream:
#                 break

#             # otherwise, ensure the queue has room in it
#             if not self._Q.full():
#                 # read the next frame from the file
#                 ret, frame = self._video_stream.read()

#                 # If at the end of the video stream
#                 if not ret:
#                     self.release()
#                     return 

#                 # add the frame to the queue
#                 self._Q.put(frame)
#             else:
#                 time.sleep(0.1)  # Rest for 10ms if we have a full queue

#         self._video_stream.release()


#     def read(self) -> Tensor:
#         """
#         Extracts frames synchronously from monitored deque, while maintaining a fixed-length frame buffer in the memory, and blocks the thread if the deque is full.

#         **Returns:** A n-dimensional numpy array.
#         """

#         return self._Q.get()


#     def release(self) -> None:
#         """
#         Safely terminates the thread, and release the Stream resources.
#         """
#         self.kill_stream = True
#         # wait until stream resources are released
#         if self._thread is not None:
#             self._thread.join()
#             self._thread = None 
#         self.frames = 0 
#         self.fps = 0
    

#     # Gets frame count
#     def count_frames(self) -> int:
#         """
#             Returns the number of frames for the current video
#             Optional: use the `frames` attribute
#         """
#         if not self.kill_stream and not self.live_video:
#             return self.frames
#             # if get_opencv_version() == '2':
#             #     return int(self.stream.get(FRAME_COUNT_DEPR))
#             # else:
#             #     return int(self.stream.get(FRAME_COUNT))
            

#         if self.live_video:
#             print('[WARNING] Frames cannot be computed on live streams')
#             return -1


#     # Gets FPS count
#     def get_fps(self) -> (float, int):
#         """
#             Returns the fps (frames per second) value for the current video
#             Optional: use the `fps` attribute
#         """
#         if not self.kill_stream:
#             return self.fps
#             # if get_opencv_version() == '2':
#             #     return math.ceil(self.stream.get(FPS_DEPR))
#             # else:
#             #     return math.ceil(self.stream.get(FPS))
    
    
#     # Get frame dimensions
#     def get_res(self) -> Tuple[int]:
#         return self.res