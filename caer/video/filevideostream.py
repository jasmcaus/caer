# Author: Jason Dsouza
# Github: http://www.github.com/jasmcaus

# Importing the necessary packages
from threading import Thread
import sys
import cv2 as cv
import time
from queue import Queue

class FileVideoStream:
    def __init__(self, source = 0, queue_size=128):
		# initialize the file video stream along with the boolean
		# used to indicate if the thread should be stopped or not
        self.video_stream = c
		self.stream = cv.VideoCapture(path)
		self.kill_stream = False
		self.transform = transform

		# initialize the queue used to store frames read from
		# the video file
		self.Q = Queue(maxsize=queue_size)
		# intialize thread
		self.thread = Thread(target=self.update, args=())
		self.thread.daemon = True

	def start(self):
		# start a thread to read frames from the file video stream
		self.thread.start()
		return self

	def update(self):
		# keep looping infinitely
		while True:
			# if the thread indicator variable is set, stop the
			# thread
			if self.kill_stream:
				break

			# otherwise, ensure the queue has room in it
			if not self.Q.full():
				# read the next frame from the file
				(grabbed, frame) = self.stream.read()

				# if the `grabbed` boolean is `False`, then we have
				# reached the end of the video file
				if not grabbed:
					self.kill_stream = True

				# add the frame to the queue
				self.Q.put(frame)
			else:
				time.sleep(0.1)  # Rest for 10ms, we have a full queue

		self.stream.release()

	def read(self):
		# return next frame in the queue
		return self.Q.get()

	# Insufficient to have consumer use while(more()) which does
	# not take into account if the producer has reached end of
	# file stream.
	def running(self):
		return self.more() or not self.kill_stream

	def more(self):
		# return True if there are still frames in the queue. If stream is not stopped, try to wait a moment
		tries = 0
		while self.Q.qsize() == 0 and not self.kill_stream and tries < 5:
			time.sleep(0.1)
			tries += 1

		return self.Q.qsize() > 0

	def stop(self):
		# indicate that the thread should be stopped
		self.kill_stream = True
		# wait until stream resources are released (producer thread might be still grabbing frame)
		self.thread.join()