#    _____           ______  _____ 
#  / ____/    /\    |  ____ |  __ \
# | |        /  \   | |__   | |__) | Caer - Modern Computer Vision
# | |       / /\ \  |  __|  |  _  /  Languages: Python, C, C++
# | |___   / ____ \ | |____ | | \ \  http://github.com/jasmcaus/caer
#  \_____\/_/    \_ \______ |_|  \_\

# Licensed under the MIT License <http://opensource.org/licenses/MIT>
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-21 The Caer Authors <http://github.com/jasmcaus>


import cv2 as cv 
import numpy as np
from collections import deque

from ..adorad import Tensor

__all__ = [
    'Stabilizer'
]

class Stabilizer:
    r"""
        This is an auxiliary class that enables Video Stabilization for caer with minimalistic latency, and at the expense
        of little to no additional computational requirements.
        
        The basic idea behind it is to tracks and save the salient feature array for the given number of frames and then uses these anchor point to cancel out all perturbations relative to it for the incoming frames in the queue. This class relies heavily on **Threaded Queue mode** for error-free & ultra-fast frame handling.
    """

    def __init__(
        self,
        smoothing_radius=25,
        border_type="black",
        border_size=0,
        crop_n_zoom=False,
    ):

        # initialize deques for handling input frames and its indexes
        self.__frame_queue = deque(maxlen=smoothing_radius)
        self.__frame_queue_indexes = deque(maxlen=smoothing_radius)

        # define and create Adaptive histogram equalization (AHE) object for optimizations
        self.__clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # initialize global vars
        self.__smoothing_radius = smoothing_radius  # averaging window, handles the quality of stabilization at expense of latency and sudden panning
        self.__smoothed_path = None  # handles the smoothed path with box filter
        self.__path = None  # handles path i.e cumulative sum of pevious_2_current transformations along a axis
        self.__transforms = []  # handles pevious_2_current transformations [dx,dy,da]
        self.__frame_transforms_smoothed = None  # handles smoothed array of pevious_2_current transformations w.r.t to frames
        self.frame_transform = None 
        self.__previous_gray = None  # handles previous gray frame
        self.__previous_keypoints = (
            None  # handles previous detect_GFTTed keypoints w.r.t previous gray frame
        )
        self.__frame_height, self.frame_width = (
            0,
            0,
        )  # handles width and height of input frames
        self.__crop_n_zoom = 0  # handles cropping and zooms frames to reduce the black borders from stabilization being too noticeable.

        # if check if crop_n_zoom defined
        if crop_n_zoom and border_size:
            self.__crop_n_zoom = border_size  # crops and zoom frame to original size
            self.__border_size = 0  # zero out border size
            self.__frame_size = None  # handles frame size for zooming

        else:
            # Add output borders to frame
            self.__border_size = border_size

        # define valid border modes
        border_modes = {
            "black": cv.BORDER_CONSTANT,
            "reflect": cv.BORDER_REFLECT,
            "reflect_101": cv.BORDER_REFLECT_101,
            "replicate": cv.BORDER_REPLICATE,
            "wrap": cv.BORDER_WRAP,
        }
        # choose valid border_mode from border_type
        if border_type in ["black", "reflect", "reflect_101", "replicate", "wrap"]:
            if not crop_n_zoom:
                # initialize global border mode variable
                self.__border_mode = border_modes[border_type]

            else:
                self.__border_mode = border_modes["black"]
        else:
            self.__border_mode = border_modes["black"]  # reset to default mode

        # define normalized box filter
        self.__box_filter = np.ones(smoothing_radius) / smoothing_radius

    def stabilize(self, frame):
        """
        This method takes an unstabilized video frame, and returns a stabilized one.
        Parameters:
            frame (Tensor): inputs unstabilized video frames.
        """
        # check if frame is None
        if frame is None:
            # return if it does
            return

        # save frame size for zooming
        if self.__crop_n_zoom and self.__frame_size == None:
            self.__frame_size = frame.shape[:2]

        # initiate transformations capturing
        if not self.__frame_queue:
            # for first frame
            previous_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)  # convert to gray
            previous_gray = self.__clahe.apply(previous_gray)  # optimize gray frame
            self.__previous_keypoints = cv.goodFeaturesToTrack(
                previous_gray,
                maxCorners=200,
                qualityLevel=0.05,
                minDistance=30.0,
                blockSize=3,
                mask=None,
                useHarrisDetector=False,
                k=0.04,
            )  # track features using GFTT
            self.__frame_height, self.frame_width = frame.shape[
                :2
            ]  # save input frame height and width
            self.__frame_queue.append(frame)  # save frame to deque
            self.__frame_queue_indexes.append(0)  # save frame index to deque
            self.__previous_gray = previous_gray[
                :
            ]  # save gray frame clone for further processing

        elif self.__frame_queue_indexes[-1] <= self.__smoothing_radius - 1:
            # for rest of frames
            self.__frame_queue.append(frame)  # save frame to deque
            self.__frame_queue_indexes.append(
                self.__frame_queue_indexes[-1] + 1
            )  # save frame index
            self.__generate_transformations()  # generate transformations
            if self.__frame_queue_indexes[-1] == self.__smoothing_radius - 1:
                # calculate smooth path once transformation capturing is completed
                for i in range(3):
                    # apply normalized box filter to the path
                    self.__smoothed_path[:, i] = self.__box_filter_convolve(
                        (self.__path[:, i]), window_size=self.__smoothing_radius
                    )
                # calculate deviation of path from smoothed path
                deviation = self.__smoothed_path - self.__path
                # save smoothed transformation
                self.__frame_transforms_smoothed = self.frame_transform + deviation
        else:
            # start applying transformations
            self.__frame_queue.append(frame)  # save frame to deque
            self.__frame_queue_indexes.append(
                self.__frame_queue_indexes[-1] + 1
            )  # save frame index
            self.__generate_transformations()  # generate transformations
            # calculate smooth path once transformation capturing is completed
            for i in range(3):
                # apply normalized box filter to the path
                self.__smoothed_path[:, i] = self.__box_filter_convolve(
                    (self.__path[:, i]), window_size=self.__smoothing_radius
                )
            # calculate deviation of path from smoothed path
            deviation = self.__smoothed_path - self.__path
            # save smoothed transformation
            self.__frame_transforms_smoothed = self.frame_transform + deviation
            # return transformation applied stabilized frame
            return self.__apply_transformations()

    def __generate_transformations(self):
        """
        An internal method that generate previous-to-current transformations [dx,dy,da].
        """
        frame_gray = cv.cvtColor(
            self.__frame_queue[-1], cv.COLOR_BGR2GRAY
        )  # retrieve current frame and convert to gray
        frame_gray = self.__clahe.apply(frame_gray)  # optimize it

        # calculate optical flow using Lucas-Kanade differential method
        curr_kps, status, _ = cv.calcOpticalFlowPyrLK(
            self.__previous_gray, frame_gray, self.__previous_keypoints, None
        )

        # select only valid key-points
        valid_curr_kps = curr_kps[status == 1]  # current
        valid_previous_keypoints = self.__previous_keypoints[status == 1]  # previous

        # calculate optimal affine transformation between pevious_2_current key-points
        transformation = cv.estimateAffinePartial2D(valid_previous_keypoints, valid_curr_kps)[0]

        # check if transformation is not None
        if not (transformation is None):
            # pevious_2_current translation in x direction
            dx = transformation[0, 2]
            # pevious_2_current translation in y direction
            dy = transformation[1, 2]
            # pevious_2_current rotation in angle
            da = np.arctan2(transformation[1, 0], transformation[0, 0])
        else:
            # otherwise zero it
            dx = dy = da = 0

        # save this transformation
        self.__transforms.append([dx, dy, da])

        # calculate path from cumulative transformations sum
        self.frame_transform = np.array(self.__transforms, dtype="float32")
        self.__path = np.cumsum(self.frame_transform, axis=0)
        # create smoothed path from a copy of path
        self.__smoothed_path = np.copy(self.__path)

        # re-calculate and save GFTT key-points for current gray frame
        self.__previous_keypoints = cv.goodFeaturesToTrack(
            frame_gray,
            maxCorners=200,
            qualityLevel=0.05,
            minDistance=30.0,
            blockSize=3,
            mask=None,
            useHarrisDetector=False,
            k=0.04,
        )
        # save this gray frame for further processing
        self.__previous_gray = frame_gray[:]


    def __box_filter_convolve(self, path, window_size):
        """
        An internal method that applies *normalized linear box filter* to path w.r.t averaging window
        Parameters:
        * path (Tensor): a cumulative sum of transformations
        * window_size (int): averaging window size
        """
        # pad path to size of averaging window
        path_padded = np.pad(path, (window_size, window_size), "median")
        # apply linear box filter to path
        path_smoothed = np.convolve(path_padded, self.__box_filter, mode="same")
        # crop the smoothed path to original path
        path_smoothed = path_smoothed[window_size:-window_size]
        # assert if cropping is completed
        assert path.shape == path_smoothed.shape
        # return smoothed path
        return path_smoothed


    def __apply_transformations(self):
        """
        An internal method that applies affine transformation to the given frame
        from previously calculated transformations
        """
        # extract frame and its index from deque
        queue_frame = self.__frame_queue.popleft()
        queue_frame_index = self.__frame_queue_indexes.popleft()

        # create border around extracted frame w.r.t border_size
        bordered_frame = cv.copyMakeBorder(
            queue_frame,
            top=self.__border_size,
            bottom=self.__border_size,
            left=self.__border_size,
            right=self.__border_size,
            borderType=self.__border_mode,
            value=[0, 0, 0],
        )
        alpha_bordered_frame = cv.cvtColor(
            bordered_frame, cv.COLOR_BGR2BGRA
        )  # create alpha channel
        # extract alpha channel
        alpha_bordered_frame[:, :, 3] = 0
        alpha_bordered_frame[
            self.__border_size : self.__border_size + self.__frame_height,
            self.__border_size : self.__border_size + self.frame_width,
            3,
        ] = 255

        # extracting Transformations w.r.t frame index
        dx = self.__frame_transforms_smoothed[queue_frame_index, 0]  # x-axis
        dy = self.__frame_transforms_smoothed[queue_frame_index, 1]  # y-axis
        da = self.__frame_transforms_smoothed[queue_frame_index, 2]  # angle

        # building 2x3 transformation matrix from extracted transformations
        queue_frame_transform = np.zeros((2, 3), np.float32)
        queue_frame_transform[0, 0] = np.cos(da)
        queue_frame_transform[0, 1] = -np.sin(da)
        queue_frame_transform[1, 0] = np.sin(da)
        queue_frame_transform[1, 1] = np.cos(da)
        queue_frame_transform[0, 2] = dx
        queue_frame_transform[1, 2] = dy

        # Applying an affine transformation to the frame
        frame_wrapped = cv.warpAffine(
            alpha_bordered_frame,
            queue_frame_transform,
            alpha_bordered_frame.shape[:2][::-1],
            borderMode=self.__border_mode,
        )

        # drop alpha channel
        frame_stabilized = frame_wrapped[:, :, :3]

        # crop and zoom
        if self.__crop_n_zoom:
            # crop stabilized frame
            frame_cropped = frame_stabilized[
                self.__crop_n_zoom : -self.__crop_n_zoom,
                self.__crop_n_zoom : -self.__crop_n_zoom,
            ]
            # zoom stabilized frame
            frame_stabilized = cv.resize(frame_cropped, self.__frame_size[::-1])

        # finally return stabilized frame
        return frame_stabilized


    def clean(self):
        """
        Cleans Stabilizer resources
        """
        # check if deque present
        if self.__frame_queue:
            # clear frame deque
            self.__frame_queue.clear()
            # clear frame indexes deque
            self.__frame_queue_indexes.clear()