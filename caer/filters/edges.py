# #    _____           ______  _____ 
# #  / ____/    /\    |  ____ |  __ \
# # | |        /  \   | |__   | |__) | Caer - Modern Computer Vision
# # | |       / /\ \  |  __|  |  _  /  Languages: Python, C, C++
# # | |___   / ____ \ | |____ | | \ \  http://github.com/jasmcaus/caer
# #  \_____\/_/    \_ \______ |_|  \_\

# # Licensed under the MIT License <http://opensource.org/licenses/MIT>
# # SPDX-License-Identifier: MIT
# # Copyright (c) 2020-21 The Caer Authors <http://github.com/jasmcaus>


# import numpy as np
# # import caer as mh

# from .convolve import gaussian_filter, convolve

# from ..opencv import bgr_to_gray
# from ..morph import regmax

# _hsobel_filter = np.array([
#     [-1, 0, 1],
#     [-2, 0, 2],
#     [-1, 0, 1]])/8.

# _vsobel_filter = np.array([
#     [-1, -2, -1],
#     [ 0,  0,  0],
#     [ 1,  2,  1]])/8.


# __all__ = [
#     'sobel',
#     'dog',
#     ]


# def sobel(tens, just_filter=False):
#     """
#     Compute edges using Sobel's algorithm.

#     `edges` is a binary image of edges computed according to Sobel's algorithm.
#     This implementation is tuned to match MATLAB's implementation.

#     Args:
#         tens (2D-Tensor) : 
#         just_filter (boolean, optional) : 
#             If true, then return the result of filtering the image with the sobel
#             filters, but do not threashold (default is False).

#     Returns:
#     edges (Tensor) : 
#         Binary image of edges, unless `just_filter`, in which case it will be
#         an array of floating point values.
#     """
#     # This is based on Octave's implementation,
#     # but with some reverse engineering to match Matlab exactly
#     tens = np.array(tens, dtype=np.float)
#     if tens.ndim > 2:
#         try:
#             tens = bgr_to_gray(tens)
#         except Exception:
#             raise ValueError('caer.filters.sobel() is only available for 2-dimensional images')

#     tens -= tens.min()
#     ptp = tens.ptp()

#     if ptp == 0:
#         return tens
#     tens /= ptp

#     # Using 'nearest' seems to be MATLAB's implementation
#     vfiltered = convolve(tens, _vsobel_filter, mode='nearest')
#     hfiltered = convolve(tens, _hsobel_filter, mode='nearest')
#     vfiltered **= 2
#     hfiltered **= 2
#     filtered = vfiltered
#     filtered += hfiltered

#     if just_filter:
#         return filtered

#     thresh = 2*np.sqrt(filtered.mean())

#     return regmax(filtered) * (np.sqrt(filtered) > thresh)


# def dog(tens, sigma1 = 2, multiplier = 1.001, just_filter = False):
#     """
#     Compute edges using the Difference of Gaussian (DoG) operator.
#     `edges` is a binary image of edges.

#     Args:
#         tens : Any 2D-Tensor
#         sigma1 : the sigma value of the first Gaussian filter. The second filter 
#             will have sigma value 1.001*sigma1
#         multiplier : the multiplier to get sigma2. sigma2 = sigma1 * multiplier
#         just_filter (boolean, optional) : 
#             If true, then return the result of filtering the image with the DoG
#             filters, no zero-crossing is detected (default is False).
      
#     Returns:
#         edges (Tensor) : 
#             Binary image of edges, unless `just_filter`, in which case it will be
#             an array of floating point values.
#     """
#     tens = np.array(tens, dtype=np.float)
#     if tens.ndim != 2:
#         raise ValueError('caer.dog: Only available for 2-dimensional images')

#     sigma2 = sigma1 * multiplier
    
#     G1 = gaussian_filter(tens, sigma1, mode = 'nearest')
#     G2 = gaussian_filter(tens, sigma2, mode = 'nearest')
#     DoG = G2 - G1
    
#     (m, n) = tens.shape
#     if not just_filter:
#         e = np.zeros((m, n), dtype=bool)
#     else:
#         return DoG
        
#     thresh = .75 * np.mean(abs(DoG))

    
#     # Look for the zero crossings:  +-, -+ and their transposes
#     # Choose the edge to be the negative point
#     rr = np.arange(1, m-2)
#     cc = np.arange(1, n-2)

#     (rx,cx) = np.nonzero(
#         np.logical_and(np.logical_and(DoG[np.ix_(rr,cc)] < 0, DoG[np.ix_(rr,cc+1)] > 0), 
#                        abs( DoG[np.ix_(rr,cc)] - DoG[np.ix_(rr,cc+1)]) > thresh) )   # [- +]
#     e[(rx,cx)] = 1
#     (rx,cx) = np.nonzero(
#         np.logical_and(np.logical_and(DoG[np.ix_(rr,cc-1)] > 0, DoG[np.ix_(rr,cc+1)] < 0), 
#                        abs( DoG[np.ix_(rr,cc-1)] - DoG[np.ix_(rr,cc)]) > thresh) )   # [+ -]
#     e[(rx,cx)] = 1
#     (rx,cx) = np.nonzero(
#         np.logical_and(np.logical_and(DoG[np.ix_(rr,cc)] < 0, DoG[np.ix_(rr+1,cc)] > 0), 
#                        abs( DoG[np.ix_(rr,cc)] - DoG[np.ix_(rr+1,cc)]) > thresh) )   # [- +]'
#     e[(rx,cx)] = 1    
#     (rx,cx) = np.nonzero(
#         np.logical_and(np.logical_and(DoG[np.ix_(rr-1,cc)] > 0, DoG[np.ix_(rr,cc)] < 0), 
#                        abs( DoG[np.ix_(rr-1,cc)] - DoG[np.ix_(rr,cc)]) > thresh) )   # [+ -]'
#     e[(rx,cx)] = 1
    
#     # Another case: DoG can be precisely zero
#     (rz,cz) = np.nonzero(DoG[np.ix_(rr,cc)] == 0)
#     if rz.size != 0:
#         # Look for the zero crossings: +0-, -0+ and their transposes
#         # The edge lies on the Zero point
        
#         (rx,cx) = np.nonzero(
#             np.logical_and(np.logical_and(DoG[np.ix_(rz,cz-1)] < 0, DoG[np.ix_(rz,cz+1)] > 0), 
#                            abs( DoG[np.ix_(rz,cz+1)] - DoG[np.ix_(rz,cz-1)]) > thresh) )   # [- 0 +]
#         e[(rx,cx)] = 1  
#         (rx,cx) = np.nonzero(
#             np.logical_and(np.logical_and(DoG[np.ix_(rz,cz-1)] > 0, DoG[np.ix_(rz,cz+1)] < 0), 
#                            abs( DoG[np.ix_(rz,cz-1)] - DoG[np.ix_(rz,cz+1)]) > thresh) )   # [+ 0 -]
#         e[(rx,cx)] = 1
#         (rx,cx) = np.nonzero(
#             np.logical_and(np.logical_and(DoG[np.ix_(rz-1,cz)] < 0, DoG[np.ix_(rz+1,cz)] > 0), 
#                            abs( DoG[np.ix_(rz+1,cz)] - DoG[np.ix_(rz-1,cz)]) > thresh) )   # [- 0 +]'
#         e[(rx,cx)] = 1
#         (rx,cx) = np.nonzero(
#             np.logical_and(np.logical_and(DoG[np.ix_(rz-1,cz)] > 0, DoG[np.ix_(rz+1,cz)] < 0), 
#                            abs( DoG[np.ix_(rz-1,cz)] - DoG[np.ix_(rz+1,cz)]) > thresh) )   # [+ 0 -]'
#         e[(rx,cx)] = 1
        
#     return e