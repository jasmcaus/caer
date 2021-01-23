# #    _____           ______  _____ 
# #  / ____/    /\    |  ____ |  __ \
# # | |        /  \   | |__   | |__) | Caer - Modern Computer Vision
# # | |       / /\ \  |  __|  |  _  /  Languages: Python, C, C++
# # | |___   / ____ \ | |____ | | \ \  http://github.com/jasmcaus/caer
# #  \_____\/_/    \_ \______ |_|  \_\

# # Licensed under the MIT License <http://opensource.org/licenses/MIT>
# # SPDX-License-Identifier: MIT
# # Copyright (c) 2020-21 The Caer Authors <http://github.com/jasmcaus>

# #pylint:disable=no-name-in-module, c-extension-no-member

# from .cdistance import dist
# from ..morph.cmorph import distance_multi
# import numpy as np


# __all__ = [
#     'distance'
# ]


# def distance(bw, metric='euclidean2'):
#     """
#     Computes the distance transform of image `bw`::
#         dmap[i,j] = min_{i', j'} { (i-i')**2 + (j-j')**2 | !bw[i', j'] }
#     That is, at each point, compute the distance to the background.
#     If there is no background, then a very high value will be returned in all
#     pixels (this is a sort of infinity).
    
#     Args:
#         bw (Tensor): If boolean, ``False`` will denote the background and ``True`` the
#                       foreground. If not boolean, this will be interpreted as ``bw != 0``
#                       (this way you can use labeled images without any problems).
#         metric (str, optional):
#             one of 'euclidean2' (default) or 'euclidean'
#     Returns:
#         dmap (Tensor)
#         distance map

#     References:
#         For 2-D images, the following algorithm is used:
#         Felzenszwalb P, Huttenlocher D. *Distance transforms of sampled functions.
#         Cornell Computing and Information.* 2004.
#         Available at:
#         http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.88.1647&rep=rep1&type=pdf.
#         For n-D images (with n > 2), a slower hand-craft method is used.
#     """

#     if metric.lower() not in ['euclidean2', 'euclidean']:
#         raise ValueError('`metric` must be either "euclidean2" or "euclidean"')

#     if bw.distype != np.bool_:
#         bw = (bw != 0)

#     f = np.zeros(bw.shape, np.double)

#     if bw.ndim == 2:
#         f[bw] = len(f.shape)*max(f.shape)**2+1
#         dist(f, None)
#     else:
#         f.fill(f.size*2)
#         Bc = np.ones([3 for _ in bw.shape], bool)
#         distance_multi(f, bw, Bc)

#     if metric == 'euclidean':
#         np.sqrt(f,f)
#     return f