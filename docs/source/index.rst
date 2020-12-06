.. caer documentation master file, created by
   sphinx-quickstart on Tue Oct  6 10:38:42 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


Documentation
===================================

Caer is a lightweight Computer Vision library for high-performance AI research. It simplifies your approach towards Computer Vision by abstracting away unnecessary boilerplate code enabling maximum flexibility. By offering powerful image and video processing algorithms, Caer provides both casual and advanced users with an elegant interface for Machine vision operations.

It leverages the power of libraries like OpenCV and Pillow to speed up your Computer Vision workflow — making it fully compatible with other frameworks such as PyTorch and Tensorflow.

This design philosophy makes Caer ideal for students, researchers, hobbyists and even experts in the fields of Deep Learning and Computer Vision to quickly prototype deep learning models or research ideas.


.. toctree::
   :maxdepth: 1
   :caption: First Steps

   getting_started/caer_at_a_glance.rst
   getting_started/install.rst


.. toctree::
   :maxdepth: 1
   :caption: Caer API

   caer
   caer.augment
   caer.color
   caer.data
   caer.distance
   caer.filters
   caer.morph
   caer.path
   caer.preprocessing
   caer.segmentation
   caer.transforms
   caer.utils
   caer.video


.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Getting Started

   self
   installation
   tutorials
   contribute
   modules

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Documentation

   api/classification
   api/regression
   api/clustering
   api/anomaly
   api/nlp
   api/arules
   api/datasets


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`