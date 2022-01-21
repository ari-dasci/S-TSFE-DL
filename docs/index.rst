**********
TSFEDL
**********

TSFEDL is a Python library that contains a wide range of models for working with time-series data
using hybrid deep learning models that uses both a Convolutional Neural Network (CNN) and a Recurrent
Neural Network (RNN). The use of these models allow both to capture the spatial features and the time
dependencies between timesteps.

This library contains the implementation of 22 architectures both in Tensorflow+Keras and PyTorch.

Installation
============

To install TSFEDL, we recommend using ``pip``.

.. code-block:: bash

    pip install TSFEDL

.. toctree::
   :maxdepth: 2

   models_keras

   models_pytorch

   blocks_pytorch

   blocks

   data

   utils


* :ref:`genindex`

.. include:: ../CHANGES.rst
