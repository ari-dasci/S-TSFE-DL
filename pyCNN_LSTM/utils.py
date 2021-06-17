import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import torch
from torch import nn
from torch.nn import functional as F


def check_inputs(include_top, weights, input_tensor, input_shape, classes, classifier_activation):
    """
        Auxiliar function for checking the input parameters of the models.
    """
    if include_top:
        if not isinstance(classes, int):
            raise ValueError("'classes' must be an int value.")
        act = keras.activations.get(classifier_activation)
        if act not in {keras.activations.get('softmax'), keras.activations.get(None)}:
            raise ValueError("'classifier_activation' must be 'softmax' or None.")

    if weights is not None and not tf.io.gfile.exists(weights):
        raise ValueError("'weights' path does not exists: ", weights)

    # Determine input
    if input_tensor is None:
        if input_shape is not None:
            inp = layers.Input(shape=input_shape)
        else:
            raise ValueError("One of input_tensor or input_shape should not be None.")
    else:
        inp = input_tensor

    return inp


def full_convolution(x, filters, kernel_size, **kwargs):
    """
        It performs a Full convolution operation on the given keras Tensor.
    """
    # Do a full convolution. Return a keras Tensor
    x = layers.ZeroPadding1D(padding=kernel_size - 1)(x)
    x = layers.Conv1D(filters=filters, kernel_size=kernel_size, **kwargs)(x)
    return x