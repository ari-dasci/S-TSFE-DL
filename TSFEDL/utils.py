import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import torch
from torch import nn
from torch.nn import functional as F

class TimeDistributed(nn.Module):
    """
    TimeDistributed module implementation.
    """
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)
        t, n = x.size(0), x.size(1)
        # merge batch and seq dimensions
        x_reshape = x.contiguous().view(t * n, *x.size()[2:]).to(torch.float32)
        y = self.module(x_reshape)
        # We have to reshape Y
        y = y.contiguous().view(t, n, *y.size()[1:]).to(torch.float32)
        return y


def flip_indices_for_conv_to_lstm(x: torch.Tensor) -> torch.Tensor:
    """
    Changes the (N, C, L) dimension to (N, L, C). This is due to features in PyTorch's LSTMs are expected on the last dim.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    x : torch.Tensor
        Output tensor.
    """
    return x.view(x.size(0), x.size(2), x.size(1))

def flip_indices_for_conv_to_lstm_reshape(x: torch.Tensor) -> torch.Tensor:
    """
    Changes the (N, C, L) dimension to (N, L, C). This is due to features in PyTorch's LSTMs are expected on the last dim.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    x : torch.Tensor
        Output tensor.
    """
    return x.reshape(x.size(0), x.size(2), x.size(1))


def check_inputs(include_top, weights, input_tensor, input_shape, classes, classifier_activation):
    """
    Auxiliar function for checking the input parameters of the models.

    Parameters
    ----------
    include_top : bool
        Boolean value to control if the classification module should be placed in the model.
    weights : str
        Route to the saved weight of the model.
    input_tensor : keras.Tensor
        Input tensor of the model.
    input_shape : tuple
        Tuple with the input shape of the model.
    classes : int
        Number of classes to predict with the model.
    classifier_activation : str
        "softmax" or None

    Returns
    -------
    inp : Keras.Tensor
        Input tensor.
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

    Parameters
    ----------
    x : Keras.Tensor
        Input tensor of the full convolution.
    filters : int
        Number of filters of the full convolution.
    kernel_size : int
        Kernel size of the convolution.
    kwargs : dict
        Rest of the arguments, optional.

    Returns
    -------
    x : Keras.Tensor
        Output tensor.
    """
    # Do a full convolution. Return a keras Tensor
    x = layers.ZeroPadding1D(padding=kernel_size - 1)(x)
    x = layers.Conv1D(filters=filters, kernel_size=kernel_size, **kwargs)(x)
    return x
