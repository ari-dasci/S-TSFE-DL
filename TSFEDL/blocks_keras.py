from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.python.keras.activations import relu, sigmoid


def densenet_transition_block(x, reduction, name):
    """A transition block of densenet for 1D data.

    Parameters
    ----------
      x: input tensor.
      reduction: float
        Compression rate at transition layers.
      name: str
        Block label.

    Returns
    -------
    x : output tensor for the block.
    """
    bn_axis = 2
    x = layers.BatchNormalization(axis=2, epsilon=1.001e-5, name=name + '_bn')(x)
    x = layers.Activation('relu', name=name + '_relu')(x)
    x = layers.Conv1D(int(x.shape[bn_axis] * reduction), 1, use_bias=False, name=name + '_conv')(x)
    x = layers.AveragePooling1D(2, strides=2, name=name + '_pool')(x)
    return x


def densenet_conv_block(x, growth_rate, name):
    """A building block for a dense block from densenet for 1D data.

    Parameters
    ----------
      x: input tensor.
      growth_rate: float
        Growth rate at dense layers.
      name: str
        Block label.

    Returns
    -------
    x : Output tensor for the block.
    """
    bn_axis = 2  # 3 if backend.image_data_format() == 'channels_last' else 1
    x1 = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(x)
    x1 = layers.Activation('relu', name=name + '_0_relu')(x1)
    x1 = layers.Conv1D(4 * growth_rate, 1, use_bias=False, name=name + '_1_conv')(x1)
    x1 = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x1)
    x1 = layers.Activation('relu', name=name + '_1_relu')(x1)
    x1 = layers.Conv1D(growth_rate, 3, padding='same', use_bias=False, name=name + '_2_conv')(x1)
    x = layers.Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
    return x


def densenet_dense_block(x, blocks, growth_rate, name):
    """A dense block of densenet for 1D data.

    Parameters
    ----------
      x: input tensor.
      blocks: int
        The number of building blocks.
      name: str
        Block label.

    Returns
    -------
    x : Output tensor for the block.
    """
    for i in range(blocks):
        x = densenet_conv_block(x, growth_rate, name=name + '_block' + str(i + 1))
    return x


def squeeze_excitation_module(x, dense_units):
    """
    Squeeze-and-Excitation Module.

    References
    ----------
     Squeeze-and-Excitation Networks, Jie Hu, Li Shen, Samuel Albanie, Gang Sun, Enhua Wu (arXiv:1709.01507v4)

    Arguments
    ---------
      x: keras.Tensor
        The input tensor.

      dense_units: int
        The number units on each dense layer.

    Returns
    -------
    se: Keras.Tensor
      Output tensor for the block.
    """
    se = layers.GlobalAveragePooling1D()(x)
    # Reshape the output to mach the number of dimensions of the input shape: (None, 1, Channels)
    se = layers.Reshape(target_shape=(1, x.shape[2]))(se)
    se = layers.Dense(units=dense_units, activation=relu)(se)  # Fully-Connected 1
    se = layers.Dense(units=x.shape[2], activation=sigmoid)(se)  # Fully-Connected 2

    # Perform the element-wise multiplication between the inputs and the Squeeze
    se = layers.multiply([x, se])
    return se


def conv_block_YiboGao(in_x, nb_filter, kernel_size):
    """
    Convolutional block of YiboGao's model.

    Parameters
    ----------
    in_x : keras.Tensor
        Input tensor of the convolution bock.
    nb_filter : int
        Number of filerts for the convolution.
    kernel_size : int
        Kernel size of the convolution.

    Returns
    -------
    x : keras.Tensor
        Output tensor of the block.
    """
    x = layers.Conv1D(filters=nb_filter, kernel_size=kernel_size, padding='same')(in_x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation=relu)(x)
    return x


def attention_branch_YiboGao(in_x, nb_filter, kernel_size):
    """
    Attention bronch of YiboGao's model.

    Parameters
    ----------
    in_x : keras.Tensor
        Input tensor.
    nb_filter : int
        Number of filerts for the convolutional YiboGao block.
    kernel_size : int
        Kernel size for the convolutional block.

    Returns
    -------
    x : keras.Tensor
        Output tensor of the block.
    """
    x1 = conv_block_YiboGao(in_x, nb_filter, kernel_size)

    x = layers.MaxPooling1D(pool_size=2)(x1)
    x = conv_block_YiboGao(x, nb_filter, kernel_size)
    x = layers.UpSampling1D(size=2)(x)

    x2 = conv_block_YiboGao(x, nb_filter, kernel_size)

    if (K.int_shape(x1) != K.int_shape(x2)):
        x2 = layers.ZeroPadding1D(padding=1)(x2)
        x2 = layers.Cropping1D((1, 0))(x2)

    x = layers.add([x1, x2])

    x = conv_block_YiboGao(x, nb_filter, kernel_size)

    x = layers.Conv1D(filters=nb_filter, kernel_size=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation=sigmoid)(x)

    return x


def RTA_block(in_x, nb_filter, kernel_size):
    """
    Residual-based Temporal Attention (RTA) block.

    References
    ----------
        Gao, Y., Wang, H., & Liu, Z. (2021). An end-to-end atrial fibrillation detection by a novel residual-based
        temporal attention convolutional neural network with exponential nonlinearity loss.
        Knowledge-Based Systems, 212, 106589.

    Parameters
    ----------
    in_x : keras.Tensor
        Input tensor.
    nb_filter : int
        Number of filerts for the convolutional YiboGao block.
    kernel_size : int
        Kernel size for the convolutional block.

    Returns
    -------
    out : keras.Tensor
        Output tensor of the block.
    """
    x1 = conv_block_YiboGao(in_x, nb_filter, kernel_size)
    x2 = conv_block_YiboGao(x1, nb_filter, kernel_size)

    attention_map = attention_branch_YiboGao(x1, nb_filter, kernel_size)

    x = layers.multiply([x2, attention_map])
    x = layers.add([x, x1])

    out = conv_block_YiboGao(x, nb_filter, kernel_size)

    return out


def spatial_attention_block_ZhangJin(decrease_ratio, x):
    """
    Spatial attention module of ZhangJin's model

    Parameters
    ----------
    x : keras.Tensor
        Input tensor.
    decrease_ratio : int
        Decrease ratio of the number of units in the neural network.

    Returns
    -------
    x : keras.Tensor
        Output tensor of the block.
    """
    shared_dense1 = layers.Dense(units=x.shape[2] // decrease_ratio, activation=relu)
    shared_dense2 = layers.Dense(units=x.shape[2], activation=relu)
    x1 = layers.GlobalAveragePooling1D()(x)
    x1 = shared_dense1(x1)
    x1 = shared_dense2(x1)
    x1 = layers.Reshape(target_shape=(1, x.shape[2]))(x1)

    x2 = layers.GlobalMaxPool1D()(x)
    x2 = shared_dense1(x2)
    x2 = shared_dense2(x2)
    x2 = layers.Reshape(target_shape=(1, x.shape[2]))(x2)

    x = layers.add([x1, x2])
    x = layers.Activation(activation=sigmoid)(x)
    return x


def temporal_attention_block_ZhangJin(x):
    """
    Temporal attention module of ZhangJin's Model.

    Parameters
    ----------
    x : keras.Tensor
        Input tensor.

    Returns
    -------
    x : keras.Tensor
        Output tensor of the block.
    """
    # Temporal attention module
    x1 = layers.GlobalMaxPool1D(data_format='channels_first')(x)
    x1 = layers.Reshape(target_shape=(x1.shape[1], 1))(x1)

    x2 = layers.GlobalAveragePooling1D(data_format='channels_first')(x)
    x2 = layers.Reshape(target_shape=(x2.shape[1], 1))(x2)

    x = layers.Concatenate()([x1, x2])
    x = layers.Conv1D(filters=1, kernel_size=7, padding="same")(x)
    x = layers.Activation(activation=sigmoid)(x)
    return x
