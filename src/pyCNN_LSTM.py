import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.activations import relu, softmax, sigmoid


def __transition_block(x, reduction, name):
    """A transition block of densenet for 1D data.

    Arguments:
      x: input tensor.
      reduction: float, compression rate at transition layers.
      name: string, block label.

    Returns:
      output tensor for the block.
    """
    bn_axis = 2
    x = layers.BatchNormalization(axis=2, epsilon=1.001e-5, name=name + '_bn')(x)
    x = layers.Activation('relu', name=name + '_relu')(x)
    x = layers.Conv1D(int(x.shape[bn_axis] * reduction), 1, use_bias=False, name=name + '_conv')(x)
    x = layers.AveragePooling1D(2, strides=2, name=name + '_pool')(x)
    return x


def __conv_block(x, growth_rate, name):
    """A building block for a dense block from densenet for 1D data.

    Arguments:
      x: input tensor.
      growth_rate: float, growth rate at dense layers.
      name: string, block label.

    Returns:
      Output tensor for the block.
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


def __dense_block(x, blocks, growth_rate, name):
    """A dense block of densenet for 1D data.

    Arguments:
      x: input tensor.
      blocks: integer, the number of building blocks.
      name: string, block label.

    Returns:
      Output tensor for the block.
    """
    for i in range(blocks):
        x = __conv_block(x, growth_rate, name=name + '_block' + str(i + 1))
    return x


def __se_module(x, dense_units):
    """
    SE Module of CaiWenjuan:

    References
    ----------
     Squeeze-and-Excitation Networks, Jie Hu, Li Shen, Samuel Albanie, Gang Sun, Enhua Wu (arXiv:1709.01507v4)

    Arguments
    ---------
      x: input tensor.
      dense_units: integer, the number units on each dense layer

    Returns
    -------
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


def __check_inputs(include_top, weights, input_tensor, input_shape, classes, classifier_activation):
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


def OhShuLih(include_top=True,
             weights=None,
             input_tensor=None,
             input_shape=None,
             classes=5,
             classifier_activation="softmax"):
    """
    CNN+LSTM model.

    References
    ----------

        Oh, Shu Lih, et al. "Automated diagnosis of arrhythmia using combination of CNN and LSTM techniques with
        variable length heart beats." Computers in biology and medicine 102 (2018): 278-287.

    Parameters
    ----------
         include_top: bool, default=True

           Whether to include the fully-connected layer at the top of the network.

        weights: str, default=None

            The path to the weights file to be loaded.

        input_tensor: keras.Tensor, defaults=None

            Optional Keras tensor (i.e. output of `layers.Input()`) to use as input for the model.

        input_shape: Tuple, defaults=None

            If `input_tensor=None`, a tuple that defines the input shape for the model.

        classes: int, defaults=5

            If `include_top=True`, the number of units in the top layer to classify data.

        classifier_activation: str or callable, defaults='softmax'

            The activation function to use on the "top" layer. Ignored unless `include_top=True`. Set
            `classifier_activation=None` to return the logits of the "top" layer.

    Returns
    -------

        A `keras.Model` instance.
    """

    # Initial checks
    inp = __check_inputs(include_top, weights, input_tensor, input_shape, classes, classifier_activation)

    # Model Definition
    x = layers.Conv1D(filters=64, kernel_size=3, activation=relu)(inp)
    x = layers.MaxPooling1D(pool_size=4)(x)
    x = layers.Dropout(rate=0.3)(x)
    x = layers.Conv1D(filters=16, kernel_size=3, activation=relu)(x)
    x = layers.MaxPooling1D(pool_size=4)(x)
    x = layers.Dropout(rate=0.3)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LSTM(units=16)(x)

    if include_top:
        x = layers.Flatten()(x)
        x = layers.Dense(units=classes, activation=classifier_activation)(x)

    model = keras.Model(inputs=inp, outputs=x, name="OhShuLih")

    # Load weights
    if weights is not None:
        model.load_weights(weights)

    return model


def KhanZulfiqar(include_top=True,
                 weights=None,
                 input_tensor=None,
                 input_shape=None,
                 classes=5,
                 classifier_activation="softmax"):
    """
    References
    ---------
        Khan, Zulfiqar Ahmad, et al. "Towards Efficient Electricity Forecasting in Residential and Commercial
        Buildings: A Novel Hybrid CNN with a LSTM-AE based Framework." Sensors 20.5 (2020): 1399.

    Parameters
    ----------
         include_top: bool, default=True

           Whether to include the fully-connected layer at the top of the network.

        weights: str, default=None

            The path to the weights file to be loaded.

        input_tensor: keras.Tensor, defaults=None

            Optional Keras tensor (i.e. output of `layers.Input()`) to use as input for the model.

        input_shape: Tuple, defaults=None

            If `input_tensor=None`, a tuple that defines the input shape for the model.

        classes: int, defaults=5

            If `include_top=True`, the number of units in the top layer to classify data.

        classifier_activation: str or callable, defaults='softmax'

            The activation function to use on the "top" layer. Ignored unless `include_top=True`. Set
            `classifier_activation=None` to return the logits of the "top" layer.

    Returns
    -------

        A `keras.Model` instance.
     """

    # Check inputs
    inp = __check_inputs(include_top, weights, input_tensor, input_shape, classes, classifier_activation)

    # Model Definition
    x = layers.Conv1D(filters=8, kernel_size=1, padding='same', activation=relu)(inp)
    x = layers.Dropout(0.1)(x)
    x = layers.Conv1D(filters=16, kernel_size=1, padding='same', activation=relu)(x)
    x = layers.Dropout(0.1)(x)
    x = layers.LSTM(50, activation=relu, return_sequences=True)(x)
    x = layers.LSTM(50, activation=relu, return_sequences=True)(x)

    if include_top:
        x = layers.Flatten()(x)
        x = layers.Dense(units=classes, activation=classifier_activation)(x)

    model = keras.Model(inputs=inp, outputs=x, name="KhanZulfiqar")

    if weights is not None:
        model.load_weights(weights)

    return model


def ZhengZhenyu(include_top=True,
                weights=None,
                input_tensor=None,
                input_shape=None,
                classes=5,
                classifier_activation="softmax"):
    """
    Parameters
    ----------
         include_top: bool, default=True

           Whether to include the fully-connected layer at the top of the network.

        weights: str, default=None

            The path to the weights file to be loaded.

        input_tensor: keras.Tensor, defaults=None

            Optional Keras tensor (i.e. output of `layers.Input()`) to use as input for the model.

        input_shape: Tuple, defaults=None

            If `input_tensor=None`, a tuple that defines the input shape for the model.

        classes: int, defaults=5

            If `include_top=True`, the number of units in the top layer to classify data.

        classifier_activation: str or callable, defaults='softmax'

            The activation function to use on the "top" layer. Ignored unless `include_top=True`. Set
            `classifier_activation=None` to return the logits of the "top" layer.

    Returns
    -------

        A `keras.Model` instance.
    """
    inp = __check_inputs(include_top, weights, input_tensor, input_shape, classes, classifier_activation)

    # Model Definition
    x = inp
    for f in [64, 128, 256]:
        x = layers.Conv1D(filters=f, kernel_size=3, strides=1, padding='same', activation='relu')(x)
        x = layers.Conv1D(filters=f, kernel_size=3, strides=1, padding='same', activation='relu')(x)
        x = layers.MaxPooling1D(4, strides=1)(x)

    x = layers.LSTM(16, return_sequences=True)(x)

    if include_top:
        x = layers.Flatten()(x)
        x = layers.Dense(units=classes, activation=classifier_activation)(x)

    model = keras.Model(inputs=inp, outputs=x, name="ZhengZhenyu")

    # Load weights
    if weights is not None:
        model.load_weights(weights)

    return model


def HouBoroui(include_top=True,
              weights=None,
              input_tensor=None,
              input_shape=None,
              classes=5,
              classifier_activation="softmax"):
    """
    References
    ----------
        Hou, Borui, et al. "LSTM based auto-encoder model for ECG arrhythmias classification." IEEE Transactions on
        Instrumentation and Measurement (2019).

    Parameters
    ----------
         include_top: bool, default=True

           Whether to include the fully-connected layer at the top of the network.

        weights: str, default=None

            The path to the weights file to be loaded.

        input_tensor: keras.Tensor, defaults=None

            Optional Keras tensor (i.e. output of `layers.Input()`) to use as input for the model.

        input_shape: Tuple, defaults=None

            If `input_tensor=None`, a tuple that defines the input shape for the model.

        classes: int, defaults=5

            If `include_top=True`, the number of units in the top layer to classify data.

        classifier_activation: str or callable, defaults='softmax'

            The activation function to use on the "top" layer. Ignored unless `include_top=True`. Set
            `classifier_activation=None` to return the logits of the "top" layer.

    Returns
    -------

        A `keras.Model` instance.
    """
    inp = __check_inputs(include_top, weights, input_tensor, input_shape, classes, classifier_activation)

    # Model definition
    x = layers.LSTM(units=146, return_sequences=True)(inp)
    x = layers.LSTM(units=31, return_sequences=True)(x)

    if include_top:
        x = layers.Flatten()(x)
        x = layers.Dense(units=classes, activation=classifier_activation)(x)

    model = keras.Model(inputs=inp, outputs=x, name="HouBoroui")

    if weights is not None:
        model.load_weights(weights)

    return model


def WangKejun(include_top=True,
              weights=None,
              input_tensor=None,
              input_shape=None,
              classes=5,
              classifier_activation="softmax"):
    """
    References
    ----------
        Wang, Kejun, Xiaoxia Qi, and Hongda Liu. "Photovoltaic power forecasting based LSTM-Convolutional Network."
        Energy 189 (2019): 116225.

    Parameters
    ----------
         include_top: bool, default=True

           Whether to include the fully-connected layer at the top of the network.

        weights: str, default=None

            The path to the weights file to be loaded.

        input_tensor: keras.Tensor, defaults=None

            Optional Keras tensor (i.e. output of `layers.Input()`) to use as input for the model.

        input_shape: Tuple, defaults=None

            If `input_tensor=None`, a tuple that defines the input shape for the model.

        classes: int, defaults=5

            If `include_top=True`, the number of units in the top layer to classify data.

        classifier_activation: str or callable, defaults='softmax'

            The activation function to use on the "top" layer. Ignored unless `include_top=True`. Set
            `classifier_activation=None` to return the logits of the "top" layer.

    Returns
    -------

        A `keras.Model` instance.
    """
    inp = __check_inputs(include_top, weights, input_tensor, input_shape, classes, classifier_activation)

    # Model definition
    x = layers.LSTM(64, return_sequences=True)(inp)
    x = layers.LSTM(128, return_sequences=True)(x)
    x = layers.Conv1D(filters=64, kernel_size=3, strides=1)(x)
    x = layers.MaxPooling1D(2, strides=2)(x)
    x = layers.Conv1D(filters=150, kernel_size=2, strides=1)(x)
    x = layers.MaxPooling1D(2, strides=2)(x)
    x = layers.Dropout(0.1)(x)

    if include_top:
        x = layers.Flatten()(x)
        x = layers.Dense(units=classes, activation=classifier_activation)(x)

    model = keras.Model(inputs=inp, outputs=x, name="WangKejun")

    if weights is not None:
        model.load_weights(weights)

    return model


def ChenChen(include_top=True,
             weights=None,
             input_tensor=None,
             input_shape=None,
             classes=5,
             classifier_activation="softmax"):
    """
    References
    ----------
        Chen, Chen, et al. "Automated arrhythmia classification based on a combination network of CNN and LSTM."
        Biomedical Signal Processing and Control 57 (2020): 101819.

    Parameters
    ----------
         include_top: bool, default=True

           Whether to include the fully-connected layer at the top of the network.

        weights: str, default=None

            The path to the weights file to be loaded.

        input_tensor: keras.Tensor, defaults=None

            Optional Keras tensor (i.e. output of `layers.Input()`) to use as input for the model.

        input_shape: Tuple, defaults=None

            If `input_tensor=None`, a tuple that defines the input shape for the model.

        classes: int, defaults=5

            If `include_top=True`, the number of units in the top layer to classify data.

        classifier_activation: str or callable, defaults='softmax'

            The activation function to use on the "top" layer. Ignored unless `include_top=True`. Set
            `classifier_activation=None` to return the logits of the "top" layer.

    Returns
    -------

        A `keras.Model` instance.
    """
    inp = __check_inputs(include_top, weights, input_tensor, input_shape, classes, classifier_activation)

    # Model definition
    x = inp
    for f in [251, 150, 100, 81, 61, 14]:
        x = layers.Conv1D(filters=f, kernel_size=2, strides=1)(x)
        x = layers.MaxPooling1D(pool_size=2, strides=1)(x)
    x = layers.LSTM(units=64, return_sequences=True)(x)
    x = layers.LSTM(units=32, return_sequences=True)(x)

    if include_top:
        x = layers.Flatten()(x)
        x = layers.Dense(units=classes, activation=classifier_activation)(x)

    model = keras.Model(inputs=inp, outputs=x, name="ChenChen")

    if weights is not None:
        model.load_weights(weights)

    return model


def KimTaeYoung(include_top=True,
                weights=None,
                input_tensor=None,
                input_shape=None,
                classes=5,
                classifier_activation="softmax"):
    """
    References
    ----------
        Kim, Tae-Young, and Sung-Bae Cho. "Predicting residential energy consumption using CNN-LSTM neural networks."
        Energy 182 (2019): 72-81.

    Parameters
    ----------
         include_top: bool, default=True

           Whether to include the fully-connected layer at the top of the network.

        weights: str, default=None

            The path to the weights file to be loaded.

        input_tensor: keras.Tensor, defaults=None

            Optional Keras tensor (i.e. output of `layers.Input()`) to use as input for the model.

        input_shape: Tuple, defaults=None

            If `input_tensor=None`, a tuple that defines the input shape for the model.

        classes: int, defaults=5

            If `include_top=True`, the number of units in the top layer to classify data.

        classifier_activation: str or callable, defaults='softmax'

            The activation function to use on the "top" layer. Ignored unless `include_top=True`. Set
            `classifier_activation=None` to return the logits of the "top" layer.

    Returns
    -------

        A `keras.Model` instance.
    """
    inp = __check_inputs(include_top, weights, input_tensor, input_shape, classes, classifier_activation)

    # Model definition
    x = inp
    for f in [64, 150]:
        x = layers.Conv1D(filters=f, kernel_size=2, strides=1)(x)
        x = layers.MaxPooling1D(pool_size=2, strides=1)(x)
    x = layers.LSTM(units=64, activation=keras.activations.tanh, return_sequences=True)(x)

    if include_top:
        x = layers.Flatten()(x)
        x = layers.Dense(units=classes, activation=classifier_activation)(x)

    model = keras.Model(inputs=inp, outputs=x, name="KimTaeYoung")

    if weights is not None:
        model.load_weights(weights)

    return model


def GenMinxing(include_top=True,
               weights=None,
               input_tensor=None,
               input_shape=None,
               classes=5,
               classifier_activation="softmax"):
    """
    References
    ----------
        Geng, Minxing, et al. "Epileptic Seizure Detection Based on Stockwell Transform and Bidirectional Long
        Short-Term Memory." IEEE Transactions on Neural Systems and Rehabilitation Engineering 28.3 (2020): 573-580.

    Parameters
    ----------
         include_top: bool, default=True

           Whether to include the fully-connected layer at the top of the network.

        weights: str, default=None

            The path to the weights file to be loaded.

        input_tensor: keras.Tensor, defaults=None

            Optional Keras tensor (i.e. output of `layers.Input()`) to use as input for the model.

        input_shape: Tuple, defaults=None

            If `input_tensor=None`, a tuple that defines the input shape for the model.

        classes: int, defaults=5

            If `include_top=True`, the number of units in the top layer to classify data.

        classifier_activation: str or callable, defaults='softmax'

            The activation function to use on the "top" layer. Ignored unless `include_top=True`. Set
            `classifier_activation=None` to return the logits of the "top" layer.

    Returns
    -------

        A `keras.Model` instance.
    """
    inp = __check_inputs(include_top, weights, input_tensor, input_shape, classes, classifier_activation)

    # Model definition
    x = layers.Bidirectional(layers.LSTM(units=40, return_sequences=True))(inp)
    if include_top:
        x = layers.Flatten()(x)
        x = layers.Dense(units=classes, activation=classifier_activation)(x)

    model = keras.Model(inputs=inp, outputs=x, name="GenMixing")

    if weights is not None:
        model.load_weights(weights)

    return model


def FuJiangmeng(include_top=True,
                weights=None,
                input_tensor=None,
                input_shape=None,
                classes=5,
                classifier_activation="softmax"):
    """
    References
    ----------
        Fu, Jiangmeng, et al. "A hybrid CNN-LSTM model based actuator fault diagnosis for six-rotor UAVs." 2019
        Chinese Control And Decision Conference (CCDC). IEEE, 2019.

    Parameters
    ----------
         include_top: bool, default=True

           Whether to include the fully-connected layer at the top of the network.

        weights: str, default=None

            The path to the weights file to be loaded.

        input_tensor: keras.Tensor, defaults=None

            Optional Keras tensor (i.e. output of `layers.Input()`) to use as input for the model.

        input_shape: Tuple, defaults=None

            If `input_tensor=None`, a tuple that defines the input shape for the model.

        classes: int, defaults=5

            If `include_top=True`, the number of units in the top layer to classify data.

        classifier_activation: str or callable, defaults='softmax'

            The activation function to use on the "top" layer. Ignored unless `include_top=True`. Set
            `classifier_activation=None` to return the logits of the "top" layer.

    Returns
    -------

        A `keras.Model` instance.
    """
    inp = __check_inputs(include_top, weights, input_tensor, input_shape, classes, classifier_activation)

    # Model definition
    x = layers.Conv1D(filters=32, kernel_size=1, padding='same', activation=relu)(inp)
    x = layers.MaxPooling1D(pool_size=2, strides=1)(x)
    x = layers.LSTM(units=256, activation=relu, return_sequences=True)(x)

    if include_top:
        x = layers.Flatten()(x)
        x = layers.Dense(units=classes, activation=classifier_activation)(x)

    model = keras.Model(inputs=inp, outputs=x, name="FuJiangmeng")

    if weights is not None:
        model.load_weights(weights)

    return model


def ShiHaotian(include_top=True,
               weights=None,
               input_tensor=None,
               input_shape=None,
               classes=5,
               classifier_activation="softmax"):
    """
    References
    ----------
        Shi, Haotian, et al. "Automated heartbeat classification based on deep neural network with multiple input
        layers." Knowledge-Based Systems 188 (2020): 105036.

    Parameters
    ----------
         include_top: bool, default=True

           Whether to include the fully-connected layer at the top of the network.

        weights: str, default=None

            The path to the weights file to be loaded.

        input_tensor: keras.Tensor, defaults=None

            Optional Keras tensor (i.e. output of `layers.Input()`) to use as input for the model.

        input_shape: Tuple, defaults=None

            If `input_tensor=None`, a tuple that defines the input shape for the model.

        classes: int, defaults=5

            If `include_top=True`, the number of units in the top layer to classify data.

        classifier_activation: str or callable, defaults='softmax'

            The activation function to use on the "top" layer. Ignored unless `include_top=True`. Set
            `classifier_activation=None` to return the logits of the "top" layer.

    Returns
    -------

        A `keras.Model` instance.
    """
    inp = __check_inputs(include_top, weights, input_tensor, input_shape, classes, classifier_activation)

    # Model Definition
    x1 = layers.Conv1D(filters=32, kernel_size=13, strides=2, activation=relu)(inp)
    x1 = layers.MaxPooling1D(pool_size=2, strides=2)(x1)

    x2 = layers.Conv1D(filters=32, kernel_size=13, strides=1, activation=relu)(inp)
    x2 = layers.MaxPooling1D(pool_size=2, strides=2)(x2)

    x3 = layers.Conv1D(filters=32, kernel_size=13, strides=2, activation=relu)(inp)
    x3 = layers.MaxPooling1D(pool_size=2, strides=2)(x3)

    x = layers.Concatenate(axis=1)([x1, x2, x3])
    x = layers.LSTM(units=32, return_sequences=True)(x)

    if include_top:
        x = layers.Flatten()(x)
        x = layers.Dense(units=classes, activation=classifier_activation)(x)

    model = keras.Model(inputs=inp, outputs=x, name="ShiHaotian")

    if weights is not None:
        model.load_weights(weights)

    return model


def HuangMeiLing(include_top=True,
                weights=None,
                input_tensor=None,
                input_shape=None,
                classes=5,
                classifier_activation="softmax"):
    """
     CNN model, employed for 2-D images of ECG data. This model is adapted for 1-D time series

     References
     -------
         Huang, Mei-Ling, and Yan-Sheng Wu. "Classification of atrial fibrillation and normal sinus rhythm based on
         convolutional neural network." Biomedical Engineering Letters (2020): 1-11.


    Parameters
    ----------
         include_top: bool, default=True

           Whether to include the fully-connected layer at the top of the network.

        weights: str, default=None

            The path to the weights file to be loaded.

        input_tensor: keras.Tensor, defaults=None

            Optional Keras tensor (i.e. output of `layers.Input()`) to use as input for the model.

        input_shape: Tuple, defaults=None

            If `input_tensor=None`, a tuple that defines the input shape for the model.

        classes: int, defaults=5

            If `include_top=True`, the number of units in the top layer to classify data.

        classifier_activation: str or callable, defaults='softmax'

            The activation function to use on the "top" layer. Ignored unless `include_top=True`. Set
            `classifier_activation=None` to return the logits of the "top" layer.

    Returns
    -------

        A `keras.Model` instance.
     """
    inp = __check_inputs(include_top, weights, input_tensor, input_shape, classes, classifier_activation)

    # Model definition
    x = inp
    for f, k, s in zip([48, 256],
                       [15, 13],
                       [6, 1]):
        x = layers.Conv1D(filters=f, kernel_size=k, strides=s, activation=relu)(x)
        x = layers.MaxPooling1D(pool_size=2, strides=1)(x)

    if include_top:
        x = layers.Flatten()(x)
        x = layers.Dense(units=classes, activation=classifier_activation)(x)

    model = keras.Model(inputs=inp, outputs=x, name="HuangMeiLing")

    if weights is not None:
        model.load_weights(weights)

    return model


def LihOhShu(include_top=True,
             weights=None,
             input_tensor=None,
             input_shape=None,
             classes=5,
             classifier_activation="softmax"):
    """
    CNN+LSTM model

    References
    ----------
        Lih, Oh Shu, et al. "Comprehensive electrocardiographic diagnosis based on deep learning." Artificial
        Intelligence in Medicine 103 (2020): 101789.

    Parameters
    ----------
         include_top: bool, default=True

           Whether to include the fully-connected layer at the top of the network.

        weights: str, default=None

            The path to the weights file to be loaded.

        input_tensor: keras.Tensor, defaults=None

            Optional Keras tensor (i.e. output of `layers.Input()`) to use as input for the model.

        input_shape: Tuple, defaults=None

            If `input_tensor=None`, a tuple that defines the input shape for the model.

        classes: int, defaults=5

            If `include_top=True`, the number of units in the top layer to classify data.

        classifier_activation: str or callable, defaults='softmax'

            The activation function to use on the "top" layer. Ignored unless `include_top=True`. Set
            `classifier_activation=None` to return the logits of the "top" layer.

    Returns
    -------

        A `keras.Model` instance.

    """
    inp = __check_inputs(include_top, weights, input_tensor, input_shape, classes, classifier_activation)

    # Model definition
    x = inp
    for filters, k_size in zip([3, 6, 6, 6, 6],
                               [20, 10, 5, 5, 10]):
        x = layers.Conv1D(filters=filters, activation=relu, kernel_size=k_size, strides=1)(x)
        x = layers.MaxPooling1D(pool_size=2, strides=1)(x)

    x = layers.LSTM(units=10, return_sequences=True)(x)

    if include_top:
        x = layers.Flatten()(x)
        x = layers.Dense(units=classes, activation=classifier_activation)(x)

    model = keras.Model(inputs=inp, outputs=x, name="LihOhShu")

    if weights is not None:
        model.load_weights(weights)

    return model


def GaoJunLi(include_top=True,
             weights=None,
             input_tensor=None,
             input_shape=None,
             classes=5,
             classifier_activation="softmax"):
    """
    LSTM network

    References
    ----------
        Gao, Junli, et al. "An Effective LSTM Recurrent Network to Detect Arrhythmia on Imbalanced ECG Dataset."
        Journal of healthcare engineering 2019 (2019).

    Parameters
    ----------
         include_top: bool, default=True

           Whether to include the fully-connected layer at the top of the network.

        weights: str, default=None

            The path to the weights file to be loaded.

        input_tensor: keras.Tensor, defaults=None

            Optional Keras tensor (i.e. output of `layers.Input()`) to use as input for the model.

        input_shape: Tuple, defaults=None

            If `input_tensor=None`, a tuple that defines the input shape for the model.

        classes: int, defaults=5

            If `include_top=True`, the number of units in the top layer to classify data.

        classifier_activation: str or callable, defaults='softmax'

            The activation function to use on the "top" layer. Ignored unless `include_top=True`. Set
            `classifier_activation=None` to return the logits of the "top" layer.

    Returns
    -------

        A `keras.Model` instance.
    """
    inp = __check_inputs(include_top, weights, input_tensor, input_shape, classes, classifier_activation)

    # Model definition
    x = layers.LSTM(units=64, return_sequences=True)(inp)

    if include_top:
        x = layers.Flatten()(x)
        x = layers.Dense(units=classes, activation=classifier_activation)(x)

    model = keras.Model(inputs=inp, outputs=x, name="GaoJunLi")

    if weights is not None:
        model.load_weights(weights)

    return model


def WeiXiaoyan(include_top=True,
               weights=None,
               input_tensor=None,
               input_shape=None,
               classes=5,
               classifier_activation="softmax"):
    """
     CNN+LSTM model employed for 2-D images of ECG data. This model is adapted for 1-D time series.


     References
     ----------

         Wei, Xiaoyan, et al. "Early prediction of epileptic seizures using a long-term recurrent convolutional
         network." Journal of neuroscience methods 327 (2019): 108395.

    Parameters
    ----------
         include_top: bool, default=True

           Whether to include the fully-connected layer at the top of the network.

        weights: str, default=None

            The path to the weights file to be loaded.

        input_tensor: keras.Tensor, defaults=None

            Optional Keras tensor (i.e. output of `layers.Input()`) to use as input for the model.

        input_shape: Tuple, defaults=None

            If `input_tensor=None`, a tuple that defines the input shape for the model.

        classes: int, defaults=5

            If `include_top=True`, the number of units in the top layer to classify data.

        classifier_activation: str or callable, defaults='softmax'

            The activation function to use on the "top" layer. Ignored unless `include_top=True`. Set
            `classifier_activation=None` to return the logits of the "top" layer.

    Returns
    -------

        A `keras.Model` instance.

     """
    inp = __check_inputs(include_top, weights, input_tensor, input_shape, classes, classifier_activation)

    # Model definition
    x = inp
    # Network Defintion (CNN)
    for filters, kernel, strides in zip([32, 64, 128, 256, 512],
                                        [5, 3, 3, 3, 3],
                                        [1, 1, 1, 1, 1]):
        x = layers.Conv1D(filters=filters, kernel_size=kernel, strides=strides)(x)
        x = layers.LeakyReLU(alpha=0.01)(x)
        x = layers.MaxPooling1D(pool_size=2, strides=2)(x)
        x = layers.BatchNormalization()(x)

    # LSTM
    x = layers.LSTM(units=512, return_sequences=True)(x)
    x = layers.LSTM(units=512, return_sequences=True)(x)

    if include_top:
        x = layers.Flatten()(x)
        x = layers.Dense(units=classes, activation=classifier_activation)(x)

    model = keras.Model(inputs=inp, outputs=x, name="WeiXiaoyan")

    if weights is not None:
        model.load_weights(weights)

    return model


def KongZhengmin(include_top=True,
                 weights=None,
                 input_tensor=None,
                 input_shape=None,
                 classes=5,
                 classifier_activation="softmax"):
    """
    CNN+LSTM

    References
    ----------
        Kong, Zhengmin, et al. "Convolution and Long Short-Term Memory Hybrid Deep Neural Networks for Remaining
        Useful Life Prognostics." Applied Sciences 9.19 (2019): 4156.

    Parameters
    ----------
         include_top: bool, default=True

           Whether to include the fully-connected layer at the top of the network.

        weights: str, default=None

            The path to the weights file to be loaded.

        input_tensor: keras.Tensor, defaults=None

            Optional Keras tensor (i.e. output of `layers.Input()`) to use as input for the model.

        input_shape: Tuple, defaults=None

            If `input_tensor=None`, a tuple that defines the input shape for the model.

        classes: int, defaults=5

            If `include_top=True`, the number of units in the top layer to classify data.

        classifier_activation: str or callable, defaults='softmax'

            The activation function to use on the "top" layer. Ignored unless `include_top=True`. Set
            `classifier_activation=None` to return the logits of the "top" layer.

    Returns
    -------

        A `keras.Model` instance.

    """
    inp = __check_inputs(include_top, weights, input_tensor, input_shape, classes, classifier_activation)

    # Model definition
    x = layers.Conv1D(filters=32, activation=relu, kernel_size=5, strides=1)(inp)
    x = layers.MaxPooling1D(pool_size=2, strides=2)(x)
    x = layers.LSTM(64, return_sequences=True)(x)
    x = layers.LSTM(64, return_sequences=True)(x)

    if include_top:
        x = layers.Flatten()(x)
        x = layers.Dense(units=classes, activation=classifier_activation)(x)

    model = keras.Model(inputs=inp, outputs=x, name="KongZhengmin")

    if weights is not None:
        model.load_weights(weights)

    return model


def YildirimOzal(include_top=True,
                 autoencoder_weights=None,
                 lstm_weights=None,
                 input_tensor=None,
                 input_shape=None,
                 classes=5,
                 classifier_activation="softmax"):
    """CAE-LSTM

     References
     ----------
         Yildirim, Ozal, et al. "A new approach for arrhythmia classification using deep coded features and LSTM
         networks." Computer methods and programs in biomedicine 176 (2019): 121-133.

     Parameters
    ----------
         include_top: bool, default=True

           Whether to include the fully-connected layer at the top of the network.

        autoencoder_weights: str, default=None

            The path to the weights file to be loaded for the autoencoder network

        lstm_weights: str, defaults=None

            The path to the weights file to be loaded for the lstm classifier network.

        input_tensor: keras.Tensor, defaults=None

            Optional Keras tensor (i.e. output of `layers.Input()`) to use as input for the model.

        input_shape: Tuple, defaults=None

            If `input_tensor=None`, a tuple that defines the input shape for the model.

        classes: int, defaults=5

            If `include_top=True`, the number of units in the top layer to classify data.

        classifier_activation: str or callable, defaults='softmax'

            The activation function to use on the "top" layer. Ignored unless `include_top=True`. Set
            `classifier_activation=None` to return the logits of the "top" layer.

    Returns
    -------

        autoenconder -> A `keras.Model` instance with the autoencoder.
        encoder      -> A `keras.Model` instance with only the encoder part
        model        -> A `keras.Model` instance representing the classification model which uses de encoder.

     Examples
     --------
     This method is composed of two parts: An Autoencoder and an LSTM that classifies the encoded data from the encoder
     part. Therefore, it is necessary to firstly train the model:

     >>> inputs = keras.Input((200, 1))
     >>> yildirim = YildirimOzal()
     >>> encoder = yildirim.encoder(inputs)
     >>> x = yildirim.decoder(e)
     >>> autoencoder = keras.Model(inputs=inputs, outputs=x)

     Now, compile and train the autoencoder with .compile() and .fit()

     After that, apply the LSTM classifier
     >>> x = yildirim.lstm(e)
     >>> classifier = keras.Model(inputs=inputs, outputs=x)

     Now, compile and train with .compile() and fit()
     """
    inp = __check_inputs(include_top, None, input_tensor, input_shape, classes, classifier_activation)
    if autoencoder_weights is not None and not tf.io.gfile.exists(autoencoder_weights):
        raise ValueError("'autoencoder_weights' path does not exists: ", autoencoder_weights)
    if lstm_weights is not None and not tf.io.gfile.exists(lstm_weights):
        raise ValueError("'lstm_weights' path does not exists: ", lstm_weights)

    # Model definition
    e = layers.Conv1D(filters=16, kernel_size=5, padding='same', strides=1)(inp)
    e = layers.MaxPooling1D(pool_size=2)(e)
    e = layers.Conv1D(filters=64, kernel_size=5, padding='same', strides=1)(e)
    e = layers.BatchNormalization()(e)
    e = layers.MaxPooling1D(pool_size=2)(e)
    e = layers.Conv1D(filters=32, kernel_size=3, padding='same', strides=1)(e)
    e = layers.Conv1D(filters=1, kernel_size=3, padding='same', strides=1)(e)

    bottleneck = layers.MaxPooling1D(pool_size=2)(e)
    decoder = layers.Conv1D(filters=1, kernel_size=3, padding='same', strides=1)(bottleneck)
    decoder = layers.Conv1D(filters=32, kernel_size=3, padding='same', strides=1)(decoder)
    decoder = layers.UpSampling1D(size=2)(decoder)
    decoder = layers.Conv1D(filters=64, kernel_size=5, padding='same', strides=1)(decoder)
    decoder = layers.UpSampling1D(size=2)(decoder)
    decoder = layers.Conv1D(filters=16, kernel_size=5, padding='same', strides=1)(decoder)
    # Final Dense layer of the autoencoder
    decoder = layers.Flatten()(decoder)
    decoder = layers.Dense(units=inp.shape[1], activation=sigmoid)(decoder)

    autoencoder = keras.Model(inputs=inp, outputs=decoder, name="YildirimOzal_autoencoder")

    if autoencoder_weights is not None:
        autoencoder.load_weights(autoencoder_weights)
    encoder = keras.Model(inputs=inp, outputs=bottleneck, name="YildirimOzal_encoder")

    lstm = layers.LSTM(units=32, return_sequences=True)(bottleneck)
    if include_top:
        lstm = layers.Flatten()(lstm)
        lstm = layers.Dense(units=classes, activation=softmax)(lstm)

    model = keras.Model(inputs=inp, outputs=lstm, name="YildirimOzal_classifier")

    return autoencoder, encoder, model


def CaiWenjuan(include_top=True,
               weights=None,
               input_tensor=None,
               input_shape=None,
               classes=5,
               classifier_activation="softmax"):
    """
    DDNN network presented in:

    References
    ----------
        Cai, Wenjuan, et al. "Accurate detection of atrial fibrillation from 12-lead ECG using deep neural network."
        Computers in biology and medicine 116 (2020): 103378.

    Parameters
    ----------
         include_top: bool, default=True

           Whether to include the fully-connected layer at the top of the network.

        weights: str, default=None

            The path to the weights file to be loaded.

        input_tensor: keras.Tensor, defaults=None

            Optional Keras tensor (i.e. output of `layers.Input()`) to use as input for the model.

        input_shape: Tuple, defaults=None

            If `input_tensor=None`, a tuple that defines the input shape for the model.

        classes: int, defaults=5

            If `include_top=True`, the number of units in the top layer to classify data.

        classifier_activation: str or callable, defaults='softmax'

            The activation function to use on the "top" layer. Ignored unless `include_top=True`. Set
            `classifier_activation=None` to return the logits of the "top" layer.

    Returns
    -------
        A `keras.Model` instance.
   """
    inp = __check_inputs(include_top, weights, input_tensor, input_shape, classes, classifier_activation)

    # Model definition
    dense_layers = [2, 4, 6, 4]
    x1 = layers.Conv1D(filters=8, kernel_size=1, strides=1, padding='same')(inp)
    x2 = layers.Conv1D(filters=16, kernel_size=3, strides=1, padding='same')(inp)
    x3 = layers.Conv1D(filters=24, kernel_size=5, strides=1, padding='same')(inp)
    x = layers.Concatenate(axis=2)([x1, x2, x3])

    for i, n_blocks in enumerate(dense_layers):
        x = __se_module(x, dense_units=128)
        x = __dense_block(x, n_blocks, growth_rate=6, name="conv" + str(i + 1))
        x = __se_module(x, dense_units=128)
        # Last dense block does not have transition block.
        if i < len(dense_layers) - 1:
            x = __transition_block(x, reduction=0.5, name="transition" + str(i + 1))

    x = layers.GlobalAveragePooling1D()(x)

    if include_top:
        x = layers.Flatten()(x)
        x = layers.Dense(units=classes, activation=classifier_activation)(x)

    model = keras.Model(inputs=inp, outputs=x, name="CaiWenjuan")

    if weights is not None:
        model.load_weights(weights)

    return model


def KimMinGu(include_top=True,
             weights=None,
             input_tensor=None,
             input_shape=None,
             classes=5,
             classifier_activation="softmax"):

    # NOTE: kernel sizes are not specified in the original paper.
    # TODO: weights must be individually checked for each model.
    inp = __check_inputs(include_top, None, input_tensor, input_shape, classes, classifier_activation)

    dropout = [0.5, 0.6, 0.6, 0.7, 0.5, 0.7]

    ensemble = []
    for d in dropout:
        model = layers.Conv1D(filters=32, kernel_size=3, padding="same", activation=relu)(inp)
        model = layers.MaxPooling1D(pool_size=2)(model)
        for f in [64, 128, 256, 512]:
            model = layers.Conv1D(filters=f, kernel_size=3, padding="same", activation=relu)(model)
            model = layers.MaxPooling1D(pool_size=2)(model)

        if include_top:
            model = layers.Flatten()(model)
            model = layers.Dense(units=1028, activation=relu)(model)
            model = layers.Dropout(d)(model)
            model = layers.Dense(units=1028, activation=relu)(model)
            model = layers.Dense(units=classes, activation=softmax)(model)

        ensemble.append(keras.Model(inputs=inp, outputs=model))

    return ensemble  # TODO: not finished yet
