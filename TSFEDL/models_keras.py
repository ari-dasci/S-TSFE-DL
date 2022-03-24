import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.activations import relu, sigmoid
from tensorflow.keras.initializers import orthogonal, he_uniform
from tensorflow.python.keras.engine.keras_tensor import KerasTensor

from TSFEDL.blocks_keras import densenet_transition_block, densenet_dense_block, squeeze_excitation_module, \
    RTA_block, spatial_attention_block_ZhangJin, temporal_attention_block_ZhangJin
from TSFEDL.utils import check_inputs, full_convolution


def OhShuLih(include_top=True,
             weights=None,
             input_tensor=None,
             input_shape=None,
             classes=5,
             classifier_activation="softmax"):
    """
    CNN+LSTM model for Arrythmia classification.

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
    `keras.Model`
        A `keras.Model` instance.

    References
    ----------
        `Oh, Shu Lih, et al. "Automated diagnosis of arrhythmia using combination of CNN and LSTM techniques with
        variable length heart beats." Computers in biology and medicine 102 (2018): 278-287.`
    """

    # Initial checks
    inp = check_inputs(include_top, weights, input_tensor, input_shape, classes, classifier_activation)

    # Model Definition
    x = full_convolution(inp, filters=3, kernel_size=20, activation=relu, use_bias=False, strides=1)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = full_convolution(x, filters=6, kernel_size=10, activation=relu, use_bias=False, strides=1)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = full_convolution(x, filters=6, kernel_size=5, activation=relu, use_bias=False, strides=1)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.LSTM(units=20, recurrent_dropout=0.2)(x)

    if include_top:
        x = layers.Dropout(rate=0.2)(x)
        x = layers.Dense(units=20, activation=relu)(x)
        x = layers.Dense(units=10, activation=relu)(x)
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
                 gru_units=(100, 50),
                 return_sequences=False,
                 classes=5,
                 classifier_activation="softmax"):
    """
    CNN+GRU model for electricity forecasting

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

        gru_units: Tuple of length=2, defaults=(100, 50)
            The number of units within the 2 GRU layers.

        return_sequences: bool, defaults=False
            If True, the last GRU layer and Top layer if `include_top=True` returns the whole sequence instead of
            the last value.

        classes: int, defaults=5
            If `include_top=True`, the number of units in the top layer to classify data.

        classifier_activation: str or callable, defaults='softmax'
            The activation function to use on the "top" layer. Ignored unless `include_top=True`. Set
            `classifier_activation=None` to return the logits of the "top" layer.

    Returns
    -------
        model: keras.Model
            A `keras.Model` instance with the Full CNN-LSTM Autoencoder

    References
    ---------
        Sajjad, M., Khan, Z. A., Ullah, A., Hussain, T., Ullah, W., Lee, M. Y., & Baik, S. W. (2020). A novel
        CNN-GRU-based hybrid approach for short-term residential load forecasting. IEEE Access, 8, 143759-143768.
     """

    # Check inputs
    inp = check_inputs(include_top, weights, input_tensor, input_shape, classes, classifier_activation)
    if len(gru_units) != 2:
        raise ValueError("'gru_units' must be a tuple of length 2")

    # Model Definition
    x = layers.Conv1D(filters=16, kernel_size=2, activation=relu)(inp)
    x = layers.Dropout(0.1)(x)
    x = layers.Conv1D(filters=8, kernel_size=2, activation=relu)(x)
    x = layers.Dropout(0.1)(x)
    x = layers.GRU(units=gru_units[0], return_sequences=True)(x)
    x = layers.GRU(units=gru_units[1], return_sequences=return_sequences)(x)

    if include_top:
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

    CNN+LSTM network for arrythmia detection. This model initialy was designed to deal with 2D images. It was adapted
    to 1D time series.

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
        model: keras.Model
            A `keras.Model` instance.

    References
    ----------
        Zheng, Z., Chen, Z., Hu, F., Zhu, J., Tang, Q., & Liang, Y. (2020). An automatic diagnosis of arrhythmias using
        a combination of CNN and LSTM technology. Electronics, 9(1), 121.
    """
    inp = check_inputs(include_top, weights, input_tensor, input_shape, classes, classifier_activation)

    # Model Definition
    x = inp
    for f in [64, 128, 256]:
        x = layers.Conv1D(filters=f, kernel_size=3, strides=1, padding='same', activation=keras.activations.elu)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1D(filters=f, kernel_size=3, strides=1, padding='same', activation=keras.activations.elu)(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(pool_size=2, strides=2)(x)

    x = layers.LSTM(units=256)(x)

    if include_top:
        x = layers.Dense(units=2048, activation=keras.activations.elu)(x)
        x = layers.Dropout(rate=0.5)(x)
        x = layers.Dense(units=2048, activation=keras.activations.elu)(x)
        x = layers.Dense(units=classes, activation=classifier_activation)(x)

    model = keras.Model(inputs=inp, outputs=x, name="ZhengZhenyu")

    # Load weights
    if weights is not None:
        model.load_weights(weights)

    return model


def HouBoroui(weights=None,
              input_tensor=None,
              input_shape=None,
              encoder_units=100):
    """
    Basic LSTM Autoencoder, where the encoded features were introduced in an SVM.

    Parameters
    ----------
        weights: str, default=None
            The path to the weights file to be loaded.

        input_tensor: keras.Tensor, defaults=None
            Optional Keras tensor (i.e. output of `layers.Input()`) to use as input for the model.

        input_shape: Tuple, defaults=None
            If `input_tensor=None`, a tuple that defines the input shape for the model.

        encoder_units: int, defaults=100
            The number of encoding features.

    Returns
    -------
        autoencoder: keras.Model
            A `keras.Model` instance representing the full autoencoder.
        encoder: keras.Model
            A `keras.Model` instance that representes the encoder.

    References
    ----------
        Hou, Borui, et al. "LSTM based auto-encoder model for ECG arrhythmias classification." IEEE Transactions on
        Instrumentation and Measurement (2019).

    """
    inp = check_inputs(True, weights, input_tensor, input_shape, 5, None)  # only check weights and inputs.

    # Model definition
    encoder = layers.LSTM(units=encoder_units)(inp)
    x = layers.RepeatVector(inp.shape[1])(encoder)
    x = layers.LSTM(units=encoder_units, return_sequences=True)(x)
    x = layers.TimeDistributed(layers.Dense(units=1))(x)

    autoencoder = keras.Model(inputs=inp, outputs=x, name="HouBoroui")

    if weights is not None:
        autoencoder.load_weights(weights)

    encoder = keras.Model(inputs=inp, outputs=encoder)

    return autoencoder, encoder


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
        model: keras.Model
            A `keras.Model` instance.
    """
    inp = check_inputs(include_top, weights, input_tensor, input_shape, classes, classifier_activation)

    # Model definition
    x = layers.LSTM(units=64, return_sequences=True)(inp)
    x = layers.LSTM(units=128, return_sequences=True)(x)
    x = layers.Conv1D(filters=64, kernel_size=3, strides=1, activation=relu)(x)
    x = layers.MaxPooling1D(pool_size=2, strides=2)(x)
    x = layers.Conv1D(filters=128, kernel_size=3, strides=1, activation=relu)(x)
    x = layers.MaxPooling1D(pool_size=2, strides=2)(x)

    if include_top:
        x = layers.Flatten()(x)
        x = layers.Dropout(rate=0.1)(x)
        x = layers.Dense(units=2048, activation=relu)(x)
        x = layers.Dense(units=1024, activation=relu)(x)
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
    CNN+LSTM model for arrythmia classification.

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
    model: keras.Model
        A `keras.Model` instance.

    References
    ----------
        Chen, Chen, et al. "Automated arrhythmia classification based on a combination network of CNN and LSTM."
        Biomedical Signal Processing and Control 57 (2020): 101819.
    """
    inp = check_inputs(include_top, weights, input_tensor, input_shape, classes, classifier_activation)

    # Model definition
    x = inp
    for k, f in zip([251, 150, 100, 81, 61, 14],
                    [5, 5, 10, 20, 20, 10]):
        x = layers.Conv1D(filters=f, kernel_size=k, strides=1, activation=relu)(x)
        x = layers.MaxPooling1D(pool_size=2, strides=2)(x)
    x = layers.LSTM(units=32, return_sequences=True)(x)
    x = layers.LSTM(units=64, return_sequences=True)(x)

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
    model: keras.Model
        A `keras.Model` instance.

    References
    ----------
        Kim, Tae-Young, and Sung-Bae Cho. "Predicting residential energy consumption using CNN-LSTM neural networks."
        Energy 182 (2019): 72-81.

    Notes
    -----
        In the original paper, the time-series is windowed. Therefore, a TimeDistributed layer is employed before the
        LSTM to traverse all the generated windows. Here, we do not implement this as it is problem-specific.
        Please note that if you need this layer then you should add it manually.
    """
    inp = check_inputs(include_top, weights, input_tensor, input_shape, classes, classifier_activation)

    # Model definition

    x = layers.Conv1D(filters=64, kernel_size=2, strides=1, activation=relu)(inp)
    x = layers.MaxPooling1D(pool_size=2, strides=1)(x)
    x = layers.Conv1D(filters=64, kernel_size=2, strides=1, activation=relu)(x)
    x = layers.MaxPooling1D(pool_size=2, strides=1)(x)
    x = layers.LSTM(units=64, activation=keras.activations.tanh)(x)

    if include_top:
        x = layers.Dense(units=32, activation=relu)(x)
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
    model: keras.Model
        A `keras.Model` instance.

    References
    ----------
        Geng, Minxing, et al. "Epileptic Seizure Detection Based on Stockwell Transform and Bidirectional Long
        Short-Term Memory." IEEE Transactions on Neural Systems and Rehabilitation Engineering 28.3 (2020): 573-580.
    """
    inp = check_inputs(include_top, weights, input_tensor, input_shape, classes, classifier_activation)

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
    model: keras.Model
        A `keras.Model` instance.

    References
    ----------
        Fu, Jiangmeng, et al. "A hybrid CNN-LSTM model based actuator fault diagnosis for six-rotor UAVs." 2019
        Chinese Control And Decision Conference (CCDC). IEEE, 2019.

    """
    inp = check_inputs(include_top, weights, input_tensor, input_shape, classes, classifier_activation)

    # Model definition
    x = layers.Conv1D(filters=32, kernel_size=1, padding='same', activation=keras.activations.tanh)(inp)
    x = layers.MaxPooling1D(pool_size=2, strides=1)(x)
    x = layers.LSTM(units=256, activation=keras.activations.tanh, dropout=0.3)(x)

    if include_top:
        x = layers.Dense(units=128, activation=keras.activations.tanh)(x)
        x = layers.Dropout(rate=0.3)(x)
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
    model: keras.Model
        A `keras.Model` instance.

    References
    ----------
        Shi, Haotian, et al. "Automated heartbeat classification based on deep neural network with multiple input
        layers." Knowledge-Based Systems 188 (2020): 105036.

    """
    inp = check_inputs(include_top, weights, input_tensor, input_shape, classes, classifier_activation)

    # Model Definition
    if isinstance(inp, KerasTensor) or inp.shape[-1] == 1:
        x1 = layers.Conv1D(filters=32, kernel_size=13, strides=2)(inp)
        x2 = layers.Conv1D(filters=32, kernel_size=13, strides=1)(inp)
        x3 = layers.Conv1D(filters=32, kernel_size=13, strides=2)(inp)
    elif inp.shape[-1] == 3 or inp.shape[-1] == 4:
        x1 = layers.Conv1D(filters=32, kernel_size=13, strides=2)(inp[0])
        x2 = layers.Conv1D(filters=32, kernel_size=13, strides=1)(inp[1])
        x3 = layers.Conv1D(filters=32, kernel_size=13, strides=2)(inp[2])
    else:
        raise ValueError("input must be a vector of tensors of length 1 or 3.")

    x1 = layers.LeakyReLU()(x1)
    x1 = layers.MaxPooling1D(pool_size=2, strides=2)(x1)

    x2 = layers.LeakyReLU()(x2)
    x2 = layers.MaxPooling1D(pool_size=2, strides=2)(x2)

    x3 = layers.LeakyReLU()(x3)
    x3 = layers.MaxPooling1D(pool_size=2, strides=2)(x3)

    x = layers.Concatenate(axis=1)([x1, x2, x3])
    x = layers.LSTM(units=32, return_sequences=True)(x)

    if include_top:
        x = layers.Flatten()(x)
        x = layers.Dense(units=518, activation=relu)(x)
        x = layers.Dense(units=88, activation=relu)(x)
        if not isinstance(inp, KerasTensor) and inp.shape[-1] == 4:
            x = layers.Concatenate([inp[3], x])
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
                 classifier_activation="softmax",
                 units_in_dense=(512, 256)):
    """
     CNN model, employed for 2-D images of ECG data. This model is adapted for 1-D time series

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

        units_in_dense: Tuple[int, int], defaults=(512, 256)
            If `include_top=True`, the number of hidden units of the two dense layers before the softmax. Note:
            The default values are arbitrarely set. The number of units in these layers are not specified in the
            original paper.

    Returns
    -------
    model: keras.Model
        A `keras.Model` instance.

    References
    -------
        Huang, Mei-Ling, and Yan-Sheng Wu. "Classification of atrial fibrillation and normal sinus rhythm based on
        convolutional neural network." Biomedical Engineering Letters (2020): 1-11.
     """
    inp = check_inputs(include_top, weights, input_tensor, input_shape, classes, classifier_activation)

    # Model definition (It is the first configuration of paper's table 4)
    x = inp
    for p, f, k, s in zip([1, 3],
                       [48, 256],
                       [15, 7],
                       [6, 2]):
        x = layers.ZeroPadding1D(padding=p)(x)
        x = layers.Conv1D(filters=f, kernel_size=k, strides=s, activation=relu)(x)
        x = layers.MaxPooling1D(pool_size=2, strides=1)(x)

    if include_top:
        x = layers.Flatten()(x)
        x = layers.Dropout(rate=0.5)(x)
        x = layers.Dense(units=units_in_dense[0], activation=relu)(x)
        x = layers.Dense(units=units_in_dense[1], activation=relu)(x)
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
    model: keras.Model
        A `keras.Model` instance.

    References
    ----------
        Lih, Oh Shu, et al. "Comprehensive electrocardiographic diagnosis based on deep learning." Artificial
        Intelligence in Medicine 103 (2020): 101789.
    """
    inp = check_inputs(include_top, weights, input_tensor, input_shape, classes, classifier_activation)

    # Model definition
    x = inp
    for filters, k_size in zip([3, 6, 6, 6, 6],
                               [20, 10, 5, 5, 10]):
        x = layers.Conv1D(filters=filters, activation=relu, kernel_size=k_size, strides=1, use_bias=False)(x)
        x = layers.MaxPooling1D(pool_size=2, strides=2)(x)

    x = layers.LSTM(units=10)(x)

    if include_top:
        # x = layers.Flatten()(x)
        x = layers.Dense(units=8)(x)
        x = layers.Dropout(rate=0.5)(x)
        x = layers.Dense(units=8)(x)
        x = layers.Dropout(rate=0.5)(x)
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
    model: keras.Model
        A `keras.Model` instance.

    References
    ----------
        Gao, Junli, et al. "An Effective LSTM Recurrent Network to Detect Arrhythmia on Imbalanced ECG Dataset."
        Journal of healthcare engineering 2019 (2019).
    """
    inp = check_inputs(include_top, weights, input_tensor, input_shape, classes, classifier_activation)

    # Model definition
    x = layers.LSTM(units=64, dropout=0.3)(inp)

    if include_top:
        x = layers.Dense(units=32)(x)
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
    model: keras.Model
        A `keras.Model` instance.

    References
    ----------
        Wei, Xiaoyan, et al. "Early prediction of epileptic seizures using a long-term recurrent convolutional
        network." Journal of neuroscience methods 327 (2019): 108395.
    """
    inp = check_inputs(include_top, weights, input_tensor, input_shape, classes, classifier_activation)

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
    x = layers.BatchNormalization()(x)
    x = layers.LSTM(units=512)(x)

    if include_top:
        x = layers.Dense(units=1024)(x)
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
                 classifier_activation="softmax",
                 return_sequences=False):
    """
    CNN+LSTM

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

        return_sequences: bool, defaults=False
            If True, the last LSTM layer will return the whole sequence instead of the last value.

    Returns
    -------
    model: keras.Model
        A `keras.Model` instance.

    References
    ----------
        Kong, Zhengmin, et al. "Convolution and Long Short-Term Memory Hybrid Deep Neural Networks for Remaining
        Useful Life Prognostics." Applied Sciences 9.19 (2019): 4156.
    """
    inp = check_inputs(include_top, weights, input_tensor, input_shape, classes, classifier_activation)

    # Model definition
    # Note: Number of filters, kernel_sizes, strides, and nº of units in LSTM layers are not specified in the paper.
    x = layers.Conv1D(filters=32, activation=relu, kernel_size=5, strides=1)(inp)
    x = layers.MaxPooling1D(pool_size=2, strides=2)(x)
    x = layers.LSTM(64, return_sequences=True)(x)
    x = layers.LSTM(64, return_sequences=return_sequences)(x)

    if include_top:
        if return_sequences:
            x = layers.Flatten()(x)
        x = layers.Dense(units=50)(x)
        x = layers.Dense(units=50)(x)
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
        autoenconder: keras.Model
            A `keras.Model` instance with the autoencoder.

        encoder: keras.Model
            A `keras.Model` instance with only the encoder part

        model: keras.Model
            A `keras.Model` instance representing the classification model which uses the encoder part.

    References
    ----------
        Yildirim, Ozal, et al. "A new approach for arrhythmia classification using deep coded features and LSTM
        networks." Computer methods and programs in biomedicine 176 (2019): 121-133.
     """
    inp = check_inputs(include_top, None, input_tensor, input_shape, classes, classifier_activation)
    if autoencoder_weights is not None and not tf.io.gfile.exists(autoencoder_weights):
        raise ValueError("'autoencoder_weights' path does not exists: ", autoencoder_weights)
    if lstm_weights is not None and not tf.io.gfile.exists(lstm_weights):
        raise ValueError("'lstm_weights' path does not exists: ", lstm_weights)

    # Model definition
    e = layers.Conv1D(filters=16, kernel_size=5, padding='same', strides=1, activation=relu)(inp)
    e = layers.MaxPooling1D(pool_size=2)(e)
    e = layers.Conv1D(filters=64, kernel_size=5, padding='same', strides=1, activation=relu)(e)
    e = layers.BatchNormalization()(e)
    e = layers.MaxPooling1D(pool_size=2)(e)
    e = layers.Conv1D(filters=32, kernel_size=3, padding='same', strides=1, activation=relu)(e)
    e = layers.Conv1D(filters=1, kernel_size=3, padding='same', strides=1, activation=relu)(e)

    bottleneck = layers.MaxPooling1D(pool_size=2)(e)
    decoder = layers.Conv1D(filters=1, kernel_size=3, padding='same', strides=1, activation=relu)(bottleneck)
    decoder = layers.Conv1D(filters=32, kernel_size=3, padding='same', strides=1, activation=relu)(decoder)
    decoder = layers.UpSampling1D(size=2)(decoder)
    decoder = layers.Conv1D(filters=64, kernel_size=5, padding='same', strides=1, activation=relu)(decoder)
    decoder = layers.UpSampling1D(size=2)(decoder)
    decoder = layers.Conv1D(filters=16, kernel_size=5, padding='same', strides=1, activation=relu)(decoder)
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
        lstm = layers.Dense(units=128, activation=relu)(lstm)
        lstm = layers.Dropout(rate=0.1)(lstm)
        lstm = layers.Dense(units=classes, activation=classifier_activation)(lstm)

    model = keras.Model(inputs=inp, outputs=lstm, name="YildirimOzal_classifier")
    if lstm_weights is not None:
        model.load_weights(lstm_weights)

    return autoencoder, encoder, model


def CaiWenjuan(include_top=True,
               weights=None,
               input_tensor=None,
               input_shape=None,
               classes=5,
               classifier_activation="softmax",
               reduction_ratio=0.6):
    """
    DDNN network presented in:

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

        reduction_ratio: float, defaults=0.5
            The reduction ratio in the transition block.

    Returns
    -------
    `keras.Model`
        A `keras.Model` instance.

    References
    ----------
        `Cai, Wenjuan, et al. "Accurate detection of atrial fibrillation from 12-lead ECG using deep neural network."
        Computers in biology and medicine 116 (2020): 103378.`
   """
    inp = check_inputs(include_top, weights, input_tensor, input_shape, classes, classifier_activation)

    # Model definition
    dense_layers = [2, 4, 6, 4]
    x1 = layers.Conv1D(filters=8, kernel_size=1, strides=1, padding='same')(inp)
    x2 = layers.Conv1D(filters=16, kernel_size=3, strides=1, padding='same')(inp)
    x3 = layers.Conv1D(filters=24, kernel_size=5, strides=1, padding='same')(inp)
    x = layers.Concatenate(axis=2)([x1, x2, x3])

    for i, n_blocks in enumerate(dense_layers):  # dense-block construction
        x = squeeze_excitation_module(x, dense_units=32)
        x = densenet_dense_block(x, n_blocks, growth_rate=6, name="conv" + str(i + 1))
        x = squeeze_excitation_module(x, dense_units=32)
        # Last dense block does not have transition block.
        if i < len(dense_layers) - 1:
            x = densenet_transition_block(x, reduction=reduction_ratio, name="transition" + str(i + 1))

    x = layers.GlobalAveragePooling1D()(x)

    if include_top:
        x = layers.Flatten()(x)
        x = layers.Dense(units=classes, activation=classifier_activation)(x)

    model = keras.Model(inputs=inp, outputs=x, name="CaiWenjuan")

    if weights is not None:
        model.load_weights(weights)

    return model


def KimMinGu(include_top=True,
             weights=None,  # Here, 'weights' is a vector of paths for each model in the ensemble
             input_tensor=None,
             input_shape=None,
             classes=5,
             classifier_activation="softmax"):
    """
    CNN ensemble model. The same model is employed n-times in an ensemble. Then, the same model is employed retraining
    using the features extracted in the ensemble.

    Parameters
    ----------
        include_top: bool, default=True
          Whether to include the fully-connected layer at the top of the network.

        weights: array(str), default=None
            An array with the paths of the weights file for each model of the ensemble.

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
        ensemble: list[keras.Model]
            A list with six `keras.Model` instances. One for each model of the ensemble.

    References
    ----------
        Kim, M. G., Choi, C., & Pan, S. B. (2020). Ensemble Networks for User Recognition in Various Situations Based on
        Electrocardiogram. IEEE Access, 8, 36527-36535.
   """

    # NOTE: kernel sizes are not specified in the original paper.
    # Initial checks
    inp = check_inputs(include_top, None, input_tensor, input_shape, classes, classifier_activation)
    if weights is not None:
        for w in weights:
            if not tf.io.gfile.exists(w):
                raise ValueError("'weights' path does not exists: ", weights)

    # Begin model definition
    dropout = [0.5, 0.6, 0.6, 0.7, 0.5, 0.7]

    ensemble = []
    for i, d in enumerate(dropout):
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
            model = layers.Dense(units=classes, activation=classifier_activation)(model)

        m = keras.Model(inputs=inp, outputs=model)
        if weights is not None and weights[i] is not None:
            m.load_weights(weights[i])
        ensemble.append(m)

    return ensemble


def HtetMyetLynn(include_top=True,
                 weights=None,
                 input_tensor=None,
                 input_shape=None,
                 classes=5,
                 classifier_activation="softmax",
                 use_rnn="gru",
                 rnn_units=40):
    """
    Hybrid CNN+Bidirectional RNN

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

        use_rnn: str, defaults='gru'
            Whether to user a Bi-direccional RNN after the CNN. Options are `'gru'` for a GRU, `'lstm'` for an LSTM or
            `None` to not use the RNN.

        rnn_units: int, defatuls=40
            If `use_rnn` is not `None`, the number of units in the chosen biderectional RNN.

    Returns
    -------
    model: `keras.Model`
        A `keras.Model` instance.

    References
    ----------
        Lynn, H. M., Pan, S. B., & Kim, P. (2019). A deep bidirectional GRU network model for biometric
        electrocardiogram classification based on recurrent neural networks. IEEE Access, 7, 145395-145405.
    """
    # Check inputs
    inp = check_inputs(include_top, weights, input_tensor, input_shape, classes, classifier_activation)

    # Model definition
    x = inp
    for filt, ker, stride in zip([30, 30, 60, 60],
                                 [5, 2, 5, 2],
                                 [2, 2, 2, 2]):
        x = layers.Conv1D(filters=filt, kernel_size=ker, padding='same')(x)
        x = layers.MaxPooling1D(pool_size=stride)(x)

    # Only use the CNN, or add a Bi-directional GRU or LSTM after the CNN.
    if use_rnn is None:
        if include_top:
            x = layers.Dense(units=40, activation=sigmoid)(x)
            x = layers.Dense(units=classes, activation=classifier_activation)(x)
    elif use_rnn.lower() == 'lstm':
        # Note: Units of RNN are not specified in the original paper
        x = layers.Bidirectional(layers.LSTM(units=rnn_units, dropout=0.2))(x)
    elif use_rnn.lower() == 'gru':
        x = layers.Bidirectional(layers.GRU(units=rnn_units, dropout=0.2))(x)
    else:
        raise ValueError("'use_rnn' parameter should one of: None, lstm or gru. Given: ", use_rnn)

    # Adds classification layer only in the case of use bidrectional layers.
    if use_rnn is not None:
        if include_top:
            x = layers.Dense(units=classes, activation=classifier_activation)(x)

    model = keras.Model(inputs=inp, outputs=x, name="HtetMyetLynn")

    if weights is not None:
        model.load_weights(weights)

    return model


def ZhangJin(include_top=True,
             weights=None,
             input_tensor=None,
             input_shape=None,
             classes=5,
             classifier_activation="softmax",
             decrease_ratio=2):
    """
    A CNN+Bi-Directional GRU with a spatio-temporal attention mechanism.

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

        decrease_ratio: float, defaults=2
            The decrease ratio of the model.

    Returns
    -------
    model: `keras.Model`
        A `keras.Model` instance.

    References
    ----------
        Zhang, J., Liu, A., Gao, M., Chen, X., Zhang, X., & Chen, X. (2020). ECG-based multi-class arrhythmia detection
        using spatio-temporal attention-based convolutional recurrent neural network.
        Artificial Intelligence in Medicine, 106, 101856.

    Notes
    -----
        The time dimension must be at least 1000 as there is a huge reduction on each convolutional block.
    """

    # Check inputs
    inp = check_inputs(include_top, weights, input_tensor, input_shape, classes, classifier_activation)
    x = inp

    # Model definition
    for conv_layers, filters in zip([2, 2, 3, 3, 3],
                                    [64, 128, 256, 256, 256]):  # 5-block layers
        for i in range(conv_layers):
            x = layers.Conv1D(filters=filters, kernel_size=3, padding="same")(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation(activation=relu)(x)
        x = layers.MaxPooling1D(pool_size=3)(x)
        x = layers.Dropout(rate=0.2)(x)

        # Adds attention module after each convolutional block
        x_spatial = spatial_attention_block_ZhangJin(decrease_ratio, x)
        x = layers.multiply([x_spatial, x])
        x_temporal = temporal_attention_block_ZhangJin(x)
        x = layers.multiply([x_temporal, x])

    x = layers.Bidirectional(layers.GRU(units=12, return_sequences=True))(x)
    x = layers.Dropout(rate=0.2)(x)
    if include_top:
        x = layers.GlobalMaxPool1D()(x)
        x = layers.Dense(units=classes, activation=classifier_activation)(x)

    model = keras.Model(inputs=inp, outputs=x, name="ZhangJin")

    if weights is not None:
        model.load_weights(weights)

    return model


def YaoQihang(include_top=True,
              weights=None,
              input_tensor=None,
              input_shape=None,
              classes=5,
              classifier_activation="softmax"):
    """
    Attention-based time-incremental convolutional neural network (ATI-CNN)

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
        model: `keras.Model`
            A `keras.Model` instance.

    References
    ----------
        Yao, Q., Wang, R., Fan, X., Liu, J., & Li, Y. (2020). Multi-class Arrhythmia detection from 12-lead varied-length
        ECG using Attention-based Time-Incremental Convolutional Neural Network. Information Fusion, 53, 174-182.
    """
    # Check inputs
    inp = check_inputs(include_top, weights, input_tensor, input_shape, classes, classifier_activation)
    x = inp

    # Model definition
    # Convolutional layers (spatial):
    for conv_layers, filters in zip([2, 2, 3, 3, 3],
                                    [64, 128, 256, 256, 256]):  # 5-block layers
        for i in range(conv_layers):
            x = layers.Conv1D(filters=filters, kernel_size=3, padding="same", kernel_initializer=he_uniform)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation(activation=relu)(x)
        x = layers.MaxPooling1D(pool_size=3, strides=3)(x)

    # Temporal layers (2 LSTM layers):
    x = layers.LSTM(units=32, return_sequences=True, dropout=0.2, kernel_initializer=orthogonal)(x)
    x = layers.LSTM(units=32, return_sequences=True, dropout=0.2, kernel_initializer=orthogonal)(x)

    # Attention module:
    if include_top:
        x1 = layers.Dense(units=32, activation=keras.activations.tanh, kernel_initializer=he_uniform)(x)
        x1 = layers.Dense(units=classes, activation=classifier_activation, kernel_initializer=he_uniform)(x1)
        x = tf.reduce_mean(x1, axis=1)

    model = keras.Model(inputs=inp, outputs=x, name="YaoQihang")

    if weights is not None:
        model.load_weights(weights)

    return model


def YiboGao(include_top=True,
            weights=None,
            input_tensor=None,
            input_shape=None,
            classes=5,
            classifier_activation="softmax",
            return_loss=False):
    """
    CNN using RTA blocks for end-to-end attrial fibrilation detection.

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

        return_loss: bool, defaults=False
            Whether to return the custom loss function (en_loss) employed for training this model.

    Returns
    -------
        model: `keras.Model`
            A `keras.Model` instance.

    References
    ----------
        Gao, Y., Wang, H., & Liu, Z. (2021). An end-to-end atrial fibrillation detection by a novel residual-based
        temporal attention convolutional neural network with exponential nonlinearity loss.
        Knowledge-Based Systems, 212, 106589.

    Notes
    -----
        Code adapted from the original implementation available at:
            https://github.com/o00O00o/RTA-CNN
    """

    def en_loss(y_true, y_pred):  # Custom loss function

        epsilon = 1.e-7
        gamma = float(0.3)

        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        pos_pred = tf.pow(-tf.math.log(y_pred), gamma)
        nag_pred = tf.pow(-tf.math.log(1 - y_pred), gamma)
        y_t = tf.multiply(y_true, pos_pred) + tf.multiply(1 - y_true, nag_pred)
        loss = tf.reduce_mean(y_t)

        return loss

    # Model definition
    inp = check_inputs(include_top, weights, input_tensor, input_shape, classes, classifier_activation)
    x = inp

    for fil, ker, pool in zip([16, 32, 64, 64],
                              [32, 16, 9, 9],
                              [4, 4, 2, 2]):
        x = RTA_block(x, fil, ker)
        x = layers.MaxPooling1D(pool)(x)

    x = layers.Dropout(0.6)(x)

    x = RTA_block(x, 128, 3)
    x = layers.MaxPooling1D(2)(x)
    x = RTA_block(x, 128, 3)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.6)(x)

    if include_top:
        x = layers.Flatten()(x)
        x = layers.Dropout(rate=0.7)(x)
        x = layers.Dense(units=100, activation=relu)(x)
        x = layers.Dropout(rate=0.7)(x)
        x = layers.Dense(classes, activation=classifier_activation)(x)

    model = keras.Model(inputs=inp, outputs=x, name="YiboGao")

    if weights is not None:
        model.load_weights(weights)

    # Return the model and its custom loss function
    if return_loss:
        return model, en_loss
    else:
        return model

# Original CNN-LSTM model
def HongTan(include_top=True,
                weights=None,
                input_tensor=None,
                input_shape=None,
                classes=5,
                classifier_activation="softmax"):
    """

    CNN+LSTM network for arrythmia detection. Application of stacked convolutional and long short-term memory network
    for accurate identification of CAD ECG signals.

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
        model: keras.Model
            A `keras.Model` instance.

    References
    ----------
        TAN, Jen Hong, et al. Application of stacked convolutional and long short-term memory network for accurate
        identification of CAD ECG signals. Computers in biology and medicine, 2018, vol. 94, p. 19-26.
    """
    inp = check_inputs(include_top, weights, input_tensor, input_shape, classes, classifier_activation)

    # Model Definition
    x = inp
    x = layers.Conv1D(filters=40, kernel_size=5, strides=1, padding='same', activation=keras.activations.elu)(x)
    x = layers.MaxPooling1D(pool_size=2, strides=2)(x)
    x = layers.Conv1D(filters=32, kernel_size=3, strides=1, padding='same', activation=keras.activations.elu)(x)
    x = layers.MaxPooling1D(pool_size=2, strides=2)(x)

    x = layers.LSTM(units=32, dropout=0.5, recurrent_dropout=0.25, return_sequences=True)(x)
    x = layers.LSTM(units=16, recurrent_dropout=0.25, return_sequences=True)(x)
    x = layers.LSTM(units=4)(x)

    if include_top:
        x = layers.Dense(units=classes, activation=classifier_activation)(x)

    model = keras.Model(inputs=inp, outputs=x, name="HongTan")

    # Load weights
    if weights is not None:
        model.load_weights(weights)

    return model
