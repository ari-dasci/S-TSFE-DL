import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, models
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


def OhShuLih(inputs: keras.layers.Layer, include_top: bool = True):
    """
    CNN+LSTM model.

    References
    ----------

        Oh, Shu Lih, et al. "Automated diagnosis of arrhythmia using combination of CNN and LSTM techniques with
        variable length heart beats." Computers in biology and medicine 102 (2018): 278-287.

     Parameters
    ----------
        inputs: keras.layers.Layer

            The input layer for this model.

        include_top: bool, default=True

            Include a Flatten layer as the last layer of the model. Default is True.

    Returns
    -------

        A set of layers as a KerasTensor that conforms the model. Note that the model itself must be defined afterwards.
    """

    x = layers.Conv1D(filters=64, kernel_size=3, activation=relu)(inputs)
    x = layers.MaxPooling1D(pool_size=4)(x)
    x = layers.Dropout(rate=0.3)(x)
    x = layers.Conv1D(filters=16, kernel_size=3, activation=relu)(x)
    x = layers.MaxPooling1D(pool_size=4)(x)
    x = layers.Dropout(rate=0.3)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LSTM(units=16)(x)

    if include_top:
        x = layers.Flatten()(x)

    return x


def KhanZulfiqar(inputs: keras.layers.Layer, include_top: bool = True):
    """
    References
    ---------
        Khan, Zulfiqar Ahmad, et al. "Towards Efficient Electricity Forecasting in Residential and Commercial
        Buildings: A Novel Hybrid CNN with a LSTM-AE based Framework." Sensors 20.5 (2020): 1399.

    Parameters
    ----------
        inputs: keras.layers.Layer

            The input layer for this model.

        include_top: bool, default=True

            Include a Flatten layer as the last layer of the model. Default is True.

    Returns
    -------

        A set of layers as a KerasTensor that conforms the model. Note that the model itself must be defined afterwards.

     """

    x = layers.Conv1D(filters=8, kernel_size=1, padding='same', activation=relu)(inputs)
    x = layers.Dropout(0.1)(x)
    x = layers.Conv1D(filters=16, kernel_size=1, padding='same', activation=relu)(x)
    x = layers.Dropout(0.1)(x)
    x = layers.LSTM(50, activation=relu, return_sequences=True)(x)
    x = layers.LSTM(50, activation=relu, return_sequences=True)(x)

    if include_top:
        x = layers.Flatten()(x)

    return x


def ZhengZhenyu(inputs: keras.layers.Layer, include_top: bool = True):
    """
    Parameters
    ----------
        inputs: keras.layers.Layer

            The input layer for this model.

        include_top: bool, default=True

            Include a Flatten layer as the last layer of the model. Default is True.

    Returns
    -------

        A set of layers as a KerasTensor that conforms the model. Note that the model itself must be defined afterwards.

    """
    x = inputs
    for f in [64, 128, 256]:
        x = layers.Conv1D(filters=f, kernel_size=3, strides=1, padding='same', activation='relu')(x)
        x = layers.Conv1D(filters=f, kernel_size=3, strides=1, padding='same', activation='relu')(x)
        x = layers.MaxPooling1D(4, strides=1)(x)

    x = layers.LSTM(16, return_sequences=True)(x)

    if include_top:
        x = layers.Flatten()(x)

    return x


def HouBoroui(inputs: keras.layers.Layer, include_top: bool = True):
    """
    References
    ----------
        Hou, Borui, et al. "LSTM based auto-encoder model for ECG arrhythmias classification." IEEE Transactions on
        Instrumentation and Measurement (2019).

    Parameters
    ----------
        inputs: keras.layers.Layer

            The input layer for this model.

        include_top: bool, default=True

            Include a Flatten layer as the last layer of the model. Default is True.

    Returns
    -------

        A set of layers as a KerasTensor that conforms the model. Note that the model itself must be defined afterwards.

    """
    x = layers.LSTM(units=146, return_sequences=True)(inputs)
    x = layers.LSTM(units=31, return_sequences=True)(x)
    if include_top:
        x = layers.Flatten()(x)

    return x


def WangKejun(inputs: keras.layers.Layer, include_top: bool = True):
    """
    References
    ----------
        Wang, Kejun, Xiaoxia Qi, and Hongda Liu. "Photovoltaic power forecasting based LSTM-Convolutional Network."
        Energy 189 (2019): 116225.

    Parameters
    ----------
        inputs: keras.layers.Layer

            The input layer for this model.

        include_top: bool, default=True

            Include a Flatten layer as the last layer of the model. Default is True.

    Returns
    -------

        A set of layers as a KerasTensor that conforms the model. Note that the model itself must be defined afterwards.

       """
    x = layers.LSTM(64, return_sequences=True)(inputs)
    x = layers.LSTM(128, return_sequences=True)(x)
    x = layers.Conv1D(filters=64, kernel_size=3, strides=1)(x)
    x = layers.MaxPooling1D(2, strides=2)(x)
    x = layers.Conv1D(filters=150, kernel_size=2, strides=1)(x)
    x = layers.MaxPooling1D(2, strides=2)(x)
    x = layers.Dropout(0.1)(x)

    if include_top:
        x = layers.Flatten()(x)

    return x


def ChenChen(inputs: keras.layers.Layer, include_top: bool=True):
    """
    References
    ----------
        Chen, Chen, et al. "Automated arrhythmia classification based on a combination network of CNN and LSTM."
        Biomedical Signal Processing and Control 57 (2020): 101819.

    Parameters
    ----------
        inputs: keras.layers.Layer

            The input layer for this model.

        include_top: bool, default=True

            Include a Flatten layer as the last layer of the model. Default is True.

    Returns
    -------

        A set of layers as a KerasTensor that conforms the model. Note that the model itself must be defined afterwards.


    """

    x = inputs
    for f in [251, 150, 100, 81, 61, 14]:
        x = layers.Conv1D(filters=f, kernel_size=2, strides=1)(x)
        x = layers.MaxPooling1D(pool_size=2, strides=1)(x)
    x = layers.LSTM(units=64, return_sequences=True)(x)
    x = layers.LSTM(units=32, return_sequences=True)(x)

    if include_top:
        x = layers.Flatten()(x)

    return x


def KimTaeYoung(inputs: keras.layers.Layer, include_top: bool=True):
    """
    References
    ----------
        Kim, Tae-Young, and Sung-Bae Cho. "Predicting residential energy consumption using CNN-LSTM neural networks."
        Energy 182 (2019): 72-81.

    Parameters
    ----------
        inputs: keras.layers.Layer

            The input layer for this model.

        include_top: bool, default=True

            Include a Flatten layer as the last layer of the model. Default is True.

    Returns
    -------

        A set of layers as a KerasTensor that conforms the model. Note that the model itself must be defined afterwards.
    """
    x = inputs
    for f in [64, 150]:
        x = layers.Conv1D(filters=f, kernel_size=2, strides=1)(x)
        x = layers.MaxPooling1D(pool_size=2, strides=1)(x)
    x = layers.LSTM(units=64, activation=keras.activations.tanh, return_sequences=True)(x)

    if include_top:
        x = layers.Flatten()(x)

    return x


def GenMinxing(inputs: keras.layers.Layer, include_top: bool = True):
    """
    References
    ----------
        Geng, Minxing, et al. "Epileptic Seizure Detection Based on Stockwell Transform and Bidirectional Long
        Short-Term Memory." IEEE Transactions on Neural Systems and Rehabilitation Engineering 28.3 (2020): 573-580.

    Parameters
    ----------
        inputs: keras.layers.Layer

            The input layer for this model.

        include_top: bool, default=True

            Include a Flatten layer as the last layer of the model. Default is True.

    Returns
    -------

        A set of layers as a KerasTensor that conforms the model. Note that the model itself must be defined afterwards.
    """
    x = layers.Bidirectional(layers.LSTM(units=40, return_sequences=True))(inputs)
    if include_top:
        x = layers.Flatten()(x)
    return x


def FuJiangmeng(inputs: keras.layers.Layer, include_top: bool = True):
    """
    References
    ----------
        Fu, Jiangmeng, et al. "A hybrid CNN-LSTM model based actuator fault diagnosis for six-rotor UAVs." 2019
        Chinese Control And Decision Conference (CCDC). IEEE, 2019.

    Parameters
    ----------
        inputs: keras.layers.Layer

            The input layer for this model.

        include_top: bool, default=True

            Include a Flatten layer as the last layer of the model. Default is True.

    Returns
    -------

        A set of layers as a KerasTensor that conforms the model. Note that the model itself must be defined afterwards.
    """

    x = layers.Conv1D(filters=32, kernel_size=1, padding='same', activation=relu)(inputs)
    x = layers.MaxPooling1D(pool_size=2, strides=1)(x)
    x = layers.LSTM(units=256, activation=relu, return_sequences=True)(x)
    if include_top:
        x = layers.Flatten()(x)
    return x


def ShiHaotian(inputs, include_top: bool = True):
    """
    References
    ----------
        Shi, Haotian, et al. "Automated heartbeat classification based on deep neural network with multiple input
        layers." Knowledge-Based Systems 188 (2020): 105036.

    Parameters
    ----------
        inputs: keras.layers.Layer

            The input layer for this model.

        include_top: bool, default=True

            Include a Flatten layer as the last layer of the model. Default is True.

    Returns
    -------

        A set of layers as a KerasTensor that conforms the model. Note that the model itself must be defined afterwards.
    """

    x1 = layers.Conv1D(filters=32, kernel_size=13, strides=2, activation=relu)(inputs)
    x1 = layers.MaxPooling1D(pool_size=2, strides=2)(x1)

    x2 = layers.Conv1D(filters=32, kernel_size=13, strides=1, activation=relu)(inputs)
    x2 = layers.MaxPooling1D(pool_size=2, strides=2)(x2)

    x3 = layers.Conv1D(filters=32, kernel_size=13, strides=2, activation=relu)(inputs)
    x3 = layers.MaxPooling1D(pool_size=2, strides=2)(x3)

    x = layers.Concatenate(axis=1)([x1, x2, x3])
    x = layers.LSTM(units=32, return_sequences=True)(x)

    if include_top:
        x = layers.Flatten()(x)
    return x


def HuangMeiLing(inputs, include_top: bool = True):
    """
     CNN model, employed for 2-D images of ECG data. This model is adapted for 1-D time series

     References
     -------
         Huang, Mei-Ling, and Yan-Sheng Wu. "Classification of atrial fibrillation and normal sinus rhythm based on
         convolutional neural network." Biomedical Engineering Letters (2020): 1-11.


    Parameters
    ----------
        inputs: keras.layers.Layer

            The input layer for this model.

        include_top: bool, default=True

            Include a Flatten layer as the last layer of the model. Default is True.

    Returns
    -------

        A set of layers as a KerasTensor that conforms the model. Note that the model itself must be defined afterwards.
     """

    x = inputs
    for f, k, s in zip([48, 256],
                       [15, 13],
                       [6, 1]):
        x = layers.Conv1D(filters=f, activation=relu, kernel_size=k, strides=s)(x)
        x = layers.MaxPooling1D(pool_size=2, strides=1)(x)

    if include_top:
        x = layers.Flatten()(x)

    return x


def LihOhShu(inputs, include_top: bool = True):
    """
    CNN+LSTM model

    References
    ----------
        Lih, Oh Shu, et al. "Comprehensive electrocardiographic diagnosis based on deep learning." Artificial
        Intelligence in Medicine 103 (2020): 101789.

    Parameters
    ----------
        inputs: keras.layers.Layer

            The input layer for this model.

        include_top: bool, default=True

            Include a Flatten layer as the last layer of the model. Default is True.

    Returns
    -------

        A set of layers as a KerasTensor that conforms the model. Note that the model itself must be defined afterwards.

    """

    x = inputs
    for filters, k_size in zip([3, 6, 6, 6, 6],
                               [20, 10, 5, 5, 10]):
        x = layers.Conv1D(filters=filters, activation=relu, kernel_size=k_size, strides=1)(x)
        x = layers.MaxPooling1D(pool_size=2, strides=1)(x)

    x = layers.LSTM(units=10, return_sequences=True)(x)

    if include_top:
        x = layers.Flatten()(x)

    return x


def GaoJunLi(inputs, include_top: bool = True):
    """
    LSTM network

    References
    ----------
        Gao, Junli, et al. "An Effective LSTM Recurrent Network to Detect Arrhythmia on Imbalanced ECG Dataset."
        Journal of healthcare engineering 2019 (2019).

    Parameters
    ----------
        inputs: keras.layers.Layer

            The input layer for this model.

        include_top: bool, default=True

            Include a Flatten layer as the last layer of the model. Default is True.

    Returns
    -------

        A set of layers as a KerasTensor that conforms the model. Note that the model itself must be defined afterwards.
    """
    x = layers.LSTM(units=64, return_sequences=True)(inputs)

    if include_top:
        x = layers.Flatten()(x)

    return x


def WeiXiaoyan(inputs, include_top: bool = True):
    """
     CNN+LSTM model employed for 2-D images of ECG data. This model is adapted for 1-D time series.


     References
     ----------

         Wei, Xiaoyan, et al. "Early prediction of epileptic seizures using a long-term recurrent convolutional
         network." Journal of neuroscience methods 327 (2019): 108395.

    Parameters
    ----------
        inputs: keras.layers.Layer

            The input layer for this model.

        include_top: bool, default=True

            Include a Flatten layer as the last layer of the model. Default is True.

    Returns
    -------

        A set of layers as a KerasTensor that conforms the model. Note that the model itself must be defined afterwards.

     """

    x = inputs
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

    return x


def KongZhengmin(inputs, include_top: bool = True):
    """
    CNN+LSTM

    References
    ----------
        Kong, Zhengmin, et al. "Convolution and Long Short-Term Memory Hybrid Deep Neural Networks for Remaining
        Useful Life Prognostics." Applied Sciences 9.19 (2019): 4156.

    Parameters
    ----------
        inputs: keras.layers.Layer

            The input layer for this model.

        include_top: bool, default=True

            Include a Flatten layer as the last layer of the model. Default is True.

    Returns
    -------

        A set of layers as a KerasTensor that conforms the model. Note that the model itself must be defined afterwards.

    """
    x = layers.Conv1D(filters=32, activation=relu, kernel_size=5, strides=1)(inputs)
    x = layers.MaxPooling1D(pool_size=2, strides=2)(x)
    x = layers.LSTM(64, return_sequences=True)(x)
    x = layers.LSTM(64, return_sequences=True)(x)

    if include_top:
        x = layers.Flatten()(x)

    return x


def YildirimOzal(inputs, include_top: bool = True):
    """CAE-LSTM

     References
     ----------
         Yildirim, Ozal, et al. "A new approach for arrhythmia classification using deep coded features and LSTM
         networks." Computer methods and programs in biomedicine 176 (2019): 121-133.

     Parameters
     ----------
     include_top: bool, default=True
                      Include a Flatten layer as the last layer of the model. Default is True.

     Attributes
     ----------
     include_top: bool

         Boolean that indicates whether the last layer is a Flatten layer or not.

     encoder: keras.Model

         The encoder part of the AutoEnconder

     decoder: keras.Model

         The decoder part of the AutoEnconder.

     lstm: keras.Model

         The LSTM part of this model.

     Examples
     --------
     This method is composed of two parts: An Autoencoder and an LSTM that classifies the encoded data from the encoder
     part. Therefore, it is necessary to firstly train the model:

     >>> inputs = keras.Input((200, 1))
     >>> yildirim = YildirimOzal()
     >>> encoder = yildirim.encoder(inputs)
     >>> x = yildirim.decoder(encoder)
     >>> autoencoder = keras.Model(inputs=inputs, outputs=x)

     Now, compile and train the autoencoder with .compile() and .fit()

     After that, apply the LSTM classifier
     >>> x = yildirim.lstm(encoder)
     >>> classifier = keras.Model(inputs=inputs, outputs=x)

     Now, compile and train with .compile() and fit()
     """
    # TODO: Check parameters of conv layers
    encoder = layers.Conv1D(filters=260, kernel_size=16, strides=1)(inputs)
    encoder = layers.MaxPooling1D(pool_size=2)(encoder)
    encoder = layers.Conv1D(filters=130, kernel_size=64, strides=1)(encoder)
    encoder = layers.BatchNormalization()(encoder)
    encoder = layers.MaxPooling1D(pool_size=2)(encoder)
    encoder = layers.Conv1D(filters=65, kernel_size=32, strides=1)(encoder)
    encoder = layers.Conv1D(filters=65, kernel_size=1, strides=1)(encoder)

    decoder = layers.MaxPooling1D(pool_size=2)(encoder)
    decoder = layers.Conv1D(filters=32, kernel_size=1, strides=1)(decoder)
    decoder = layers.Conv1D(filters=32, kernel_size=32, strides=1)(decoder)
    decoder = layers.UpSampling1D(size=2)(decoder)
    decoder = layers.Conv1D(filters=64, kernel_size=64, strides=1)(decoder)
    decoder = layers.UpSampling1D(size=2)(decoder)
    decoder = layers.Conv1D(filters=128, kernel_size=16, strides=1)(decoder)

    lstm = layers.LSTM(units=32, return_sequences=True)(encoder)

    if include_top:
        lstm = layers.Flatten()(lstm)

    return encoder, decoder, lstm


def CaiWenjuan(inputs, include_top=True):
    """
    DDNN network presented in:

    Cai, Wenjuan, et al. "Accurate detection of atrial fibrillation from 12-lead ECG using deep neural network."
    Computers in biology and medicine 116 (2020): 103378.

    Args:
        include_top (bool):  Include a Flatten layer as the last layer of the model. Default is True.
   """
    dense_layers = [2, 4, 6, 4]
    x1 = layers.Conv1D(filters=8, kernel_size=1, strides=1, padding='same')(inputs)
    x2 = layers.Conv1D(filters=16, kernel_size=3, strides=1, padding='same')(inputs)
    x3 = layers.Conv1D(filters=24, kernel_size=5, strides=1, padding='same')(inputs)
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

    return x


def KimMinGu(inputs, X, y, include_top=True, num_classes=89):  # NOTE: kernel sizes are not specified in the original paper.

    epochs = [5, 5, 5, 5, 7, 7]#[500, 500, 500, 500, 750, 750]
    b_sizes = [512, 512, 256, 512, 256]
    dropout = [0.5, 0.6, 0.6, 0.7, 0.5, 0.7]

    models = []
    for d in dropout:
        model = layers.Conv1D(filters=32, kernel_size=3, padding="same", activation=relu)(inputs)
        model = layers.MaxPooling1D(pool_size=2)(model)
        for f in [64, 128, 256, 512]:
            model = layers.Conv1D(filters=f, kernel_size=3, padding="same", activation=relu)(model)
            model = layers.MaxPooling1D(pool_size=2)(model)

        if include_top:
            model = layers.Flatten()(model)
            model = layers.Dense(units=1028, activation=relu)(model)
            model = layers.Dropout(d)(model)
            model = layers.Dense(units=1028, activation=relu)(model)
            model = layers.Dense(units=num_classes, activation=softmax)(model)

        models.append(keras.Model(inputs=inputs, outputs=model))

    # Train the models
    for m, e, b in zip(models, epochs, b_sizes):
        m.compile(optimizer='Adam', loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
        m.fit(X, y, batch_size=b, epochs=e)

    # Get the predictions of each model
    preds = [np.argmax(m.predict(X), axis=1) for m in models]

    # Join the predictions of each model
    # TODO: Y a continuación, que?
    newX = np.column_stack(preds)

    return None  # TODO: not finished yet
