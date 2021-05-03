from _ast import arg

from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.activations import relu, softmax, sigmoid


def transition_block(x, reduction, name):
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


def conv_block(x, growth_rate, name):
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


def dense_block(x, blocks, growth_rate, name):
    """A dense block of densenet for 1D data.

    Arguments:
      x: input tensor.
      blocks: integer, the number of building blocks.
      name: string, block label.

    Returns:
      Output tensor for the block.
    """
    for i in range(blocks):
        x = conv_block(x, growth_rate, name=name + '_block' + str(i + 1))
    return x


def se_module(x, dense_units):
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


class OhShuLih(models.Model):
    """
    CNN+LSTM model.

    References
    ----------

        Oh, Shu Lih, et al. "Automated diagnosis of arrhythmia using combination of CNN and LSTM techniques with
        variable length heart beats." Computers in biology and medicine 102 (2018): 278-287.

     Parameters
    ----------
        include_top: bool, default=True

            Include a Flatten layer as the last layer of the model. Default is True.

    Attributes
    ----------
        include_top: bool

            Boolean that indicates whether the last layer is a Flatten layer or not.
    """

    def __init__(self, include_top: bool = True):
        super(OhShuLih, self).__init__()
        self.include_top = include_top

        # Network Defintion:
        self.conv_A = layers.Conv1D(filters=64, kernel_size=3, activation=relu)
        self.maxpooling_A = layers.MaxPooling1D(pool_size=4)
        self.dropout_A = layers.Dropout(rate=0.3)
        self.conv_B = layers.Conv1D(filters=16, kernel_size=3, activation=relu)
        self.maxpooling_B = layers.MaxPooling1D(pool_size=4)
        self.dropout_B = layers.Dropout(rate=0.3)
        self.normalization_A = layers.BatchNormalization()
        self.lstm_A = layers.LSTM(units=16)
        if include_top:
            self.flatten_A = layers.Flatten()

        # self.dense_A = layers.Dense(1)

    def call(self, inputs):
        model = self.conv_A(inputs)
        model = self.maxpooling_A(model)
        model = self.dropout_A(model)
        model = self.conv_B(model)
        model = self.maxpooling_B(model)
        model = self.dropout_B(model)
        model = self.normalization_A(model)
        model = self.lstm_A(model)
        if self.include_top:
            model = self.flatten_A(model)
        # model = self.dense_A(model)
        return model


class KhanZulfiqar(models.Model):
    """Khan, Zulfiqar Ahmad, et al. "Towards Efficient Electricity Forecasting in Residential and Commercial
    Buildings: A Novel Hybrid CNN with a LSTM-AE based Framework." Sensors 20.5 (2020): 1399. """

    def __init__(self, include_top: bool = True):
        super(KhanZulfiqar, self).__init__()
        self.include_top = include_top

        self.conv_A = layers.Conv1D(filters=8, kernel_size=1, padding='same', activation='relu')
        self.dropout_A = layers.Dropout(0.1)
        self.conv_B = layers.Conv1D(filters=16, kernel_size=1, padding='same', activation='relu')
        self.dropout_B = layers.Dropout(0.1)
        self.lstm_A = layers.LSTM(50, activation='relu', return_sequences=True)
        self.lstm_B = layers.LSTM(50, activation='relu', return_sequences=True)
        if self.include_top:
            self.flatten_A = layers.Flatten()
        #self.dense_A = layers.Dense(1)

    def call(self, inputs):
        model = self.conv_A(inputs)
        model = self.dropout_A(model)
        model = self.conv_B(model)
        model = self.dropout_B(model)
        model = self.lstm_A(model)
        model = self.lstm_B(model)
        if self.include_top:
            model = self.flatten_A(model)
        #model = self.dense_A(model)
        return model


class ZhengZhenyu(models.Model):
    """Zheng, Zhenyu, et al. "An Automatic Diagnosis of Arrhythmias Using a Combination of CNN and LSTM Technology."
    Electronics 9.1 (2020): 121. """

    def __init__(self, include_top: bool = True):
        super(ZhengZhenyu, self).__init__()
        self.include_top = include_top

        self.conv_A = layers.Conv1D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu')
        self.conv_B = layers.Conv1D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu')
        self.maxpooling_A = layers.MaxPooling1D(4, strides=1)
        self.conv_C = layers.Conv1D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu')
        self.conv_D = layers.Conv1D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu')
        self.maxpooling_B = layers.MaxPooling1D(4, strides=1)
        self.conv_E = layers.Conv1D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu')
        self.conv_F = layers.Conv1D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu')
        self.maxpooling_C = layers.MaxPooling1D(4, strides=1)
        self.lstm_A = layers.LSTM(16, return_sequences=True)
        if self.include_top:
            self.flatten_A = layers.Flatten()
        #self.dense_A = layers.Dense(1)

    def call(self, inputs):
        model = self.conv_A(inputs)
        model = self.conv_B(model)
        model = self.maxpooling_A(model)
        model = self.conv_C(model)
        model = self.conv_D(model)
        model = self.maxpooling_B(model)
        model = self.conv_E(model)
        model = self.conv_F(model)
        model = self.maxpooling_C(model)
        model = self.lstm_A(model)
        if self.include_top:
            model = self.flatten_A(model)
        #model = self.dense_A(model)
        return model


# This model has no flatten layers
class HouBoroui(models.Model):
    """Hou, Borui, et al. "LSTM based auto-encoder model for ECG arrhythmias classification." IEEE Transactions on
    Instrumentation and Measurement (2019). """

    def __init__(self):
        super(HouBoroui, self).__init__()
        self.lstm_A = layers.LSTM(146)
        self.lstm_B = layers.LSTM(31)
        #self.dense_A = layers.Dense(73)

    def call(self, inputs):
        model = self.lstm_A(inputs)
        model = self.lstm_B(model)
        #model = self.dense_A(model)
        return model


class WangKejun(models.Model):
    """Wang, Kejun, Xiaoxia Qi, and Hongda Liu. "Photovoltaic power forecasting based LSTM-Convolutional Network."
    Energy 189 (2019): 116225. """

    def __init__(self, include_top: bool = True):
        super(WangKejun, self).__init__()
        self.include_top = include_top

        self.lstm_A = layers.LSTM(64, return_sequences=True)
        self.lstm_B = layers.LSTM(128, return_sequences=True)
        self.conv_A = layers.Conv1D(filters=64, kernel_size=3, strides=1)
        self.maxpooling_A = layers.MaxPooling1D(2, strides=2)
        self.conv_B = layers.Conv1D(filters=150, kernel_size=2, strides=1)
        self.maxpooling_B = layers.MaxPooling1D(2, strides=2)
        self.dropout_A = layers.Dropout(0.1)
        if self.include_top:
            self.flatten_A = layers.Flatten()
        #self.dense_A = layers.Dense(2048)
        #self.dense_B = layers.Dense(1024)
        #self.dense_C = layers.Dense(1)

    def call(self, inputs):
        model = self.lstm_A(inputs)
        model = self.conv_A(model)
        model = self.maxpooling_A(model)
        model = self.conv_B(model)
        model = self.maxpooling_B(model)
        model = self.dropout_A(model)
        if self.include_top:
            model = self.flatten_A(model)
        #model = self.dense_A(model)
        #model = self.dense_B(model)
        #model = self.dense_C(model)
        return model


class ChenChen(models.Model):
    """Chen, Chen, et al. "Automated arrhythmia classification based on a combination network of CNN and LSTM."
    Biomedical Signal Processing and Control 57 (2020): 101819."""

    def __init__(self, include_top: bool = True):
        super(ChenChen, self).__init__()
        self.include_top = include_top

        self.conv_A = layers.Conv1D(filters=251, kernel_size=2, strides=1)
        self.maxpooling_A = layers.MaxPooling1D(2, strides=1)
        self.conv_B = layers.Conv1D(filters=150, kernel_size=2, strides=1)
        self.maxpooling_B = layers.MaxPooling1D(2, strides=1)
        self.conv_C = layers.Conv1D(filters=100, kernel_size=2, strides=1)
        self.maxpooling_C = layers.MaxPooling1D(2, strides=1)
        self.conv_D = layers.Conv1D(filters=81, kernel_size=2, strides=1)
        self.maxpooling_D = layers.MaxPooling1D(2, strides=1)
        self.conv_E = layers.Conv1D(filters=61, kernel_size=2, strides=1)
        self.maxpooling_E = layers.MaxPooling1D(2, strides=1)
        self.conv_F = layers.Conv1D(filters=14, kernel_size=2, strides=1)
        self.maxpooling_F = layers.MaxPooling1D(2, strides=1)
        self.lstm_A = layers.LSTM(64, return_sequences=True)
        self.lstm_A = layers.LSTM(32)
        if self.include_top:
            self.flatten_A = layers.Flatten()
        #self.dense_A = layers.Dense(1)

    def call(self, inputs):
        model = self.conv_A(inputs)
        model = self.conv_B(model)
        model = self.maxpooling_A(model)
        model = self.conv_C(model)
        model = self.conv_D(model)
        model = self.maxpooling_B(model)
        model = self.conv_E(model)
        model = self.conv_F(model)
        model = self.maxpooling_C(model)
        model = self.lstm_A(model)
        if self.include_top:
            model = self.flatten_A(model)
        #model = self.dense_A(model)
        return model


class KimTaeYoung(models.Model):
    """Kim, Tae-Young, and Sung-Bae Cho. "Predicting residential energy consumption using CNN-LSTM neural networks."
    Energy 182 (2019): 72-81. """

    def __init__(self, include_top: bool = True):
        super(KimTaeYoung, self).__init__()
        self.include_top = include_top

        self.conv_A = layers.Conv1D(filters=64, kernel_size=2, strides=1)
        self.maxpooling_A = layers.MaxPooling1D(2, strides=1)
        self.conv_B = layers.Conv1D(filters=150, kernel_size=2, strides=1)
        self.maxpooling_B = layers.MaxPooling1D(2, strides=1)
        self.lstm_A = layers.LSTM(64, activation='tanh', return_sequences=True)
        if self.include_top:
            self.flatten_A = layers.Flatten()
        #self.dense_A = layers.Dense(32)
        #self.dense_B = layers.Dense(1)

    def call(self, inputs):
        model = self.conv_A(inputs)
        model = self.maxpooling_A(model)
        model = self.conv_B(model)
        model = self.maxpooling_B(model)
        model = self.lstm_A(model)
        if self.include_top:
            model = self.flatten_A(model)
        #model = self.dense_A(model)
        #model = self.dense_B(model)
        return model


class GenMinxing(models.Model):
    """Geng, Minxing, et al. "Epileptic Seizure Detection Based on Stockwell Transform and Bidirectional Long
    Short-Term Memory." IEEE Transactions on Neural Systems and Rehabilitation Engineering 28.3 (2020): 573-580. """

    def __init__(self, include_top: bool = True):
        super(GenMinxing, self).__init__()
        self.include_top = include_top

        self.bidirectional_A = layers.Bidirectional(layers.LSTM(units=40, return_sequences=True))
        if self.include_top:
            self.flatten_A = layers.Flatten()
        #self.dense_A = layers.Dense(1)

    def call(self, inputs):
        model = self.bidirectional_A(inputs)
        if self.include_top:
            model = self.flatten_A(model)
        #model = self.dense_A(model)
        return model


class FuJiangmeng(models.Model):
    """Fu, Jiangmeng, et al. "A hybrid CNN-LSTM model based actuator fault diagnosis for six-rotor UAVs." 2019
    Chinese Control And Decision Conference (CCDC). IEEE, 2019. """

    def __init__(self, include_top: bool = True):
        super(FuJiangmeng, self).__init__()
        self.include_top = include_top

        self.conv_A = layers.Conv1D(filters=32, kernel_size=1, padding='same', activation='relu')
        self.maxpooling_A = layers.MaxPooling1D(2, strides=1)
        self.lstm_A = layers.LSTM(256, activation='relu', return_sequences=True)
        if self.include_top:
            self.flatten_A = layers.Flatten()
        #self.dense_A = layers.Dense(1, activation='softmax')

    def call(self, inputs):
        model = self.conv_A(inputs)
        model = self.maxpooling_A(model)
        model = self.lstm_A(model)
        if self.include_top:
            model = self.flatten_A(model)
        #model = self.dense_A(model)
        return model


class ShiHaotian(models.Model):
    """Shi, Haotian, et al. "Automated heartbeat classification based on deep neural network with multiple input
    layers." Knowledge-Based Systems 188 (2020): 105036. """

    def __init__(self, include_top: bool = True):
        super(ShiHaotian, self).__init__()
        self.include_top = include_top

        self.conv_A = layers.Conv1D(filters=32, activation='relu', kernel_size=13, strides=2)
        self.maxpooling_A = layers.MaxPooling1D(2, strides=2)
        self.conv_B = layers.Conv1D(filters=32, activation='relu', kernel_size=13, strides=1)
        self.maxpooling_B = layers.MaxPooling1D(2, strides=2)
        self.conv_C = layers.Conv1D(filters=32, activation='relu', kernel_size=13, strides=2)
        self.maxpooling_C = layers.MaxPooling1D(2, strides=2)
        self.concatenate_A = layers.Concatenate(axis=1)
        self.lstm_A = layers.LSTM(32, return_sequences=True)
        if self.include_top:
            self.flatten_A = layers.Flatten()
        #self.dense_A = layers.Dense(1)

    def call(self, inputs):
        model_A = self.conv_A(inputs)
        model_A = self.maxpooling_A(model_A)
        model_B = self.conv_B(inputs)
        model_B = self.maxpooling_B(model_B)
        model_C = self.conv_C(inputs)
        model_C = self.maxpooling_C(model_C)
        model = self.concatenate_A([model_A, model_B, model_C])
        model = self.lstm_A(model)
        if self.include_top:
            model = self.flatten_A(model)
        #model = self.dense_A(model)
        return model


class HuangMeiLing(models.Model):
    """
    CNN model, employed for 2-D images of ECG data. This model is adapted for 1-D time series

    References
    -------
        Huang, Mei-Ling, and Yan-Sheng Wu. "Classification of atrial fibrillation and normal sinus rhythm based on
        convolutional neural network." Biomedical Engineering Letters (2020): 1-11.


    Parameters
    ----------
        include_top: bool, default=True

            Include a Flatten layer as the last layer of the model. Default is True.

    Attributes
    ----------
        include_top: bool

            Boolean that indicates whether the last layer is a Flatten layer or not.
    """

    def __init__(self, include_top: bool = True):
        super(HuangMeiLing, self).__init__()
        self.include_top = include_top

        # Network definition
        self.conv_A = layers.Conv1D(filters=48, activation=relu, kernel_size=15, strides=6)
        self.maxpooling_A = layers.MaxPooling1D(pool_size=2, strides=1)
        self.conv_B = layers.Conv1D(filters=256, activation=relu, kernel_size=13, strides=1)
        self.maxpooling_B = layers.MaxPooling1D(pool_size=2, strides=1)

        if include_top:
            self.flatten_A = layers.Flatten()

    def call(self, inputs):
        model = self.conv_A(inputs)
        model = self.maxpooling_A(model)
        model = self.conv_B(model)
        model = self.maxpooling_B(model)
        if self.include_top:
            model = self.flatten_A(model)

        return model


class LihOhShu(models.Model):
    """
    CNN+LSTM model

    References
    ----------
        Lih, Oh Shu, et al. "Comprehensive electrocardiographic diagnosis based on deep learning." Artificial
        Intelligence in Medicine 103 (2020): 101789.

    Parameters
    ----------
        include_top: bool, default=True

            Include a Flatten layer as the last layer of the model. Default is True.

    Attributes
    ----------
        include_top: bool

            Boolean that indicates whether the last layer is a Flatten layer or not.
    """

    def __init__(self, include_top: bool = True):
        super(LihOhShu, self).__init__()
        self.include_top = include_top

        # Network defintiion
        self.conv_blocks = models.Sequential()
        for filters, k_size in zip([3, 6, 6, 6, 6],
                                   [20, 10, 5, 5, 10]):
            self.conv_blocks.add(layers.Conv1D(filters=filters, activation=relu, kernel_size=k_size, strides=1))
            self.conv_blocks.add(layers.MaxPooling1D(pool_size=2, strides=1))
        # self.conv_A = layers.Conv1D(filters=3, activation=relu, kernel_size=20, strides=1)
        # self.maxpooling_A = layers.MaxPooling1D(pool_size=2, strides=1)
        # self.conv_B = layers.Conv1D(filters=6, activation=relu, kernel_size=10, strides=1)
        # self.maxpooling_B = layers.MaxPooling1D(pool_size=2, strides=1)
        # self.conv_C = layers.Conv1D(filters=6, activation=relu, kernel_size=5, strides=1)
        # self.maxpooling_C = layers.MaxPooling1D(pool_size=2, strides=1)
        # self.conv_D = layers.Conv1D(filters=6, activation=relu, kernel_size=5, strides=1)
        # self.maxpooling_D = layers.MaxPooling1D(pool_size=2, strides=1)
        # self.conv_E = layers.Conv1D(filters=6, activation=relu, kernel_size=10, strides=1)
        # self.maxpooling_E = layers.MaxPooling1D(pool_size=2, strides=1)

        self.lstm_A = layers.LSTM(units=10, return_sequences=True)
        if include_top:
            self.flatten_A = layers.Flatten()

        # top_layers:
        # self.dense_A = layers.Dense(8)
        # self.dropout_A = layers.Dropout(0.5)
        # self.dense_B = layers.Dense(4)
        # self.dropout_B = layers.Dropout(0.5)
        # self.dense_C = layers.Dense(4)
        # self.dropout_C = layers.Dropout(0.5)
        # self.dense_D = layers.Dense(1)

    def call(self, inputs):
        # model = self.conv_A(inputs)
        # model = self.maxpooling_A(model)
        # model = self.conv_B(model)
        # model = self.maxpooling_B(model)
        # model = self.conv_C(model)
        # model = self.maxpooling_C(model)
        # model = self.conv_D(model)
        # model = self.maxpooling_D(model)
        # model = self.conv_E(model)
        # model = self.maxpooling_E(model)
        model = self.conv_blocks(inputs)
        model = self.lstm_A(model)
        if self.include_top:
            model = self.flatten_A(model)
        # model = self.dense_A(model)
        # model = self.dropout_A(model)
        # model = self.dense_B(model)
        # model = self.dropout_B(model)
        # model = self.dense_C(model)
        # model = self.dropout_C(model)
        # model = self.dense_D(model)
        return model


class GaoJunli(models.Model):
    """
    LSTM network

    References
    ----------
        Gao, Junli, et al. "An Effective LSTM Recurrent Network to Detect Arrhythmia on Imbalanced ECG Dataset."
        Journal of healthcare engineering 2019 (2019).

    Parameters
    ----------
        include_top: bool, default=True

            Include a Flatten layer as the last layer of the model. Default is True.

    Attributes
    ----------
        include_top: bool

            Boolean that indicates whether the last layer is a Flatten layer or not.

    """

    def __init__(self, include_top: bool = True):
        super(GaoJunli, self).__init__()
        self.include_top = include_top

        # Network definiton
        self.lstm_A = layers.LSTM(units=64, return_sequences=True)
        if include_top:
            self.flatten_A = layers.Flatten()

        # self.dense_A = layers.Dense(32)
        # self.dense_B = layers.Dense(8)
        # self.dense_C = layers.Dense(1)

    def call(self, inputs):
        model = self.lstm_A(inputs)
        if self.include_top:
            model = self.flatten_A(model)
        # model = self.dense_A(model)
        # model = self.dense_B(model)
        # model = self.dense_C(model)
        return model


class WeiXiaoyan(models.Model):
    """
    CNN+LSTM model employed for 2-D images of ECG data. This model is adapted for 1-D time series.


    References
    ----------

        Wei, Xiaoyan, et al. "Early prediction of epileptic seizures using a long-term recurrent convolutional
        network." Journal of neuroscience methods 327 (2019): 108395.

    Parameters
    ----------
        include_top: bool, default=True

            Include a Flatten layer as the last layer of the model. Default is True.

    Attributes
    ----------
        include_top: bool

             Boolean that indicates whether the last layer is a Flatten layer or not.
    """

    def __init__(self, include_top: bool = True):
        super(WeiXiaoyan, self).__init__()

        self.include_top = include_top

        # Network Defintion (CNN)
        self.convolutional_blocks = keras.Sequential()
        for filters, kernel, strides in zip([32, 64, 128, 256, 512],
                                            [5, 3, 3, 3, 3],
                                            [1, 1, 1, 1, 1]):
            self.convolutional_blocks.add(layers.Conv1D(filters=filters, kernel_size=kernel, strides=strides))
            self.convolutional_blocks.add(layers.LeakyReLU(alpha=0.01))
            self.convolutional_blocks.add(layers.MaxPooling1D(pool_size=2, strides=2))
            self.convolutional_blocks.add(layers.BatchNormalization())

        # LSTM
        self.lstm_A = layers.LSTM(units=512, return_sequences=True)
        self.lstm_B = layers.LSTM(units=512, return_sequences=True)

        if include_top:
            self.flatten_A = layers.Flatten()


    def call(self, inputs):
        model = self.convolutional_blocks(inputs)
        model = self.lstm_A(model)
        model = self.lstm_B(model)
        if self.include_top:
            model = self.flatten_A(model)

        return model


class KongZhengmin(models.Model):
    """
    CNN+LSTM

    References
    ----------
        Kong, Zhengmin, et al. "Convolution and Long Short-Term Memory Hybrid Deep Neural Networks for Remaining
        Useful Life Prognostics." Applied Sciences 9.19 (2019): 4156.

    Parameters
    ----------
        include_top: bool, default=True

            Include a Flatten layer as the last layer of the model. Default is True.

    Attributes
    ----------
        include_top: bool

            Boolean that indicates whether the last layer is a Flatten layer or not.
    """

    def __init__(self, include_top: bool = True):
        super(KongZhengmin, self).__init__()
        self.include_top = include_top

        # Network Definition
        self.conv_A = layers.Conv1D(filters=32, activation=relu, kernel_size=5, strides=1)
        self.maxpooling_A = layers.MaxPooling1D(pool_size=2, strides=2)
        self.lstm_A = layers.LSTM(64, return_sequences=True)
        self.lstm_B = layers.LSTM(64, return_sequences=True)
        if include_top:
            self.flatten_A = layers.Flatten()

    def call(self, inputs):
        model = self.conv_A(inputs)
        model = self.maxpooling_A(model)
        model = self.lstm_A(model)
        model = self.lstm_B(model)
        if self.include_top:
            model = self.flatten_A(model)

        return model


class YildirimOzal(models.Model):
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

    def __init__(self, include_top: bool = True):
        super(YildirimOzal, self).__init__()
        self.include_top = include_top

        # Network definition # TODO: CHECK DIMENSIONS OF KERNELS AND FILTERS
        self.encoder = models.Sequential([
            layers.Conv1D(filters=260, kernel_size=16, strides=5),
            layers.MaxPooling1D(pool_size=2),
            layers.Conv1D(filters=130, kernel_size=64, strides=5),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Conv1D(filters=65, kernel_size=32, strides=3),
            layers.Conv1D(filters=65, kernel_size=1, strides=3)]
        )

        # decoder
        self.decoder = models.Sequential([
            layers.MaxPooling1D(pool_size=2),
            layers.Conv1D(filters=32, kernel_size=1, strides=3),
            layers.Conv1D(filters=32, kernel_size=32, strides=3),
            layers.UpSampling1D(size=2),
            layers.Conv1D(filters=64, kernel_size=64, strides=5),
            layers.UpSampling1D(size=2),
            layers.Conv1D(filters=128, kernel_size=16, strides=5)]
        )

        # lstm
        self.lstm = layers.LSTM(units=32)

        if include_top:
            self.flatten_A = layers.Flatten()

    def call(self, inputs):

        # This part only connects the already trained encoder with the LSTM classification layer
        model = self.encoder(inputs)
        model = self.lstm(model)

        if self.include_top:
            model = self.flatten_A(model)
        return model

# TODO: Change all models -> Use def instead of classes
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
        x = se_module(x, dense_units=128)
        x = dense_block(x, n_blocks, growth_rate=6, name="conv" + str(i + 1))
        x = se_module(x, dense_units=128)
        # Last dense block does not have transition block.
        if i < len(dense_layers) - 1:
            x = transition_block(x, reduction=0.5, name="transition" + str(i + 1))

    x = layers.GlobalAveragePooling1D()(x)

    if include_top:
        x = layers.Flatten()(x)

    return x
