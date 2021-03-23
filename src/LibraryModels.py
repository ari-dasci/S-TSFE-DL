from keras import layers, models


class OhShuLih(models.Model):
    """Oh, Shu Lih, et al. "Automated diagnosis of arrhythmia using combination of CNN and LSTM techniques with
    variable length heart beats." Computers in biology and medicine 102 (2018): 278-287. """

    def __init__(self):
        super(OhShuLih, self).__init__()
        self.conv_A = layers.Conv1D(64, 3, activation="relu")
        self.maxpooling_A = layers.MaxPooling1D(4)
        self.dropout_A = layers.Dropout(0.3)
        self.conv_B = layers.Conv1D(16, 3, activation="relu")
        self.maxpooling_B = layers.MaxPooling1D(4)
        self.dropout_B = layers.Dropout(0.3)
        self.normalization_A = layers.BatchNormalization()
        self.lstm_A = layers.LSTM(16)
        self.flatten_A = layers.Flatten()
        self.dense_A = layers.Dense(1)

    def call(self, inputs):
        model = self.conv_A(inputs)
        model = self.maxpooling_A(model)
        model = self.dropout_A(model)
        model = self.conv_B(model)
        model = self.maxpooling_B(model)
        model = self.dropout_B(model)
        model = self.normalization_A(model)
        model = self.lstm_A(model)
        model = self.flatten_A(model)
        model = self.dense_A(model)
        return model


class KhanZulfiqar(models.Model):
    """Khan, Zulfiqar Ahmad, et al. "Towards Efficient Electricity Forecasting in Residential and Commercial
    Buildings: A Novel Hybrid CNN with a LSTM-AE based Framework." Sensors 20.5 (2020): 1399. """

    def __init__(self):
        super(KhanZulfiqar, self).__init__()
        self.conv_A = layers.Conv1D(filters=8, kernel_size=1, padding='same', activation='relu')
        self.dropout_A = layers.Dropout(0.1)
        self.conv_B = layers.Conv1D(filters=16, kernel_size=1, padding='same', activation='relu')
        self.dropout_B = layers.Dropout(0.1)
        self.lstm_A = layers.LSTM(50, activation='relu', return_sequences=True)
        self.lstm_B = layers.LSTM(50, activation='relu', return_sequences=True)
        self.flatten_A = layers.Flatten()
        self.dense_A = layers.Dense(1)

    def call(self, inputs):
        model = self.conv_A(inputs)
        model = self.dropout_A(model)
        model = self.conv_B(model)
        model = self.dropout_B(model)
        model = self.lstm_A(model)
        model = self.lstm_B(model)
        model = self.flatten_A(model)
        model = self.dense_A(model)
        return model


class ZhengZhenyu(models.Model):
    """Zheng, Zhenyu, et al. "An Automatic Diagnosis of Arrhythmias Using a Combination of CNN and LSTM Technology."
    Electronics 9.1 (2020): 121. """

    def __init__(self):
        super(ZhengZhenyu, self).__init__()
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
        self.flatten_A = layers.Flatten()
        self.dense_A = layers.Dense(1)

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
        model = self.flatten_A(model)
        model = self.dense_A(model)
        return model


class HouBoroui(models.Model):
    """Hou, Borui, et al. "LSTM based auto-encoder model for ECG arrhythmias classification." IEEE Transactions on
    Instrumentation and Measurement (2019). """

    def __init__(self):
        super(HouBoroui, self).__init__()
        self.lstm_A = layers.LSTM(146)
        self.lstm_B = layers.LSTM(31)
        self.dense_A = layers.Dense(73)

    def call(self, inputs):
        model = self.lstm_A(inputs)
        model = self.lstm_B(model)
        model = self.dense_A(model)
        return model


class WangKejun(models.Model):
    """Wang, Kejun, Xiaoxia Qi, and Hongda Liu. "Photovoltaic power forecasting based LSTM-Convolutional Network."
    Energy 189 (2019): 116225. """

    def __init__(self):
        super(WangKejun, self).__init__()
        self.lstm_A = layers.LSTM(64, return_sequences=True)
        self.lstm_B = layers.LSTM(128, return_sequences=True)
        self.conv_A = layers.Conv1D(filters=64, kernel_size=3, strides=1)
        self.maxpooling_A = layers.MaxPooling1D(2, strides=2)
        self.conv_B = layers.Conv1D(filters=150, kernel_size=2, strides=1)
        self.maxpooling_B = layers.MaxPooling1D(2, strides=2)
        self.dropout_A = layers.Dropout(0.1)
        self.flatten_A = layers.Flatten()
        self.dense_A = layers.Dense(2048)
        self.dense_B = layers.Dense(1024)
        self.dense_C = layers.Dense(1)

    def call(self, inputs):
        model = self.lstm_A(inputs)
        model = self.conv_A(model)
        model = self.maxpooling_A(model)
        model = self.conv_B(model)
        model = self.maxpooling_B(model)
        model = self.dropout_A(model)
        model = self.flatten_A(model)
        model = self.dense_A(model)
        model = self.dense_B(model)
        model = self.dense_C(model)
        return model


class ChenChen(models.Model):
    """Chen, Chen, et al. "Automated arrhythmia classification based on a combination network of CNN and LSTM."
    Biomedical Signal Processing and Control 57 (2020): 101819."""

    def __init__(self):
        super(ChenChen, self).__init__()
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
        self.flatten_A = layers.Flatten()
        self.dense_A = layers.Dense(1)

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
        model = self.flatten_A(model)
        model = self.dense_A(model)
        return model


class KimTaeYoung(models.Model):
    """Kim, Tae-Young, and Sung-Bae Cho. "Predicting residential energy consumption using CNN-LSTM neural networks."
    Energy 182 (2019): 72-81. """

    def __init__(self):
        super(KimTaeYoung, self).__init__()
        self.conv_A = layers.Conv1D(filters=64, kernel_size=2, strides=1)
        self.maxpooling_A = layers.MaxPooling1D(2, strides=1)
        self.conv_B = layers.Conv1D(filters=150, kernel_size=2, strides=1)
        self.maxpooling_B = layers.MaxPooling1D(2, strides=1)
        self.lstm_A = layers.LSTM(64, activation='tanh', return_sequences=True)
        self.flatten_A = layers.Flatten()
        self.dense_A = layers.Dense(32)
        self.dense_B = layers.Dense(1)

    def call(self, inputs):
        model = self.conv_A(inputs)
        model = self.maxpooling_A(model)
        model = self.conv_B(model)
        model = self.maxpooling_B(model)
        model = self.lstm_A(model)
        model = self.flatten_A(model)
        model = self.dense_A(model)
        model = self.dense_B(model)
        return model


class GenMinxing(models.Model):
    """Geng, Minxing, et al. "Epileptic Seizure Detection Based on Stockwell Transform and Bidirectional Long
    Short-Term Memory." IEEE Transactions on Neural Systems and Rehabilitation Engineering 28.3 (2020): 573-580. """

    def __init__(self):
        super(GenMinxing, self).__init__()
        self.bidirectional_A = layers.Bidirectional(layers.LSTM(units=40, return_sequences=True))
        self.flatten_A = layers.Flatten()
        self.dense_A = layers.Dense(1)

    def call(self, inputs):
        model = self.bidirectional_A(inputs)
        model = self.flatten_A(model)
        model = self.dense_A(model)
        return model


class FuJiangmeng(models.Model):
    """Fu, Jiangmeng, et al. "A hybrid CNN-LSTM model based actuator fault diagnosis for six-rotor UAVs." 2019
    Chinese Control And Decision Conference (CCDC). IEEE, 2019. """

    def __init__(self):
        super(FuJiangmeng, self).__init__()
        self.conv_A = layers.Conv1D(filters=32, kernel_size=1, padding='same', activation='relu')
        self.maxpooling_A = layers.MaxPooling1D(2, strides=1)
        self.lstm_A = layers.LSTM(256, activation='relu', return_sequences=True)
        self.flatten_A = layers.Flatten()
        self.dense_A = layers.Dense(1, activation='softmax')

    def call(self, inputs):
        model = self.conv_A(inputs)
        model = self.maxpooling_A(model)
        model = self.lstm_A(model)
        model = self.flatten_A(model)
        model = self.dense_A(model)
        return model


class ShiHaotian(models.Model):
    """Shi, Haotian, et al. "Automated heartbeat classification based on deep neural network with multiple input
    layers." Knowledge-Based Systems 188 (2020): 105036. """

    def __init__(self):
        super(ShiHaotian, self).__init__()
        self.conv_A = layers.Conv1D(filters=32, activation='relu', kernel_size=13, strides=2)
        self.maxpooling_A = layers.MaxPooling1D(2, strides=2)
        self.conv_B = layers.Conv1D(filters=32, activation='relu', kernel_size=13, strides=1)
        self.maxpooling_B = layers.MaxPooling1D(2, strides=2)
        self.conv_C = layers.Conv1D(filters=32, activation='relu', kernel_size=13, strides=2)
        self.maxpooling_C = layers.MaxPooling1D(2, strides=2)
        self.concatenate_A = layers.Concatenate(axis=1)
        self.lstm_A = layers.LSTM(32, return_sequences=True)
        self.flatten_A = layers.Flatten()
        self.dense_A = layers.Dense(1)

    def call(self, inputs):
        model_A = self.conv_A(inputs)
        model_A = self.maxpooling_A(model_A)
        model_B = self.conv_B(inputs)
        model_B = self.maxpooling_B(model_B)
        model_C = self.conv_C(inputs)
        model_C = self.maxpooling_C(model_C)
        model = self.concatenate_A([model_A, model_B, model_C])
        model = self.lstm_A(model)
        model = self.flatten_A(model)
        model = self.dense_A(model)
        return model


class HuangMeiLing(models.Model):
    """Huang, Mei-Ling, and Yan-Sheng Wu. "Classification of atrial fibrillation and normal sinus rhythm based on
    convolutional neural network." Biomedical Engineering Letters (2020): 1-11. """

    def __init__(self):
        super(HuangMeiLing, self).__init__()
        self.conv_A = layers.Conv1D(filters=48, activation='relu', kernel_size=15, strides=6)
        self.maxpooling_A = layers.MaxPooling1D(2, strides=1)
        self.conv_B = layers.Conv1D(filters=256, activation='relu', kernel_size=13, strides=1)
        self.maxpooling_B = layers.MaxPooling1D(2, strides=1)
        self.flatten_A = layers.Flatten()
        self.dense_A = layers.Dense(1)

    def call(self, inputs):
        model = self.conv_A(inputs)
        model = self.maxpooling_A(model)
        model = self.conv_B(model)
        model = self.maxpooling_B(model)
        model = self.flatten_A(model)
        model = self.dense_A(model)
        return model


class LihOhShu(models.Model):
    """Lih, Oh Shu, et al. "Comprehensive electrocardiographic diagnosis based on deep learning." Artificial
    Intelligence in Medicine 103 (2020): 101789. """

    def __init__(self):
        super(LihOhShu, self).__init__()
        self.conv_A = layers.Conv1D(filters=3, activation='relu', kernel_size=20, strides=1)
        self.maxpooling_A = layers.MaxPooling1D(2, strides=1)
        self.conv_B = layers.Conv1D(filters=6, activation='relu', kernel_size=10, strides=1)
        self.maxpooling_B = layers.MaxPooling1D(2, strides=1)
        self.conv_C = layers.Conv1D(filters=6, activation='relu', kernel_size=5, strides=1)
        self.maxpooling_C = layers.MaxPooling1D(2, strides=1)
        self.conv_D = layers.Conv1D(filters=6, activation='relu', kernel_size=5, strides=1)
        self.maxpooling_D = layers.MaxPooling1D(2, strides=1)
        self.conv_E = layers.Conv1D(filters=6, activation='relu', kernel_size=10, strides=1)
        self.maxpooling_E = layers.MaxPooling1D(2, strides=1)
        self.lstm_A = layers.LSTM(10, return_sequences=True)
        self.flatten_A = layers.Flatten()
        self.dense_A = layers.Dense(8)
        self.dropout_A = layers.Dropout(0.5)
        self.dense_B = layers.Dense(4)
        self.dropout_B = layers.Dropout(0.5)
        self.dense_C = layers.Dense(4)
        self.dropout_C = layers.Dropout(0.5)
        self.dense_D = layers.Dense(1)

    def call(self, inputs):
        model = self.conv_A(inputs)
        model = self.maxpooling_A(model)
        model = self.conv_B(model)
        model = self.maxpooling_B(model)
        model = self.conv_C(model)
        model = self.maxpooling_C(model)
        model = self.conv_D(model)
        model = self.maxpooling_D(model)
        model = self.conv_E(model)
        model = self.maxpooling_E(model)
        model = self.lstm_A(model)
        model = self.flatten_A(model)
        model = self.dense_A(model)
        model = self.dropout_A(model)
        model = self.dense_B(model)
        model = self.dropout_B(model)
        model = self.dense_C(model)
        model = self.dropout_C(model)
        model = self.dense_D(model)
        return model


class GaoJunli(models.Model):
    """Gao, Junli, et al. "An Effective LSTM Recurrent Network to Detect Arrhythmia on Imbalanced ECG Dataset."
    Journal of healthcare engineering 2019 (2019). """

    def __init__(self):
        super(GaoJunli, self).__init__()
        self.lstm_A = layers.LSTM(64, return_sequences=True)
        self.flatten_A = layers.Flatten()
        self.dense_A = layers.Dense(32)
        self.dense_B = layers.Dense(8)
        self.dense_C = layers.Dense(1)

    def call(self, inputs):
        model = self.lstm_A(inputs)
        model = self.flatten_A(model)
        model = self.dense_A(model)
        model = self.dense_B(model)
        model = self.dense_C(model)
        return model


class WeiXiaoyan(models.Model):
    """Wei, Xiaoyan, et al. "Early prediction of epileptic seizures using a long-term recurrent convolutional
    network." Journal of neuroscience methods 327 (2019): 108395. """

    def __init__(self):
        super(WeiXiaoyan, self).__init__()
        self.conv_A = layers.Conv1D(filters=32, activation='relu', kernel_size=5, strides=1)
        self.maxpooling_A = layers.MaxPooling1D(2, strides=2)
        self.normalization_A = layers.BatchNormalization()
        self.conv_B = layers.Conv1D(filters=64, activation='relu', kernel_size=3, strides=1)
        self.maxpooling_B = layers.MaxPooling1D(2, strides=2)
        self.normalization_B = layers.BatchNormalization()
        self.conv_C = layers.Conv1D(filters=128, activation='relu', kernel_size=3, strides=1)
        self.maxpooling_C = layers.MaxPooling1D(2, strides=2)
        self.normalization_C = layers.BatchNormalization()
        self.conv_D = layers.Conv1D(filters=256, activation='relu', kernel_size=3, strides=1)
        self.maxpooling_D = layers.MaxPooling1D(2, strides=2)
        self.normalization_D = layers.BatchNormalization()
        self.conv_E = layers.Conv1D(filters=512, activation='relu', kernel_size=3, strides=1)
        self.maxpooling_E = layers.MaxPooling1D(2, strides=2)
        self.normalization_E = layers.BatchNormalization()
        self.lstm_A = layers.LSTM(512, return_sequences=True)
        self.lstm_B = layers.LSTM(512, return_sequences=True)
        self.flatten_A = layers.Flatten()
        self.dense_A = layers.Dense(1)

    def call(self, inputs):
        model = self.conv_A(inputs)
        model = self.maxpooling_A(model)
        model = self.normalization_A(model)
        model = self.conv_B(model)
        model = self.maxpooling_B(model)
        model = self.normalization_B(model)
        model = self.conv_C(model)
        model = self.maxpooling_C(model)
        model = self.normalization_C(model)
        model = self.conv_D(model)
        model = self.maxpooling_D(model)
        model = self.normalization_D(model)
        model = self.conv_E(model)
        model = self.maxpooling_E(model)
        model = self.normalization_E(model)
        model = self.lstm_A(model)
        model = self.lstm_B(model)
        model = self.flatten_A(model)
        model = self.dense_A(model)
        return model


class CaiWenjuan(models.Model):
    """Cai, Wenjuan, et al. "Accurate detection of atrial fibrillation from 12-lead ECG using deep neural network."
    Computers in biology and medicine 116 (2020): 103378. """

    def __init__(self):
        super(CaiWenjuan, self).__init__()
        self.conv_A = layers.Conv1D(8, 1)
        self.conv_B = layers.Conv1D(16, 3)
        self.conv_C = layers.Conv1D(24, 5)
        self.concatenate_A = layers.Concatenate(axis=1)
        self.globalaveragepooling_A = layers.GlobalAveragePooling1D()
        self.dense_A = layers.Dense(32)
        self.dense_B = layers.Dense(32, activation='sigmoid')
        self.conv_D = layers.Conv1D(8, 1)
        self.conv_E = layers.Conv1D(16, 3)
        self.globalaveragepooling_B = layers.GlobalAveragePooling1D()
        self.dense_C = layers.Dense(32)
        self.dense_D = layers.Dense(32, activation='sigmoid')
        self.globalaveragepooling_C = layers.GlobalAveragePooling1D()
        self.dense_E = layers.Dense(1)

    def call(self, inputs):
        model_A = self.conv_A(inputs)
        model_B = self.conv_B(inputs)
        model_C = self.conv_C(inputs)
        model = self.concatenate_A()([model_A, model_B, model_C])
        model = self.globalaveragepooling_A(model)
        model = self.dense_A(model)
        model = self.dense_B(model)
        model = self.conv_A(model)
        model = self.conv_B(model)
        model = self.globalaveragepooling_B(model)
        model = self.dense_C(model)
        model = self.dense_D(model)
        model = self.globalaveragepooling_C(model)
        model = self.dense_E(model)
        return model


class KongZhengmin(models.Model):
    """Kong, Zhengmin, et al. "Convolution and Long Short-Term Memory Hybrid Deep Neural Networks for Remaining
    Useful Life Prognostics." Applied Sciences 9.19 (2019): 4156. """

    def __init__(self):
        super(KongZhengmin, self).__init__()
        self.conv_A = layers.Conv1D(filters=32, activation='relu', kernel_size=5, strides=1)
        self.maxpooling_A = layers.MaxPooling1D(2, strides=2)
        self.lstm_A = layers.LSTM(64, return_sequences=True)
        self.lstm_B = layers.LSTM(64, return_sequences=True)
        self.flatten_A = layers.Flatten()
        self.dense_A = layers.Dense(50)
        self.dense_B = layers.Dense(50)
        self.dense_C = layers.Dense(1)

    def call(self, inputs):
        model = self.conv_A(inputs)
        model = self.maxpooling_A(model)
        model = self.lstm_A(model)
        model = self.lstm_B(model)
        model = self.dense_A(model)
        model = self.dense_B(model)
        model = self.dense_C(model)
        return model


class YildirimOzal(models.Model):
    """Yildirim, Ozal, et al. "A new approach for arrhythmia classification using deep coded features and LSTM
    networks." Computer methods and programs in biomedicine 176 (2019): 121-133. """

    def __init__(self):
        super(YildirimOzal, self).__init__()
        self.conv_A = layers.Conv1D(260, 16, strides=5)
        self.maxpooling_A = layers.MaxPooling1D(2)
        self.conv_B = layers.Conv1D(130, 64, strides=5)
        self.normalization_A = layers.BatchNormalization()
        self.maxpooling_B = layers.MaxPooling1D(2)
        self.conv_C = layers.Conv1D(65, 32, strides=3)
        self.conv_D = layers.Conv1D(65, 1, strides=3)
        self.maxpooling_C = layers.MaxPooling1D(2)
        self.conv_E = layers.Conv1D(32, 1, strides=3)
        self.conv_F = layers.Conv1D(32, 32, strides=3)
        self.upsampling_A = layers.UpSampling1D(size=2)
        self.conv_G = layers.Conv1D(64, 64, strides=5)
        self.upsampling_B = layers.UpSampling1D(size=2)
        self.conv_H = layers.Conv1D(128, 16, strides=5)
        self.flatten_A = layers.Flatten()
        self.dense_E = layers.Dense(73)

    def call(self, inputs):
        model = self.conv_A(inputs)
        model = self.maxpooling_A(model)
        model = self.conv_B(model)
        model = self.normalization_A(model)
        model = self.maxpooling_B(model)
        model = self.conv_C(model)
        model = self.conv_D(model)
        model = self.maxpooling_C(model)
        model = self.conv_E(model)
        model = self.conv_F(model)
        model = self.upsampling_A(model)
        model = self.conv_G(model)
        model = self.upsampling_B(model)
        model = self.conv_H(model)
        model = self.flatten_A(model)
        model = self.dense_E(model)
        return model
