import pyCNN_LSTM
import os
import wfdb
import numpy as np
from tensorflow import keras
from typing import Optional, Tuple
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


def get_segments(data: wfdb.Record,
                 annotations: wfdb.Annotation,
                 labels: np.ndarray,
                 left_offset: int = 99,
                 right_offset: int = 160,
                 fixed_length: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """ It generates the segments of uninterrupted sequences of arrythmia beats into the corresponding arrythmia groups
    in labels.

    :param data:            The arrythmia signal as a wfdb Record class
    :param annotations:     The set of annotations as a wfdb Annotation class
    :param labels:          The set of valid labels for the different segments. Segments with different labels are discarded
    :param left_offset:     The number of instance at the left of the first R peak of the segment. Default to 99
    :param right_offset:    The number of instances at the right of the last R peak of the segment. Default to 160
    :param fixed_length:    Should the segments have a fixed length? If fixed_length is a number, then the segments will
                            have the specified length. If the segment length is greater than fixed_length, it is truncated
                            or padded with zeros otherwise. Default to None.

    :return:                A tuple that contains the data and the associated labels. Data has a shape of (N, T, V)
                            where N is the number of segments (or instances), V is the number of variables (1 in this case)
                            and T is the number of timesteps of each segment.  Labels are numerically encoded according to the
                            value passed in the :parameter labels param.
    """
    i = 0
    annot_segments = []

    # Get the tuples for consecutive symbols. The tuple is (first, last, symbol) where first is the index of the first occurrence of symbol,
    # and last is the index of the last consecutive ocurrence.
    while(i < len(annotations.symbol)):
        first = i
        current_symbol = annotations.symbol[i]
        while(i < len(annotations.symbol) and annotations.symbol[i] == current_symbol):
            i += 1
        last = i-1
        tup = (first, last, current_symbol)
        annot_segments.append(tup)

    # Now, for each extracted tuple, get the X segments:
    result = []
    classes = []
    for s in annot_segments:  # s is a tuple (first, last, symbol)
        if s[2] in labels:
            classes.append(s[2])
            init = annotations.sample[s[0]] - left_offset
            if init < 0:
                init = 0

            end = annotations.sample[s[1]] + right_offset
            if end >= len(data.p_signal):
                end = len(data.p_signal) - 1

            r = range(init, end)

            # Get the samples of the segments (p_signal is a 2D array, we only want the first axis)
            new_segment = np.array(data.p_signal[r, 1], dtype='float32')

            # truncate or pad with zeros the segment if necessary
            if (fixed_length != None):
                if (len(new_segment) > fixed_length):  # truncate
                    new_segment = new_segment[:fixed_length]
                elif (len(new_segment < fixed_length)):  # pad with zeros to the right
                    number_of_zeros = fixed_length - len(new_segment)
                    new_segment = np.pad(new_segment, (0, number_of_zeros), mode='constant', constant_values=0)

            result.append(new_segment)

    result = np.stack(result, axis=0)
    result = np.reshape(result, (result.shape[0], result.shape[1], 1))  # shape[0] segments with 1 variable, with shape[1] timestamps each
    classes = np.array(classes, dtype=str)

    # Encode labels: from string to numeric.
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    classes = label_encoder.transform(classes)

    return (result, classes)


def read_mit_bih(path: str,
                 labels: np.ndarray = np.array(['N', 'L', 'R', 'A', 'V']),
                 left_offset: int = 99,
                 right_offset: int = 160,
                 fixed_length: Optional[int] = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """ It reads the MIT-BIH Arrythmia X with the specified default configuration of the work presented at:
    Oh, Shu Lih, et al. "Automated diagnosis of arrhythmia using combination of CNN and LSTM techniques with
    variable length heart beats." Computers in biology and medicine 102 (2018): 278-287.

    :param labels:              The labels of the different types of arrythmia to be employed
    :param path:                The path of the directory where the X files are stored. Note: The X and annotations
                                files must have the same name, but different extension (annotations must have .atr extension)
    :param left_offset:         The number of instances at the left of the first R peak of the segment. Defaults to 99
    :param right_offset:        The number of instances at the right of the last R peak of the segment. Defaults to 160
    :param fixed_length:        If different to None, the segment will have the specified number of instances. Note that
                                if the segment length > fixed_length it will be truncate or padded with zeros otherwise.

    :return:                     A tuple that contains the data and the associated labels as an ndarray. Data has a shape of (N, T, V)
                                where N is the number of segments (or instances), V is the number of variables (1 in this case)
                                and T is the number of timesteps of each segment.  Labels are numerically encoded according to the
                                value passed in the :parameter labels param.
    """
    print("reading data...")
    segments = []
    classes = []

    files = [file[:-4] for file in os.listdir(path) if file.endswith('.dat')]
    for f in files:
        data = wfdb.rdrecord(path + f)
        annotation = wfdb.rdann(path + f, 'atr')

        s, clazz = get_segments(data=data,
                                 annotations=annotation,
                                 labels=labels,
                                 left_offset=left_offset,
                                 right_offset=right_offset,
                                 fixed_length=fixed_length)

        segments.append(s)
        classes.append(clazz)

    segments = np.vstack(segments)
    classes = np.concatenate(classes)
    print("done.")

    return (segments, classes)


# Leemos los datos
dir = "../physionet.org/files/mitdb/1.0.0/"

X, y = read_mit_bih(dir)

# mostramos la forma de los datos de entrada. En total tenemos 16499 series temporales
# de 1 variable con 1000 instantes de tiempo cada una de ellas.
# Cada serie temporal tiene únicamente 1 valor asociado o clase que determina el tipo de arritmia
print("X shape: ", X.shape)
print("y shape: ", y.shape)

y_hot_encoded = np.zeros((y.size, y.max()+1))
y_hot_encoded[np.arange(y.size), y] = 1
y_hot_encoded

# Define the input
inp = keras.Input((X.shape[1], X.shape[2]))

# Define the models to be tested
models = {
    # 'KimMinGu': pyCNN_LSTM.KimMinGu(inp, X, y_hot_encoded, num_classes=5)
    'CaiWenjuan': pyCNN_LSTM.CaiWenjuan(inp),
    'OhShuLih': pyCNN_LSTM.OhShuLih(inp),
    'KhanZulfiqar': pyCNN_LSTM.KhanZulfiqar(inp),
    'ZhengZhenyu': pyCNN_LSTM.ZhengZhenyu(inp),
    'HouBourui': pyCNN_LSTM.HouBoroui(inp),
    'WangKejun': pyCNN_LSTM.WangKejun(inp),
    'ChenChen': pyCNN_LSTM.ChenChen(inp),
    'KimTaeYoung': pyCNN_LSTM.KimTaeYoung(inp),
    'GenMixing': pyCNN_LSTM.GenMinxing(inp),
    'FuJiangmeng': pyCNN_LSTM.FuJiangmeng(inp),
    'ShiHaotian': pyCNN_LSTM.ShiHaotian(inp),
    'HuangMeiLing': pyCNN_LSTM.HuangMeiLing(inp),
    'LihOhShu': pyCNN_LSTM.LihOhShu(inp),
    'GaoJunli': pyCNN_LSTM.GaoJunLi(inp),
    'WeiXiaoyan': pyCNN_LSTM.WeiXiaoyan(inp),
    'KongZhengmin': pyCNN_LSTM.KongZhengmin(inp),
    # 'YildirimOzal': pyCNN_LSTM.YildirimOzal(inp)
}

# Run each model
for name, model in models.items():
    # Define the model
    x = keras.layers.Dense(10, activation='relu')(model)
    out = keras.layers.Dense(5, activation='softmax')(x)
    new_model = keras.Model(inputs=inp, outputs=out, name=name)
    new_model.summary()

    # Compile & fit
    print("running ", name, " .....")
    new_model.compile(optimizer='Adam', loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
    new_model.fit(X, y_hot_encoded, batch_size=256, epochs=25, verbose=0)

    # evaluate
    pred = np.argmax(new_model.predict(X), axis=1)
    acc = accuracy_score(y, pred)
    print("Accuracy: " + str(acc))
