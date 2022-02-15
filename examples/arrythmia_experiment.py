import sys
sys.path.append("..")
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import TSFEDL.models_keras as TSFEDL
from TSFEDL.data import read_mit_bih
import csv


def getData(size=1000, test_size=0.2):
    """Gets the segments with the specified size from the MIT-BIH arrythmia dataset.
        Returns a partition of the data as follow:
        (X_tra, y_tra, y_hotcoded_tra) -> the instances and labels of the training set
        (X_tst, y_tst, y_hotcoded_tst) -> the instances and labels of the test set.

        Parameters
        ----------
        size : int, default=1000
            Size of the time series.
        test_size : float, default=0.2
            Percentage of the dataset to be used for testing. This number
            should be between 0 and 1.

        Returns
        -------
        (X_tra, y_tra, y_HC_tra) :
            Train triplet.
        (X_tst, y_tst, y_HC_tst) :
            Test triplet.
    """
    print("Load data...")
    dir = "physionet.org/files/mitdb/1.0.0/"
    X, y = read_mit_bih(dir, fixed_length=size)
    y_hot_encoded = np.zeros((y.size, y.max() + 1))
    y_hot_encoded[np.arange(y.size), y] = 1
    X_tra, X_tst, y_tra, y_tst, y_HC_tra, y_HC_tst = train_test_split(X, y, y_hot_encoded,
                                                                      test_size=test_size,
                                                                      random_state=1234,
                                                                      stratify=y)
    return (X_tra, y_tra, y_HC_tra), (X_tst, y_tst, y_HC_tst)


def trainModel(model, X_tra, y_tra, y_hot_encoded_tra,
               X_tst, y_tst, y_hot_encoded_tst,
               loss=keras.losses.categorical_crossentropy,
               metrics=['accuracy'],
               batch_size=256,
               epochs=6):
    """
    It trains the given model using the given data and label

    Parameters
    ----------
    model : Keras model,
        Model to be trained.
    X_tra : array-like
        Training data.
    y_tra : array-like
        Training labels.
    y_hot_encoded_tra : array-like
        Training labels hot encoded for the softmax layer.
    X_tst : array-like
        Test data.
    y_tst : array-like
        Test labels.
    y_hot_encoded_tst : array-like
        Test labels hot encoded for the softmax layer.
    loss : Keras loss
        Loss for training the model.
    metrics : list of str
        This string list represents the metrics passed to the fit function of Keras.
    batch_size : int, default=256
        Batch size passed to the neural network for training.
    epochs : int, default=6
        Number of epochs of the training phase.


    Returns
    -------
    acc : float
        Accuracy obtained with the model over the test data.
    pred : array-like
        Labels predicted for the test data.
    """
    # use early stopping and model checkpoint to save the best model found so far
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
    mc = tf.keras.callbacks.ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True)

    # compile and train
    model.compile(optimizer='Adam', loss=loss, metrics=metrics)
    model.fit(X_tra, y_hot_encoded_tra, batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[early_stopping, mc])

    # evaluate
    pred = np.argmax(model.predict(X_tst), axis=1)
    acc = accuracy_score(y_tst, pred)
    print("Accuracy: " + str(acc))
    return acc, pred


def generateNewModel(input_tensor):
    """
    It generates a new model using a the OhShuLih model plus an additional building block in the TSFEDL library.

    Parameters
    ----------
    input_tensor : Tensorflow tensor
        Input tensor of the model.

    Returns
    -------
    new_model : Keras model
        OhShuLih model to predict 5 classes.
    """
    model = TSFEDL.OhShuLih(input_tensor=input_tensor, include_top=False)
    lstm_layer = model.layers[-1]  # get the LSTM
    x = model.layers[-2].output
    x = TSFEDL.RTA_block(x, nb_filter=6, kernel_size=3)  # Add the RTA block
    x = lstm_layer(x)

    # now, include the top_module:
    x = keras.layers.Dropout(rate=0.2)(x)
    x = keras.layers.Dense(units=20, activation='relu')(x)
    x = keras.layers.Dense(units=10, activation='relu')(x)
    x = keras.layers.Dense(units=5, activation='softmax')(x)

    # Defin the model and return
    new_model = keras.Model(inputs=input, outputs=x)
    return new_model


metrics = ['accuracy']
data_seq_length = 1000  # the length of each instance
batch_size = 256
epochs = 150

# Retrieve the data instances
# We generate instances with different sizes as some models in the library requires more timesteps to work properly.
(X_tra, y_tra, y_hc_tra), (X_tst, y_tst, y_hc_tst) = getData()
(X_tra_2k, y_tra_2k, y_hc_tra_2k), (X_tst_2k, y_tst_2k, y_hc_tst_2k) = getData(size=2000)
(X_tra_3k, y_tra_3k, y_hc_tra_3k), (X_tst_3k, y_tst_3k, y_hc_tst_3k) = getData(size=3600)
input = keras.Input((X_tra.shape[1], X_tra.shape[2]))
input_2k = keras.Input((X_tra_2k.shape[1], X_tra_2k.shape[2]))
input3k = keras.Input((X_tra_3k.shape[1], X_tra_3k.shape[2]))

# Creates a dictionary that contains all the models of the library plus the new one.
methods_dict = {
    'ShiHaotian': TSFEDL.ShiHaotian(input_tensor=input, include_top=True),
    'YildirimOzal': TSFEDL.YildirimOzal(input_tensor=input, include_top=True),
    'ChenChen': TSFEDL.ChenChen(input_tensor=input3k, include_top=True),
    'LihOhShu': TSFEDL.LihOhShu(input_tensor=input_2k, include_top=True),
    'OhShuLi' : TSFEDL.OhShuLih(input_tensor=input, include_top=True),
    'KhanZulfiqar' : TSFEDL.KhanZulfiqar(input_tensor=input, include_top=True),
    'ZhengZhenyu' : TSFEDL.ZhengZhenyu(input_tensor=input, include_top=True),
    'WangKejun': TSFEDL.WangKejun(input_tensor=input, include_top=True),
    'KimTaeYoung': TSFEDL.KimTaeYoung(input_tensor=input, include_top=True),
    'GenMinxing': TSFEDL.GenMinxing(input_tensor=input, include_top=True),
    'FuJiangmeng': TSFEDL.FuJiangmeng(input_tensor=input, include_top=True),
    'HuangMeiLing': TSFEDL.HuangMeiLing(input_tensor=input, include_top=True),
    'GaoJunLi': TSFEDL.GaoJunLi(input_tensor=input, include_top=True),
    'WeiXiaoyan': TSFEDL.WeiXiaoyan(input_tensor=input, include_top=True),
    'KongZhengmin': TSFEDL.KongZhengmin(input_tensor=input, include_top=True),
    'CaiWenjuan': TSFEDL.CaiWenjuan(input_tensor=input, include_top=True),
    'HtetMyetLynn': TSFEDL.HtetMyetLynn(input_tensor=input, include_top=True),
    'ZhangJin': TSFEDL.ZhangJin(input_tensor=input, include_top=True),
    'YaoQihang': TSFEDL.YaoQihang(input_tensor=input, include_top=True),
    'YiboGao': TSFEDL.YiboGao(input_tensor=input, include_top=True, return_loss=False),
    'NewModel': generateNewModel(input)
}

# Accuracy results for each method will be stored in this dictionary
results = {}

# Now, run each model accordingly:
for name, model in methods_dict.items():
    print("Training", name)
    if name == "YildirimOzal":
        # This model is an autoencoder + an LSTM and the top module, so:
        # First, train the autoencoder
        ae, enc, classifier = model
        ae.compile(optimizer='Adam', loss=keras.losses.MSE, metrics=['mse'])
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
        mc = tf.keras.callbacks.ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True)
        ae.fit(X_tra, X_tra, batch_size=batch_size, epochs=epochs, verbose=0, callbacks=[early_stopping, mc])

        # Next, train the LSTM and the classifier with the encoded features.
        enc.trainable = False
        ae.trainable = False
        acc, class_preds = trainModel(classifier, X_tra, y_tra, y_hc_tra, X_tst, y_tst, y_hc_tst, epochs=epochs, batch_size=batch_size)
        results[name] = acc
    elif name == "ChenChen":
        # This models requires instances with more than 3600 timesteps
        acc, preds = trainModel(model, X_tra_3k, y_tra_3k, y_hc_tra_3k, X_tst_3k, y_tst_3k, y_hc_tst_3k, batch_size=batch_size, epochs=epochs)
        results[name] = acc
    elif name == "LihOhShu":
        # this model requires instances with more than 2000 timesteps
        acc, preds = trainModel(model, X_tra_2k, y_tra_2k, y_hc_tra_2k, X_tst_2k, y_tst_2k, y_hc_tst_2k,
                                batch_size=batch_size, epochs=epochs)
    else:
        # The remaining models can be trained as usual.
        acc, preds = acc, preds = trainModel(model, X_tra, y_tra, y_hc_tra, X_tst, y_tst, y_hc_tst, batch_size=batch_size, epochs=epochs)
        results[name] = acc

# write results to the file
with open('results_TSFEDL_arrythmia.csv', 'w') as f:
    writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for i in results.items():
        writer.writerow(i)

print("DONE.")
