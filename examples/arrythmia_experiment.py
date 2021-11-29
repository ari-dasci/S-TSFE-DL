import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pyCNN_LSTM.models_keras as pyCNN_LSTM
from pyCNN_LSTM.data import read_mit_bih
import csv


def getData(size=1000, test_size=0.2):
    """Gets the segments with the specified size from the MIT-BIH arrythmia dataset.
        Returns a partition of the data as follow:
        (X_tra, y_tra, y_hotcoded_tra) -> the instances and labels of the training set
        (X_tst, y_tst, y_hotcoded_tst) -> the instances and labels of the test set.
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
    It generates a new model using a the OhShuLih model plus an additional building block in the pyCNN_LSTM library.

    Returns
    -------
    The new model
    """
    model = pyCNN_LSTM.OhShuLih(input_tensor=input_tensor, include_top=False)
    lstm_layer = model.layers[-1]  # get the LSTM
    x = model.layers[-2].output
    x = pyCNN_LSTM.RTA_block(x, nb_filter=6, kernel_size=3)  # Add the RTA block
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
    'ShiHaotian': pyCNN_LSTM.ShiHaotian(input_tensor=input, include_top=True),
    'YildirimOzal': pyCNN_LSTM.YildirimOzal(input_tensor=input, include_top=True),
    'ChenChen': pyCNN_LSTM.ChenChen(input_tensor=input3k, include_top=True),
    'LihOhShu': pyCNN_LSTM.LihOhShu(input_tensor=input_2k, include_top=True),
    'OhShuLi' : pyCNN_LSTM.OhShuLih(input_tensor=input, include_top=True),
    'KhanZulfiqar' : pyCNN_LSTM.KhanZulfiqar(input_tensor=input, include_top=True),
    'ZhengZhenyu' : pyCNN_LSTM.ZhengZhenyu(input_tensor=input, include_top=True),
    'WangKejun': pyCNN_LSTM.WangKejun(input_tensor=input, include_top=True),
    'KimTaeYoung': pyCNN_LSTM.KimTaeYoung(input_tensor=input, include_top=True),
    'GenMinxing': pyCNN_LSTM.GenMinxing(input_tensor=input, include_top=True),
    'FuJiangmeng': pyCNN_LSTM.FuJiangmeng(input_tensor=input, include_top=True),
    'HuangMeiLing': pyCNN_LSTM.HuangMeiLing(input_tensor=input, include_top=True),
    'GaoJunLi': pyCNN_LSTM.GaoJunLi(input_tensor=input, include_top=True),
    'WeiXiaoyan': pyCNN_LSTM.WeiXiaoyan(input_tensor=input, include_top=True),
    'KongZhengmin': pyCNN_LSTM.KongZhengmin(input_tensor=input, include_top=True),
    'CaiWenjuan': pyCNN_LSTM.CaiWenjuan(input_tensor=input, include_top=True),
    'HtetMyetLynn': pyCNN_LSTM.HtetMyetLynn(input_tensor=input, include_top=True),
    'ZhangJin': pyCNN_LSTM.ZhangJin(input_tensor=input, include_top=True),
    'YaoQihang': pyCNN_LSTM.YaoQihang(input_tensor=input, include_top=True),
    'YiboGao': pyCNN_LSTM.YiboGao(input_tensor=input, include_top=True, return_loss=False),
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
with open('results_pyCNN_LSTM_arrythmia.csv', 'w') as f:
    writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for i in results.items():
        writer.writerow(i)

print("DONE.")
