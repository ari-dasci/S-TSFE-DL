import optuna
import sys
sys.path.append("../TSFEDL")
import TSFEDL.models_keras as models_keras
import utils
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import sklearn
from timeseries_batch_generator import TimeseriesGenerator_Multistep

INSTANCE_BATCH = 4096
BATCH_SIZE = 128
NTHREADS = 64
QUEUE_SIZE = 16

print("Reading dataset...")
data, attack_types, classes = utils.readDataset("./dataset/kddcup/kddcup.data", "./dataset/kddcup/kddcup.names")
data = sklearn.preprocessing.StandardScaler().fit_transform(data)

attack_types_dict = {}
cont=0
for att in attack_types:
    attack_types_dict[att]=cont
    cont+=1

for i in range(len(classes)):
    classes[i] = attack_types_dict[classes[i]]

classes[classes!=attack_types_dict["normal"]] = 1
classes[classes==attack_types_dict["normal"]] = 0
classes_test = classes[-500000:]

print("Preparing data for training...")
# normal_train = np.where(classes[:700000]==0)[0]
# X_train, y_train = utils.sliceDataset(data[normal_train], batch_size=INSTANCE_BATCH, npred=4)
# X_test, y_test = utils.sliceDataset(data[700000:850000], batch_size=INSTANCE_BATCH, npred=4)
normal_train = np.where(classes[:-500000]==0)[0]

train_generator = TimeseriesGenerator_Multistep(data[normal_train], data[normal_train], length=INSTANCE_BATCH, sampling_rate=1,
                                                        batch_size = BATCH_SIZE, stride=4, length_target=4,
                                                        shuffle=True, behaviour="forecast")

test_generator = TimeseriesGenerator_Multistep(data[-500000:], data[-500000:], length=INSTANCE_BATCH, sampling_rate=1,
                                                        batch_size = BATCH_SIZE, stride=4, length_target=4,
                                                        shuffle=True, behaviour="forecast")

print("Creating model...")

model = None

if sys.argv[1]=="OhShuLih":
    model_ = models_keras.OhShuLih(include_top=False, input_shape=(INSTANCE_BATCH, 126))

    x = keras.layers.Dense(units=32, activation=tf.keras.activations.linear, name="dense_1")(model_.layers[-1].output)
    x = keras.layers.Dense(units=64, activation=tf.keras.activations.linear, name="dense_2")(x)
    x = keras.layers.Dense(units=126, activation=tf.keras.activations.linear, name="dense_3")(x)
    x = keras.layers.Dense(units=126*4, activation=tf.keras.activations.linear, name="output")(x)
    x = keras.layers.Reshape((4,126))(x)
    model = keras.Model(inputs=model_.layers[0].output, outputs=x)

    print(model.summary())

    model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.mse)

    my_callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, monitor="loss"),
        tf.keras.callbacks.ModelCheckpoint(filepath='checkpoints/OhShuLih/model.{epoch:02d}-{loss:.2f}.h5', monitor="loss", save_best_only=True),
        tf.keras.callbacks.TensorBoard(log_dir='./logs_ohshulih'),
        tf.keras.callbacks.CSVLogger('log_ohshulih.csv', append=True, separator=';')
    ]

elif sys.argv[1]=="KhanZulfiqar":
    model_ = models_keras.KhanZulfiqar(include_top=False, input_shape=(INSTANCE_BATCH, 126))

    x = keras.layers.Dense(units=32, activation=tf.keras.activations.linear, name="dense_1")(model_.layers[-1].output)
    x = keras.layers.Dense(units=64, activation=tf.keras.activations.linear, name="dense_2")(x)
    x = keras.layers.Dense(units=126, activation=tf.keras.activations.linear, name="dense_3")(x)
    x = keras.layers.Dense(units=126*4, activation=tf.keras.activations.linear, name="output")(x)
    x = keras.layers.Reshape((4,126))(x)
    model = keras.Model(inputs=model_.layers[0].output, outputs=x)

    print(model.summary())

    model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.mse)

    my_callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, monitor="loss"),
        tf.keras.callbacks.ModelCheckpoint(filepath='checkpoints/KhanZulfiqar/model.{epoch:02d}-{loss:.2f}.h5', monitor="loss", save_best_only=True),
        tf.keras.callbacks.TensorBoard(log_dir='./logs_KhanZulfiqar'),
        tf.keras.callbacks.CSVLogger('log_KhanZulfiqar.csv', append=True, separator=';')
    ]

elif sys.argv[1]=="ZhengZhenyu":
    model_ = models_keras.ZhengZhenyu(include_top=False, input_shape=(INSTANCE_BATCH, 126))

    x = keras.layers.Dense(units=32, activation=tf.keras.activations.linear, name="dense_1")(model_.layers[-1].output)
    x = keras.layers.Dense(units=64, activation=tf.keras.activations.linear, name="dense_2")(x)
    x = keras.layers.Dense(units=126, activation=tf.keras.activations.linear, name="dense_3")(x)
    x = keras.layers.Dense(units=126*4, activation=tf.keras.activations.linear, name="output")(x)
    x = keras.layers.Reshape((4,126))(x)
    model = keras.Model(inputs=model_.layers[0].output, outputs=x)

    print(model.summary())

    model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.mse)

    my_callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, monitor="loss"),
        tf.keras.callbacks.ModelCheckpoint(filepath='checkpoints/ZhengZhenyu/model.{epoch:02d}-{loss:.2f}.h5', monitor="loss", save_best_only=True),
        tf.keras.callbacks.TensorBoard(log_dir='./logs_ZhengZhenyu'),
        tf.keras.callbacks.CSVLogger('log_ZhengZhenyu.csv', append=True, separator=';')
    ]

elif sys.argv[1]=="HouBoroui":
    _, model_ = models_keras.HouBoroui(input_shape=(INSTANCE_BATCH, 126), encoder_units=32)

    x = keras.layers.Dense(units=32, activation=tf.keras.activations.linear, name="dense_1")(model_.layers[-1].output)
    x = keras.layers.Dense(units=64, activation=tf.keras.activations.linear, name="dense_2")(x)
    x = keras.layers.Dense(units=126, activation=tf.keras.activations.linear, name="dense_3")(x)
    x = keras.layers.Dense(units=126*4, activation=tf.keras.activations.linear, name="output")(x)
    x = keras.layers.Reshape((4,126))(x)
    model = keras.Model(inputs=model_.layers[0].output, outputs=x)

    print(model.summary())

    model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.mse)

    my_callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, monitor="loss"),
        tf.keras.callbacks.ModelCheckpoint(filepath='checkpoints/HouBoroui/model.{epoch:02d}-{loss:.2f}.h5', monitor="loss", save_best_only=True),
        tf.keras.callbacks.TensorBoard(log_dir='./logs_HouBoroui'),
        tf.keras.callbacks.CSVLogger('log_HouBoroui.csv', append=True, separator=';')
    ]

elif sys.argv[1]=="WangKejun":
    model_ = models_keras.WangKejun(include_top=False, input_shape=(INSTANCE_BATCH, 126))

    x = keras.layers.Flatten()(model_.layers[-1].output)
    x = keras.layers.Dense(units=32, activation=tf.keras.activations.linear, name="dense_1")(x)
    x = keras.layers.Dense(units=64, activation=tf.keras.activations.linear, name="dense_2")(x)
    x = keras.layers.Dense(units=126, activation=tf.keras.activations.linear, name="dense_3")(x)
    x = keras.layers.Dense(units=126*4, activation=tf.keras.activations.linear, name="output")(x)
    x = keras.layers.Reshape((4,126))(x)
    model = keras.Model(inputs=model_.layers[0].output, outputs=x)

    print(model.summary())

    model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.mse)

    my_callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, monitor="loss"),
        tf.keras.callbacks.ModelCheckpoint(filepath='checkpoints/WangKejun/model.{epoch:02d}-{loss:.2f}.h5', monitor="loss", save_best_only=True),
        tf.keras.callbacks.TensorBoard(log_dir='./logs_WangKejun'),
        tf.keras.callbacks.CSVLogger('log_WangKejun.csv', append=True, separator=';')
    ]

elif sys.argv[1]=="ChenChen":
    model_ = models_keras.ChenChen(include_top=False, input_shape=(INSTANCE_BATCH, 126))

    x = keras.layers.Flatten()(model_.layers[-1].output)
    x = keras.layers.Dense(units=32, activation=tf.keras.activations.linear, name="dense_1")(x)
    x = keras.layers.Dense(units=64, activation=tf.keras.activations.linear, name="dense_2")(x)
    x = keras.layers.Dense(units=126, activation=tf.keras.activations.linear, name="dense_3")(x)
    x = keras.layers.Dense(units=126*4, activation=tf.keras.activations.linear, name="output")(x)
    x = keras.layers.Reshape((4,126))(x)
    model = keras.Model(inputs=model_.layers[0].output, outputs=x)

    print(model.summary())

    model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.mse)

    my_callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, monitor="loss"),
        tf.keras.callbacks.ModelCheckpoint(filepath='checkpoints/ChenChen/model.{epoch:02d}-{loss:.2f}.h5', monitor="loss", save_best_only=True),
        tf.keras.callbacks.TensorBoard(log_dir='./logs_ChenChen'),
        tf.keras.callbacks.CSVLogger('log_ChenChen.csv', append=True, separator=';')
    ]

elif sys.argv[1]=="KimTaeYoung":
    model_ = models_keras.KimTaeYoung(include_top=False, input_shape=(INSTANCE_BATCH, 126))

    x = keras.layers.Dense(units=32, activation=tf.keras.activations.linear, name="dense_1")(model_.layers[-1].output)
    x = keras.layers.Dense(units=64, activation=tf.keras.activations.linear, name="dense_2")(x)
    x = keras.layers.Dense(units=126, activation=tf.keras.activations.linear, name="dense_3")(x)
    x = keras.layers.Dense(units=126*4, activation=tf.keras.activations.linear, name="output")(x)
    x = keras.layers.Reshape((4,126))(x)
    model = keras.Model(inputs=model_.layers[0].output, outputs=x)

    print(model.summary())

    model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.mse)

    my_callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, monitor="loss"),
        tf.keras.callbacks.ModelCheckpoint(filepath='checkpoints/KimTaeYoung/model.{epoch:02d}-{loss:.2f}.h5', monitor="loss", save_best_only=True),
        tf.keras.callbacks.TensorBoard(log_dir='./logs_KimTaeYoung'),
        tf.keras.callbacks.CSVLogger('log_KimTaeYoung.csv', append=True, separator=';')
    ]

elif sys.argv[1]=="GenMinxing":
    model_ = models_keras.GenMinxing(include_top=False, input_shape=(INSTANCE_BATCH, 126))

    x = keras.layers.Flatten()(model_.layers[-1].output)
    x = keras.layers.Dense(units=32, activation=tf.keras.activations.linear, name="dense_1")(x)
    x = keras.layers.Dense(units=64, activation=tf.keras.activations.linear, name="dense_2")(x)
    x = keras.layers.Dense(units=126, activation=tf.keras.activations.linear, name="dense_3")(x)
    x = keras.layers.Dense(units=126*4, activation=tf.keras.activations.linear, name="output")(x)
    x = keras.layers.Reshape((4,126))(x)
    model = keras.Model(inputs=model_.layers[0].output, outputs=x)

    print(model.summary())

    model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.mse)

    my_callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, monitor="loss"),
        tf.keras.callbacks.ModelCheckpoint(filepath='checkpoints/GenMinxing/model.{epoch:02d}-{loss:.2f}.h5', monitor="loss", save_best_only=True),
        tf.keras.callbacks.TensorBoard(log_dir='./logs_GenMinxing'),
        tf.keras.callbacks.CSVLogger('log_GenMinxing.csv', append=True, separator=';')
    ]

elif sys.argv[1]=="FuJiangmeng":
    model_ = models_keras.FuJiangmeng(include_top=False, input_shape=(INSTANCE_BATCH, 126))

    x = keras.layers.Dense(units=32, activation=tf.keras.activations.linear, name="dense_1")(model_.layers[-1].output)
    x = keras.layers.Dense(units=64, activation=tf.keras.activations.linear, name="dense_2")(x)
    x = keras.layers.Dense(units=126, activation=tf.keras.activations.linear, name="dense_3")(x)
    x = keras.layers.Dense(units=126*4, activation=tf.keras.activations.linear, name="output")(x)
    x = keras.layers.Reshape((4,126))(x)
    model = keras.Model(inputs=model_.layers[0].output, outputs=x)

    print(model.summary())

    model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.mse)

    my_callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, monitor="loss"),
        tf.keras.callbacks.ModelCheckpoint(filepath='checkpoints/FuJiangmeng/model.{epoch:02d}-{loss:.2f}.h5', monitor="loss", save_best_only=True),
        tf.keras.callbacks.TensorBoard(log_dir='./logs_FuJiangmeng'),
        tf.keras.callbacks.CSVLogger('log_FuJiangmeng.csv', append=True, separator=';')
    ]

elif sys.argv[1]=="ShiHaotian":
    ############################################################################
    ##           DA ERROR CON LOS INPUTS, NECESITA O 1 O 3                    ##
    ############################################################################
    input = keras.Input((INSTANCE_BATCH, 126))
    model_ = models_keras.ShiHaotian(include_top=False, input_tensor=input)

    x = keras.layers.Flatten()(model_.layers[-1].output)
    x = keras.layers.Dense(units=32, activation=tf.keras.activations.linear, name="dense_1")(x)
    x = keras.layers.Dense(units=64, activation=tf.keras.activations.linear, name="dense_2")(x)
    x = keras.layers.Dense(units=126, activation=tf.keras.activations.linear, name="dense_3")(x)
    x = keras.layers.Dense(units=126*4, activation=tf.keras.activations.linear, name="output")(x)
    x = keras.layers.Reshape((4,126))(x)
    model = keras.Model(inputs=model_.layers[0].output, outputs=x)

    print(model.summary())

    model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.mse)

    my_callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, monitor="loss"),
        tf.keras.callbacks.ModelCheckpoint(filepath='checkpoints/ShiHaotian/model.{epoch:02d}-{loss:.2f}.h5', monitor="loss", save_best_only=True),
        tf.keras.callbacks.TensorBoard(log_dir='./logs_ShiHaotian'),
        tf.keras.callbacks.CSVLogger('log_ShiHaotian.csv', append=True, separator=';')
    ]

elif sys.argv[1]=="HuangMeiLing":
    model_ = models_keras.HuangMeiLing(include_top=False, input_shape=(INSTANCE_BATCH, 126))

    x = keras.layers.Flatten()(model_.layers[-1].output)
    x = keras.layers.Dense(units=32, activation=tf.keras.activations.linear, name="dense_1")(x)
    x = keras.layers.Dense(units=64, activation=tf.keras.activations.linear, name="dense_2")(x)
    x = keras.layers.Dense(units=126, activation=tf.keras.activations.linear, name="dense_3")(x)
    x = keras.layers.Dense(units=126*4, activation=tf.keras.activations.linear, name="output")(x)
    x = keras.layers.Reshape((4,126))(x)
    model = keras.Model(inputs=model_.layers[0].output, outputs=x)

    print(model.summary())

    model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.mse)

    my_callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, monitor="loss"),
        tf.keras.callbacks.ModelCheckpoint(filepath='checkpoints/HuangMeiLing/model.{epoch:02d}-{loss:.2f}.h5', monitor="loss", save_best_only=True),
        tf.keras.callbacks.TensorBoard(log_dir='./logs_HuangMeiLing'),
        tf.keras.callbacks.CSVLogger('log_HuangMeiLing.csv', append=True, separator=';')
    ]

elif sys.argv[1]=="LihOhShu":
    model_ = models_keras.LihOhShu(include_top=False, input_shape=(INSTANCE_BATCH, 126))

    x = keras.layers.Dense(units=32, activation=tf.keras.activations.linear, name="dense_1")(model_.layers[-1].output)
    x = keras.layers.Dense(units=64, activation=tf.keras.activations.linear, name="dense_2")(x)
    x = keras.layers.Dense(units=126, activation=tf.keras.activations.linear, name="dense_3")(x)
    x = keras.layers.Dense(units=126*4, activation=tf.keras.activations.linear, name="output")(x)
    x = keras.layers.Reshape((4,126))(x)
    model = keras.Model(inputs=model_.layers[0].output, outputs=x)

    print(model.summary())

    model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.mse)

    my_callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, monitor="loss"),
        tf.keras.callbacks.ModelCheckpoint(filepath='checkpoints/LihOhShu/model.{epoch:02d}-{loss:.2f}.h5', monitor="loss", save_best_only=True),
        tf.keras.callbacks.TensorBoard(log_dir='./logs_LihOhShu'),
        tf.keras.callbacks.CSVLogger('log_LihOhShu.csv', append=True, separator=';')
    ]

elif sys.argv[1]=="GaoJunLi":
    model_ = models_keras.GaoJunLi(include_top=False, input_shape=(INSTANCE_BATCH, 126))

    x = keras.layers.Dense(units=32, activation=tf.keras.activations.linear, name="dense_1")(model_.layers[-1].output)
    x = keras.layers.Dense(units=64, activation=tf.keras.activations.linear, name="dense_2")(x)
    x = keras.layers.Dense(units=126, activation=tf.keras.activations.linear, name="dense_3")(x)
    x = keras.layers.Dense(units=126*4, activation=tf.keras.activations.linear, name="output")(x)
    x = keras.layers.Reshape((4,126))(x)
    model = keras.Model(inputs=model_.layers[0].output, outputs=x)

    print(model.summary())

    model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.mse)

    my_callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, monitor="loss"),
        tf.keras.callbacks.ModelCheckpoint(filepath='checkpoints/GaoJunLi/model.{epoch:02d}-{loss:.2f}.h5', monitor="loss", save_best_only=True),
        tf.keras.callbacks.TensorBoard(log_dir='./logs_GaoJunLi'),
        tf.keras.callbacks.CSVLogger('log_GaoJunLi.csv', append=True, separator=';')
    ]

elif sys.argv[1]=="WeiXiaoyan":
    model_ = models_keras.WeiXiaoyan(include_top=False, input_shape=(INSTANCE_BATCH, 126))

    x = keras.layers.Dense(units=32, activation=tf.keras.activations.linear, name="dense_1")(model_.layers[-1].output)
    x = keras.layers.Dense(units=64, activation=tf.keras.activations.linear, name="dense_2")(x)
    x = keras.layers.Dense(units=126, activation=tf.keras.activations.linear, name="dense_3")(x)
    x = keras.layers.Dense(units=126*4, activation=tf.keras.activations.linear, name="output")(x)
    x = keras.layers.Reshape((4,126))(x)
    model = keras.Model(inputs=model_.layers[0].output, outputs=x)

    print(model.summary())

    model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.mse)

    my_callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, monitor="loss"),
        tf.keras.callbacks.ModelCheckpoint(filepath='checkpoints/WeiXiaoyan/model.{epoch:02d}-{loss:.2f}.h5', monitor="loss", save_best_only=True),
        tf.keras.callbacks.TensorBoard(log_dir='./logs_WeiXiaoyan'),
        tf.keras.callbacks.CSVLogger('log_WeiXiaoyan.csv', append=True, separator=';')
    ]

elif sys.argv[1]=="KongZhengmin":
    model_ = models_keras.KongZhengmin(include_top=False, input_shape=(INSTANCE_BATCH, 126))

    x = keras.layers.Dense(units=32, activation=tf.keras.activations.linear, name="dense_1")(model_.layers[-1].output)
    x = keras.layers.Dense(units=64, activation=tf.keras.activations.linear, name="dense_2")(x)
    x = keras.layers.Dense(units=126, activation=tf.keras.activations.linear, name="dense_3")(x)
    x = keras.layers.Dense(units=126*4, activation=tf.keras.activations.linear, name="output")(x)
    x = keras.layers.Reshape((4,126))(x)
    model = keras.Model(inputs=model_.layers[0].output, outputs=x)

    print(model.summary())

    model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.mse)

    my_callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, monitor="loss"),
        tf.keras.callbacks.ModelCheckpoint(filepath='checkpoints/KongZhengmin/model.{epoch:02d}-{loss:.2f}.h5', monitor="loss", save_best_only=True),
        tf.keras.callbacks.TensorBoard(log_dir='./logs_KongZhengmin'),
        tf.keras.callbacks.CSVLogger('log_KongZhengmin.csv', append=True, separator=';')
    ]

elif sys.argv[1]=="YildirimOzal":
    _, model_, _ = models_keras.YildirimOzal(include_top=False, input_shape=(INSTANCE_BATCH, 126))

    x = keras.layers.Flatten()(model_.layers[-1].output)
    x = keras.layers.Dense(units=32, activation=tf.keras.activations.linear, name="dense_1")(x)
    x = keras.layers.Dense(units=64, activation=tf.keras.activations.linear, name="dense_2")(x)
    x = keras.layers.Dense(units=126, activation=tf.keras.activations.linear, name="dense_3")(x)
    x = keras.layers.Dense(units=126*4, activation=tf.keras.activations.linear, name="output")(x)
    x = keras.layers.Reshape((4,126))(x)
    model = keras.Model(inputs=model_.layers[0].output, outputs=x)

    print(model.summary())

    model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.mse)

    my_callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, monitor="loss"),
        tf.keras.callbacks.ModelCheckpoint(filepath='checkpoints/YildirimOzal/model.{epoch:02d}-{loss:.2f}.h5', monitor="loss", save_best_only=True),
        tf.keras.callbacks.TensorBoard(log_dir='./logs_YildirimOzal'),
        tf.keras.callbacks.CSVLogger('log_YildirimOzal.csv', append=True, separator=';')
    ]

elif sys.argv[1]=="CaiWenjuan":
    model_ = models_keras.CaiWenjuan(include_top=False, input_shape=(INSTANCE_BATCH, 126))

    x = keras.layers.Flatten()(model_.layers[-1].output)
    x = keras.layers.Dense(units=32, activation=tf.keras.activations.linear, name="out_dense_1")(x)
    x = keras.layers.Dense(units=64, activation=tf.keras.activations.linear, name="out_dense_2")(x)
    x = keras.layers.Dense(units=126, activation=tf.keras.activations.linear, name="out_dense_3")(x)
    x = keras.layers.Dense(units=126*4, activation=tf.keras.activations.linear, name="output")(x)
    x = keras.layers.Reshape((4,126))(x)
    model = keras.Model(inputs=model_.layers[0].output, outputs=x)

    print(model.summary())

    model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.mse)

    my_callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, monitor="loss"),
        tf.keras.callbacks.ModelCheckpoint(filepath='checkpoints/CaiWenjuan/model.{epoch:02d}-{loss:.2f}.h5', monitor="loss", save_best_only=True),
        tf.keras.callbacks.TensorBoard(log_dir='./logs_CaiWenjuan'),
        tf.keras.callbacks.CSVLogger('log_CaiWenjuan.csv', append=True, separator=';')
    ]

elif sys.argv[1]=="KimMinGu":
    ############################################################################
    ## ESTE MODELO ES UN ENSEMBLE TENEMOS QUE DECIDIR CÓMO VAMOS A COMBINAR   ##
    ## LAS SALIDAS DE ESTE MODELO PARA PODER GENERAR PUNTUACIONES             ##
    ############################################################################
    model_ = models_keras.KimMinGu(include_top=False, input_shape=(INSTANCE_BATCH, 126))

    x = keras.layers.Flatten()(model_.layers[-1].output)
    x = keras.layers.Dense(units=32, activation=tf.keras.activations.linear, name="out_dense_1")(x)
    x = keras.layers.Dense(units=64, activation=tf.keras.activations.linear, name="out_dense_2")(x)
    x = keras.layers.Dense(units=126, activation=tf.keras.activations.linear, name="out_dense_3")(x)
    x = keras.layers.Dense(units=126*4, activation=tf.keras.activations.linear, name="output")(x)
    x = keras.layers.Reshape((4,126))(x)
    model = keras.Model(inputs=model_.layers[0].output, outputs=x)

    print(model.summary())

    model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.mse)

    my_callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, monitor="loss"),
        tf.keras.callbacks.ModelCheckpoint(filepath='checkpoints/KimMinGu/model.{epoch:02d}-{loss:.2f}.h5', monitor="loss", save_best_only=True),
        tf.keras.callbacks.TensorBoard(log_dir='./logs_KimMinGu'),
        tf.keras.callbacks.CSVLogger('log_KimMinGu.csv', append=True, separator=';')
    ]

elif sys.argv[1]=="HtetMyetLynn":
    model_ = models_keras.HtetMyetLynn(include_top=False, input_shape=(INSTANCE_BATCH, 126))

    x = keras.layers.Flatten()(model_.layers[-1].output)
    x = keras.layers.Dense(units=32, activation=tf.keras.activations.linear, name="out_dense_1")(x)
    x = keras.layers.Dense(units=64, activation=tf.keras.activations.linear, name="out_dense_2")(x)
    x = keras.layers.Dense(units=126, activation=tf.keras.activations.linear, name="out_dense_3")(x)
    x = keras.layers.Dense(units=126*4, activation=tf.keras.activations.linear, name="output")(x)
    x = keras.layers.Reshape((4,126))(x)
    model = keras.Model(inputs=model_.layers[0].output, outputs=x)

    print(model.summary())

    model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.mse)

    my_callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, monitor="loss"),
        tf.keras.callbacks.ModelCheckpoint(filepath='checkpoints/HtetMyetLynn/model.{epoch:02d}-{loss:.2f}.h5', monitor="loss", save_best_only=True),
        tf.keras.callbacks.TensorBoard(log_dir='./logs_HtetMyetLynn'),
        tf.keras.callbacks.CSVLogger('log_HtetMyetLynn.csv', append=True, separator=';')
    ]

elif sys.argv[1]=="ZhangJin":
    model_ = models_keras.ZhangJin(include_top=False, input_shape=(INSTANCE_BATCH, 126))

    x = keras.layers.Flatten()(model_.layers[-1].output)
    x = keras.layers.Dense(units=32, activation=tf.keras.activations.linear, name="out_dense_1")(x)
    x = keras.layers.Dense(units=64, activation=tf.keras.activations.linear, name="out_dense_2")(x)
    x = keras.layers.Dense(units=126, activation=tf.keras.activations.linear, name="out_dense_3")(x)
    x = keras.layers.Dense(units=126*4, activation=tf.keras.activations.linear, name="output")(x)
    x = keras.layers.Reshape((4,126))(x)
    model = keras.Model(inputs=model_.layers[0].output, outputs=x)

    print(model.summary())

    model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.mse)

    my_callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, monitor="loss"),
        tf.keras.callbacks.ModelCheckpoint(filepath='checkpoints/ZhangJin/model.{epoch:02d}-{loss:.2f}.h5', monitor="loss", save_best_only=True),
        tf.keras.callbacks.TensorBoard(log_dir='./logs_ZhangJin'),
        tf.keras.callbacks.CSVLogger('log_ZhangJin.csv', append=True, separator=';')
    ]

elif sys.argv[1]=="YaoQihang":
    model_ = models_keras.YaoQihang(include_top=False, input_shape=(INSTANCE_BATCH, 126))

    x = keras.layers.Flatten()(model_.layers[-1].output)
    x = keras.layers.Dense(units=32, activation=tf.keras.activations.linear, name="out_dense_1")(x)
    x = keras.layers.Dense(units=64, activation=tf.keras.activations.linear, name="out_dense_2")(x)
    x = keras.layers.Dense(units=126, activation=tf.keras.activations.linear, name="out_dense_3")(x)
    x = keras.layers.Dense(units=126*4, activation=tf.keras.activations.linear, name="output")(x)
    x = keras.layers.Reshape((4,126))(x)
    model = keras.Model(inputs=model_.layers[0].output, outputs=x)

    print(model.summary())

    model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.mse)

    my_callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, monitor="loss"),
        tf.keras.callbacks.ModelCheckpoint(filepath='checkpoints/YaoQihang/model.{epoch:02d}-{loss:.2f}.h5', monitor="loss", save_best_only=True),
        tf.keras.callbacks.TensorBoard(log_dir='./logs_YaoQihang'),
        tf.keras.callbacks.CSVLogger('log_YaoQihang.csv', append=True, separator=';')
    ]

elif sys.argv[1]=="YiboGao":
    model_ = models_keras.YiboGao(include_top=False, input_shape=(INSTANCE_BATCH, 126))

    x = keras.layers.Flatten()(model_.layers[-1].output)
    x = keras.layers.Dense(units=32, activation=tf.keras.activations.linear, name="out_dense_1")(x)
    x = keras.layers.Dense(units=64, activation=tf.keras.activations.linear, name="out_dense_2")(x)
    x = keras.layers.Dense(units=126, activation=tf.keras.activations.linear, name="out_dense_3")(x)
    x = keras.layers.Dense(units=126*4, activation=tf.keras.activations.linear, name="output")(x)
    x = keras.layers.Reshape((4,126))(x)
    model = keras.Model(inputs=model_.layers[0].output, outputs=x)

    print(model.summary())

    model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.mse)

    my_callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, monitor="loss"),
        tf.keras.callbacks.ModelCheckpoint(filepath='checkpoints/YiboGao/model.{epoch:02d}-{loss:.2f}.h5', monitor="loss", save_best_only=True),
        tf.keras.callbacks.TensorBoard(log_dir='./logs_YiboGao'),
        tf.keras.callbacks.CSVLogger('log_YiboGao.csv', append=True, separator=';')
    ]

history = model.fit(train_generator,
                    batch_size=BATCH_SIZE,
                    epochs=150,
                    callbacks=my_callbacks)
y_pred = model.predict(test_generator)
y_pred = y_pred.reshape(y_pred.shape[0]*y_pred.shape[1], y_pred.shape[2])
#y_test = y_test.reshape(y_test.shape[0]*y_test.shape[1], y_test.shape[2])
y_test = data[-500000:]
y_test = y_test[INSTANCE_BATCH:len(y_pred)+INSTANCE_BATCH]
classes_test = classes_test[INSTANCE_BATCH:len(y_pred)+INSTANCE_BATCH].astype("int")
scores = utils.computeScore(y_test, y_pred)
anomalies_num = np.sum(classes_test)
anomalies = scores.argsort()[-anomalies_num:][::-1]
anomalies_labels = np.zeros(len(classes_test))
anomalies_labels[anomalies]=1

accuracy = sklearn.metrics.accuracy_score(classes_test, anomalies_labels)
auc = sklearn.metrics.roc_auc_score(classes_test, scores)
f1 = sklearn.metrics.f1_score(classes_test, anomalies_labels)
precision = sklearn.metrics.precision_score(classes_test, anomalies_labels)
recall = sklearn.metrics.recall_score(classes_test, anomalies_labels)

f = open("metrics_results/" + sys.argv[1] + "_metrics.txt", "w+")

f.write("Accuracy: " + str(accuracy) + "\n")
f.write("AUC: " + str(auc) + "\n")
f.write("F1: " + str(f1) + "\n")
f.write("Precision: " + str(precision) + "\n")
f.write("Recall: " + str(recall) + "\n")

f.close()
