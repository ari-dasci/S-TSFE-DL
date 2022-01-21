import obspy
import numpy as np
import tensorflow as tf
import TSFEDL.models_keras as TSFEDL
import csv
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--epochs', type=int, default=20, help="Number of max epochs to run.")
parser.add_argument('--input_size', type=int, default=4000, help="Size of the input sequence for each instance.")
parser.add_argument('--output_size', type=int, default=100, help="Size of the predicted output sequence.")

args = parser.parse_args()

def compile_and_fit(model, window, MAX_EPOCHS, patience=2):
    """
    It performs the training of the given model

    """
    # We use early stopping and model checkpoint to store the vest model found so far.
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')

    mc = tf.keras.callbacks.ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True)

    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=[tf.metrics.MeanAbsoluteError()])

    history = model.fit(window.train, epochs=MAX_EPOCHS,
                        validation_data=window.val,
                        callbacks=[early_stopping, mc])
    return history


def generateNewModel(input_tensor):
    """
    It generates a new model using a given model plus an additional building block in the TSFEDL library.

    Returns
    -------
    The new model
    """
    model = TSFEDL.OhShuLih(input_tensor=input_tensor, include_top=False)
    lstm_layer = model.layers[-1]  # get the LSTM
    x = model.layers[-2].output
    x = TSFEDL.RTA_block(x, nb_filter=6, kernel_size=3)
    x = lstm_layer(x)


    new_model = tf.keras.Model(inputs=input, outputs=x)
    return new_model


class WindowGenerator():
    """
    Helper class that generates instances as splits of the fiven time series.
    This is the same class as the example provided in:

    https://www.tensorflow.org/tutorials/structured_data/time_series
    """
    def __init__(self, input_width, label_width, shift,
                 train_df, val_df, test_df, batch_size=32,
                 label_columns=None):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.batch_size = batch_size

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                          enumerate(label_columns)}
        # self.column_indices = {name: i for i, name in
        # enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def split_window_no_labels(self, features):
        inputs = features[:, self.input_slice, :]

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])

        return inputs, inputs

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=self.batch_size, )

        ds = ds.map(self.split_window)

        return ds

    def make_dataset_no_labels(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=self.batch_size, )

        ds = ds.map(self.split_window_no_labels)

        return ds

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def train_autoencoder(self):
        return self.make_dataset_no_labels(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)


def moving_average(x, w):
    """
    It performs the moving average over the time series
    Args:
        x: the time series
        w: the size of the window

    Returns:
        The preprocessed time series
    """
    return np.convolve(x, np.ones(w), 'valid') / w


# Read the data
data = obspy.read("ES.EADA..HHE.D.2021.002")
trace = data[0]
data = trace.data

# Preprocess data: Smooth the line by applying moving average several times
w = 50                              # Moving average window size
moving_avg_application_times = 5    # No. of times that moving average is applied.
data_smooth = data
for i in range(moving_avg_application_times):
    data_smooth = moving_average(data_smooth, w)

# Split data into train-test, we have to take the last N timesteps as test data.
num_samples = len(data_smooth)
data_smooth = data_smooth.reshape((-1, 1))
train_data = data_smooth[:int(num_samples * 0.7), :]
val_data = data_smooth[int(num_samples * 0.7):int(num_samples * 0.9), :]
test_data = data_smooth[int(num_samples * 0.9):, :]

# We need to generate batches of slices of the original time series.
# The predictions made on each instance are a set of future points:
# This number of future points are the OUTPUT_STEPS value. INPUT_LENGTH is the length of the training data
INPUT_LENGTH = args.input_size  # Input must be at least 3600 or more.
OUTPUT_STEPS = args.output_size
w1 = WindowGenerator(input_width=INPUT_LENGTH,
                     label_width=OUTPUT_STEPS,
                     shift=OUTPUT_STEPS,
                     train_df=train_data,
                     val_df=val_data,
                     test_df=test_data,
                     batch_size=256,
                     label_columns=[0])

# We set a second window of length 3600 for models that require at least 3600 timesteps
w2 = None
if INPUT_LENGTH < 3600:
    w2 = WindowGenerator(input_width=3600,
                         label_width=OUTPUT_STEPS,
                         shift=OUTPUT_STEPS,
                         train_df=train_data,
                         val_df=val_data,
                         test_df=test_data,
                         batch_size=256,
                         label_columns=[0])

# Train the model using the different methods of the framework
input = tf.keras.Input(shape=(INPUT_LENGTH, 1))
input2 = tf.keras.Input(shape=(3600, 1)) if w2 is not None else input
num_epochs = args.epochs

# Train all models
methods_dict = {
    'OhShuLi' : TSFEDL.OhShuLih(input_tensor=input, include_top=False),
    'ChenChen': TSFEDL.ChenChen(input_tensor=input2, include_top=False),
    'YildirimOzal': TSFEDL.YildirimOzal(input_tensor=input, include_top=False),
    'LihOhShu': TSFEDL.LihOhShu(input_tensor=input2, include_top=False),
    'ShiHaotian': TSFEDL.ShiHaotian(input_tensor=input, include_top=False),
    'KhanZulfiqar': TSFEDL.KhanZulfiqar(input_tensor=input, include_top=False),
    'ZhengZhenyu' : TSFEDL.ZhengZhenyu(input_tensor=input, include_top=False),
    'WangKejun': TSFEDL.WangKejun(input_tensor=input, include_top=False),
    'KimTaeYoung': TSFEDL.KimTaeYoung(input_tensor=input, include_top=False),
    'GenMinxing': TSFEDL.GenMinxing(input_tensor=input, include_top=False),
    'FuJiangmeng': TSFEDL.FuJiangmeng(input_tensor=input, include_top=False),
    'HuangMeiLing': TSFEDL.HuangMeiLing(input_tensor=input, include_top=False),
    'GaoJunLi': TSFEDL.GaoJunLi(input_tensor=input, include_top=False),
    'WeiXiaoyan': TSFEDL.WeiXiaoyan(input_tensor=input, include_top=False),
    'KongZhengmin': TSFEDL.KongZhengmin(input_tensor=input, include_top=False),
    'CaiWenjuan': TSFEDL.CaiWenjuan(input_tensor=input, include_top=False),
    'HtetMyetLynn': TSFEDL.HtetMyetLynn(input_tensor=input, include_top=False),
    'ZhangJin': TSFEDL.ZhangJin(input_tensor=input, include_top=False),
    'YaoQihang': TSFEDL.YaoQihang(input_tensor=input, include_top=False),
    'YiboGao': TSFEDL.YiboGao(input_tensor=input, include_top=False, return_loss=False),
    'NewModel': generateNewModel(input_tensor=input)
}

# These models require an input with length != 1000
special_input_models = ["ChenChen", "LihOhShu"]

# Here we store the results:
results = {}

# Now for each method: train and test
for name, model in methods_dict.items():
    print("Training", name)
    # Linear output for each predicted timestep: This is the top module of each method.
    linear_model = tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(OUTPUT_STEPS, kernel_initializer=tf.initializers.zeros()),
            tf.keras.layers.Reshape([OUTPUT_STEPS, 1])

        ]
    )

    if name == "YildirimOzal":
        # First, train the autoencoder
        ae, enc, classifier = model
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                          patience=2,
                                                          mode='min')
        mc = tf.keras.callbacks.ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True)

        ae.compile(loss=tf.losses.MeanSquaredError(),
                      optimizer=tf.optimizers.Adam(),
                      metrics=[tf.metrics.MeanAbsoluteError()])

        ae.fit(w1.train_autoencoder, epochs=20)

        # Next, train the LSTM and the classifier with the encoded features.
        enc.trainable = False
        ae.trainable = False
        x = classifier.layers[-1].output
        out = linear_model(x)
        model2 = tf.keras.Model(inputs=input, outputs=out)
    else:
        # Conect the model output to the top module  and create the new model
        x = model.output
        out = linear_model(x)
        model2 = tf.keras.Model(inputs=input, outputs=out) if name not in special_input_models else tf.keras.Model(inputs=input2, outputs=out)

    # Training and test the model, and then store the results.
    res = compile_and_fit(model2, w1, MAX_EPOCHS=num_epochs) if name not in special_input_models else compile_and_fit(model2, w2, MAX_EPOCHS=num_epochs)
    saved_model = tf.keras.models.load_model('best_model.h5')
    eval = saved_model.evaluate(w1.test, verbose=0) if name not in special_input_models else saved_model.evaluate(w2.test, verbose=0)
    print("MAE:", eval[1])
    results[name] = eval[1]

# write results to file
with open('results_TSFEDL_IGN.csv', 'w') as f:
    writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for i in results.items():
        writer.writerow(i)

print("DONE.")
