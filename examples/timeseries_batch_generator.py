import numpy as np
import tensorflow.keras as keras
import threading

class DataGenerator(keras.utils.Sequence):
    def __init__(self, data, targets, length,
                 sampling_rate=1,
                 stride=1,
                 start_index=0,
                 end_index=None,
                 shuffle=False,
                 reverse=False,
                 batch_size=128):

        if len(data) != len(targets):
            raise ValueError('Data and targets have to be' +
                             ' of same length. '
                             'Data length is {}'.format(len(data)) +
                             ' while target length is {}'.format(len(targets)))

        self.data = data
        self.targets = targets
        self.length = length
        self.sampling_rate = sampling_rate
        self.stride = stride
        self.start_index = start_index + length
        if end_index is None:
            end_index = len(data) - 1
        self.end_index = end_index
        self.shuffle = shuffle
        self.reverse = reverse
        self.batch_size = batch_size

        self.lock = threading.Lock()

        if self.start_index > self.end_index:
            raise ValueError('`start_index+length=%i > end_index=%i` '
                             'is disallowed, as no part of the sequence '
                             'would be left to be used as current step.'
                             % (self.start_index, self.end_index))

    def __len__(self):
        s =  ((self.end_index - self.start_index +
                self.batch_size * self.stride) // (self.batch_size * self.stride))
        if self.length*self.batch_size*s+self.stride*self.batch_size*s>len(self.data):
            s-=1
        return s

    def __getitem__(self, index):
        if self.shuffle:
            rows = np.random.randint(
                self.start_index, self.end_index + 1, size=self.batch_size)
        else:
            i = self.start_index + self.batch_size * self.stride * index
            rows = np.arange(i, min(i + self.batch_size *
                                    self.stride, self.end_index + 1), self.stride)

        samples = np.array([self.data[row - self.length:row:self.sampling_rate]
                            for row in rows])
        if samples.size>0:
            targets = np.array([self.targets[row:row+self.stride] for row in rows])
        else:
            targets=samples

        with self.lock:
            if self.reverse:
                yield samples[:, ::-1, ...], targets
            yield samples, targets

    def get_config(self):
        '''Returns the TimeseriesGenerator configuration as Python dictionary.

        # Returns
            A Python dictionary with the TimeseriesGenerator configuration.
        '''
        data = self.data
        if type(self.data).__module__ == np.__name__:
            data = self.data.tolist()
        try:
            json_data = json.dumps(data)
        except TypeError:
            raise TypeError('Data not JSON Serializable:', data)

        targets = self.targets
        if type(self.targets).__module__ == np.__name__:
            targets = self.targets.tolist()
        try:
            json_targets = json.dumps(targets)
        except TypeError:
            raise TypeError('Targets not JSON Serializable:', targets)

        return {
            'data': json_data,
            'targets': json_targets,
            'length': self.length,
            'sampling_rate': self.sampling_rate,
            'stride': self.stride,
            'start_index': self.start_index,
            'end_index': self.end_index,
            'shuffle': self.shuffle,
            'reverse': self.reverse,
            'batch_size': self.batch_size
        }

    def to_json(self, **kwargs):
        """Returns a JSON string containing the timeseries generator
        configuration. To load a generator from a JSON string, use
        `keras.preprocessing.sequence.timeseries_generator_from_json(json_string)`.

        # Arguments
            **kwargs: Additional keyword arguments
                to be passed to `json.dumps()`.

        # Returns
            A JSON string containing the tokenizer configuration.
        """
        config = self.get_config()
        timeseries_generator_config = {
            'class_name': self.__class__.__name__,
            'config': config
        }
        return json.dumps(timeseries_generator_config, **kwargs)


def get_indices_change(data, event_codes_name):
    '''Return indices of the data where there is change between
    events

    Args:
        data (pd.DataFrame): data with events.
        event_codes_name (str): column name with unique code for each interval
            between failures.

    Returns:
        indices_change (list, int): indices where there is a new failure.
    '''
    data = data.reset_index(drop=True).copy()
    data['prev_code'] = data[event_codes_name].shift(1)
    data['flag_change'] = data['prev_code'] != data[event_codes_name]
    indices_change = np.array(data.loc[data['flag_change'], :].index)
    return indices_change

class TimeseriesGenerator_Multistep(keras.utils.Sequence):
    """Modification of the TimeseriesGenerator from Keras, to
    be adapted to the case when output sequence has length bigger
    than 1.
    This class takes in a sequence of data-points gathered at
    equal intervals, along with time series parameters such as
    stride, length of history, etc., to produce batches for
    training/validation.
    # Arguments
        data: Indexable generator (such as list or Numpy array)
            containing consecutive data points (timesteps).
            The data should be at 2D, and axis 0 is expected
            to be the time dimension.
        targets: Targets corresponding to timesteps in `data`.
            It should have same length as `data`.
        length: Length of the output sequences (in number of timesteps).
        length_target: Length of the target sequence (in number of timesteps).
        filter_indices: indices that should not be contained in any sequence.
        sampling_rate: Period between successive individual timesteps
            within sequences. For rate `r`, timesteps
            `data[i]`, `data[i-r]`, ... `data[i - length]`
            are used for create a sample sequence.
        stride: Period between successive output sequences.
            For stride `s`, consecutive output samples would
            be centered around `data[i]`, `data[i+s]`, `data[i+2*s]`, etc.
        start_index: Data points earlier than `start_index` will not be used
            in the output sequences. This is useful to reserve part of the
            data for test or validation.
        end_index: Data points later than `end_index` will not be used
            in the output sequences. This is useful to reserve part of the
            data for test or validation.
        shuffle: Whether to shuffle output samples,
            or instead draw them in chronological order.
        reverse: Boolean: if `true`, timesteps in each output sample will be
            in reverse chronological order.
        batch_size: Number of timeseries samples in each batch
            (except maybe the last one).
        behaviour: if they will be used with autoencoders ('autoencoder')
            or forecasting ('forecast')
    # Returns
        A [Sequence](/utils/#sequence) instance.
    """

    def __init__(self, data, targets, length,
                 length_target,
                 filter_indices=None,
                 sampling_rate=1,
                 stride=1,
                 start_index=0,
                 end_index=None,
                 shuffle=False,
                 reverse=False,
                 batch_size=128,
                 behaviour='autoencoder'):

        if len(data) != len(targets):
            raise ValueError('Data and targets have to be' +
                             ' of same length. '
                             'Data length is {}'.format(len(data)) +
                             ' while target length is {}'.format(len(targets)))

        self.data = data
        self.targets = targets
        self.length = length
        if behaviour == 'autoencoder':
            self.length_target = 0
        else:
            self.length_target = length_target
        self.filter_indices = None if filter_indices==None else filter_indices.copy()
        self.sampling_rate = sampling_rate
        self.stride = stride
        self.start_index = start_index + length
        if end_index is None:
            end_index = len(data) - 1
        self.end_index = end_index - self.length_target
        self.shuffle = shuffle
        self.reverse = reverse
        self.batch_size = batch_size
        self.behaviour = behaviour

        self.lock = threading.Lock()

        if self.start_index > self.end_index:
            raise ValueError('`start_index+length=%i > end_index=%i` '
                             'is disallowed, as no part of the sequence '
                             'would be left to be used as current step.'
                             % (self.start_index, self.end_index))

        available_indices = np.arange(self.start_index, self.end_index + 1)
        if self.filter_indices is not None:
            self.filter_indices -= self.length
            extended_indices = self._extend_indices()
            available_indices = np.delete(available_indices, extended_indices)
        self.available_indices = available_indices

    def _extend_indices(self):
        '''
        Extend the indices affected by temporal jumps, based on length and length_target parameters. This list
        will be used when initialized the class to remove all those indices from data that could be affected
        by temporal jumps.
        Example: for length=3, length_target=2, and index=15, extended indices would be: [14, 15, 16, 17, 18].
            - index is used as the point between input sequence and output sequence (see _get_item() for
                details)
            - 14 and 15 discarded because it would generate output sequences comprising jump (e.g. for 14,
                output would be 15 and 16)
            - 16, 17, and 18 discarded because it would generate input sequences comprising jump (e.g. for
                17, input would be 14, 15, and 16).

        Returns:
            extended_indices (list, int): list with all indices affected by temporal jumps
        '''
        extended_indices = np.concatenate([np.arange(i - self.length_target + 1, i + self.length + 1) for i in self.filter_indices])
        extended_indices = extended_indices[extended_indices > 0]
        return extended_indices

    def __len__(self):
        '''
        Provides the length of the generator, i.e. the total number of batches available.

        Returns:
            n_batches (int): number of batches the generator can yield.
        '''
        n_obs_per_batch = self.batch_size * self.stride
        n_batches = int(np.ceil((len(self.available_indices) - n_obs_per_batch) / n_obs_per_batch))
        return n_batches

    def __getitem__(self, index):
        '''
        Returns a batch with sequences to be used, according to the specified batch_size.
        Depending on the self.shuffle argument, it will provide sequences picked randomly or the whole
        data split in batches.

        Returns:
            samples (np.array, float): three dimensional array with input sequences
            targets (np.array, float): three dimensional array with output sequences.
        '''
        with self.lock:
            if self.shuffle:
                rows = np.random.choice(
                    self.available_indices, size=self.batch_size, replace=False)
            else:
                i = self.batch_size * self.stride * index
                rows_indices = np.arange(i, min(i + self.batch_size *
                                        self.stride, len(self.available_indices)), self.stride)
                rows = self.available_indices[rows_indices]

            samples = np.array([self.data[row - self.length:row:self.sampling_rate]
                                for row in rows])
            if self.behaviour == 'autoencoder':
                targets = samples.copy()
            else:
                targets = np.array([self.targets[row:row + self.length_target:self.sampling_rate]
                                    for row in rows])

            if self.reverse:
                return samples[:, ::-1, ...], targets
            return samples, targets


# Examplification of use: not reproducible
# n_input = 128
# n_output = 5
# batch_size = 500
# indices_change_train = get_indices_change(data_train, event_codes_name='aux_codes')
# train_generator = TimeseriesGenerator_Multistep(X_train, X_train,
#                                                 length=n_input, length_target=n_output,
#                                                 filter_indices=indices_change_train, sampling_rate=1, stride=1,
#                                                 batch_size=batch_size, shuffle=True)
# Note that data_train should be a dataframe with id variables like aux_codes, while X_train should be an array only with features.
