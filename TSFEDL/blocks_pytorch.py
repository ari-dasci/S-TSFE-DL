import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlockYiboGao(nn.Module):
    """
    Convolutional block of YiboGao's model

    Parameters
    ----------
    in_features : int
        Number of input features.
    nb_filter : int
        Number of filters for the convolution.
    kernel_size : int
        Size of the convolution kernel.
    """
    def __init__(self, in_features, nb_filter, kernel_size):
        super(ConvBlockYiboGao, self).__init__()
        self.kernel_size = kernel_size
        self.nb_filter = nb_filter
        self.in_features = in_features
        self.out_features = nb_filter

        self.module = nn.Sequential(
            nn.Conv1d(in_channels=in_features, out_channels=nb_filter, kernel_size=kernel_size, padding='same'),
            nn.BatchNorm1d(nb_filter),
            nn.ReLU()
        )

    """
    Forward function of the model. This function returns the operation of the
    neural network over data.

    Parameters
    ----------
    x : array-like
        Data to operate with.

    Returns
    -------
    x : torch.Tensor
        Result of the operation with the neural network.
    """
    def forward(self, x):
        return self.module(x)


class AttentionBranchYiboGao(nn.Module):
    """
    Attention branch of YiboGao's model

    Parameters
    ----------
    in_features : int
        Number of input features.
    nb_filter : int
        Number of filters for the convolution.
    kernel_size : int
        Size of the convolution kernel.
    """

    def __init__(self, in_features, nb_filter, kernel_size):
        super(AttentionBranchYiboGao, self).__init__()
        self.nb_filter = nb_filter
        self.kernel_size = kernel_size
        self.in_features = in_features
        self.out_features = nb_filter

        self.convBlock1 = ConvBlockYiboGao(in_features, nb_filter, kernel_size)  # nb_filter is the number of features
        self.convBlock2 = nn.Sequential(
            nn.MaxPool1d(kernel_size=2),
            ConvBlockYiboGao(in_features=nb_filter, nb_filter=nb_filter, kernel_size=kernel_size),
            nn.Upsample(scale_factor=2)
        )
        self.convBlock3 = ConvBlockYiboGao(nb_filter, nb_filter, kernel_size)
        self.finalBlock = nn.Sequential(
            ConvBlockYiboGao(nb_filter, nb_filter, kernel_size),
            nn.Conv1d(in_channels=nb_filter, out_channels=nb_filter, kernel_size=1, padding='same'),
            nn.BatchNorm1d(nb_filter),
            nn.Sigmoid()
        )

    """
    Forward function of the model. This function returns the operation of the
    neural network over data.

    Parameters
    ----------
    x : array-like
        Data to operate with.

    Returns
    -------
    x : torch.Tensor
        Result of the operation with the neural network.
    """
    def forward(self, x):
        x1 = self.convBlock1(x)
        x = self.convBlock2(x1)
        x2 = self.convBlock3(x)

        if(x1.size() != x2.size()):
            # For odd input sizes, the convolutions could produce differences of length 1 in sizes. Pad with zeroes
            x2 = F.pad(x2, pad=(0, 1))

        x = torch.add(x1, x2)

        return self.finalBlock(x)


class RTABlock(nn.Module):
    """
    Residual-based Temporal Attention (RTA) block.

    References
    ----------
        Gao, Y., Wang, H., & Liu, Z. (2021). An end-to-end atrial fibrillation detection by a novel residual-based
        temporal attention convolutional neural network with exponential nonlinearity loss.
        Knowledge-Based Systems, 212, 106589.

    Parameters
    ----------
    in_features : int
        Number of input features.
    nb_filter : int
        Number of filters for the convolution.
    kernel_size : int
        Size of the convolution kernel.
    """
    def __init__(self, in_features, nb_filter, kernel_size):
        super(RTABlock, self).__init__()
        self.nb_filter = nb_filter
        self.kernel_size = kernel_size
        self.in_features = in_features

        self.conv1 = ConvBlockYiboGao(in_features, nb_filter, kernel_size)
        self.conv2 = ConvBlockYiboGao(nb_filter, nb_filter, kernel_size)
        self.attention = AttentionBranchYiboGao(nb_filter, nb_filter, kernel_size)
        self.conv3 = ConvBlockYiboGao(nb_filter, nb_filter, kernel_size)

    """
    Forward function of the model. This function returns the operation of the
    neural network over data.

    Parameters
    ----------
    x : array-like
        Data to operate with.

    Returns
    -------
    x : torch.Tensor
        Result of the operation with the neural network.
    """
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        attention_map = self.attention(x1)

        x = torch.multiply(x2, attention_map)
        x = torch.add(x, x1)

        return self.conv3(x)


class SqueezeAndExcitationModule(nn.Module):
    """
    Squeeze-and-Excitation Module.

    References
    ----------
     Squeeze-and-Excitation Networks, Jie Hu, Li Shen, Samuel Albanie, Gang Sun, Enhua Wu (arXiv:1709.01507v4)

    Parameters
    ---------
      in_features: int
        The number of input features (channels)

      dense_units: int
        The number units on each dense layer.

    Returns
    -------
    se: torch.Tensor
      Output tensor for the block.
    """
    def __init__(self, in_features: int, dense_units: int):
        super(SqueezeAndExcitationModule, self).__init__()
        self.dense_units = dense_units
        self.in_features = in_features

        self.fully_connected = nn.Sequential(
            nn.Linear(in_features, dense_units),
            nn.ReLU(),
            nn.Linear(dense_units, in_features),
            nn.Sigmoid()
        )

    """
    Forward function of the model. This function returns the operation of the
    neural network over data.

    Parameters
    ----------
    x : array-like
        Data to operate with.

    Returns
    -------
    se : torch.Tensor
        Result of the operation with the neural network.
    """
    def forward(self, x):  # x shape is (N, C, L) N -> num_batch, C -> channels, L -> length of sequence
        # Global average pooling is just getting the average value of each channel
        se = torch.mean(x, dim=2)

        # reshape se to match x's shape
        se = se.view((se.size(0), 1, se.size(1)))  # change to (N, *, H_in) format
        se = self.fully_connected(se)
        se = se.view((se.size(0), se.size(2), 1))  # Go back to the (N, C, L) format to continue the convolutions.

        # element-wise multiplication between inputs and squeeze
        se = torch.multiply(x, se)
        return se


class DenseNetTransitionBlock(nn.Module):
    """
    Densenet Transition Block for CaiWenjuan model.

    Parameters
    ---------
      in_features: int
        The number of input features (channels)

      reduction: float
        Number between 0 and 1 representing the percentage reduction on the number
        of units.
    """
    def __init__(self, in_features, reduction):
        super(DenseNetTransitionBlock, self).__init__()
        self.out_features = int(in_features * reduction)
        self.module = nn.Sequential(
            nn.BatchNorm1d(num_features=in_features, eps=1.001e-5),
            nn.ReLU(),
            nn.Conv1d(in_channels=in_features, out_channels=int(in_features * reduction), kernel_size=1, bias=False),
            nn.AvgPool1d(kernel_size=2, stride=2)
        )

    """
    Forward function of the model. This function returns the operation of the
    neural network over data.

    Parameters
    ----------
    x : array-like
        Data to operate with.

    Returns
    -------
    se : torch.Tensor
        Result of the operation with the neural network.
    """
    def forward(self, x):
        return self.module(x)


class DenseNetConvBlock(nn.Module):
    """
    Densenet convolution block.

    Parameters
    ---------
      in_features: int
        The number of input features (channels)

      growth_rate: int
        Growth rate of the number of units in the layers.
    """
    def __init__(self, in_features, growth_rate):
        super(DenseNetConvBlock, self).__init__()
        self.module = nn.Sequential(
            nn.BatchNorm1d(num_features=in_features, eps=1.001e-5),
            nn.ReLU(),
            nn.Conv1d(in_channels=in_features, out_channels=4 * growth_rate, kernel_size=1, bias=False),
            nn.BatchNorm1d(num_features=4 * growth_rate),
            nn.ReLU(),
            nn.Conv1d(in_channels=4 * growth_rate, out_channels=growth_rate, kernel_size=3, padding='same', bias=False)
        )

    """
    Forward function of the model. This function returns the operation of the
    neural network over data.

    Parameters
    ----------
    x : array-like
        Data to operate with.

    Returns
    -------
    se : torch.Tensor
        Result of the operation with the neural network.
    """
    def forward(self, x):
        x1 = self.module(x)
        return torch.cat((x1, x), dim=1)


class DenseNetDenseBlock(nn.Module):
    """
    Densenet dense block.

    Parameters
    ---------
    in_features: int
        The number of input features (channels)

    layers : int
        Number of layers of the block.

    growth_rate: int
        Growth rate of the number of units in the layers.
    """
    def __init__(self, in_features, layers, growth_rate):
        super(DenseNetDenseBlock, self).__init__()
        self.output_features = in_features + layers * growth_rate
        self.module = nn.ModuleList(
            [DenseNetConvBlock(in_features=in_features + i * growth_rate, growth_rate=growth_rate)
             for i in range(layers)]
        )

    """
    Forward function of the model. This function returns the operation of the
    neural network over data.

    Parameters
    ----------
    x : array-like
        Data to operate with.

    Returns
    -------
    se : torch.Tensor
        Result of the operation with the neural network.
    """
    def forward(self, x):
        xs = x
        for layer in self.module:
            xs = layer(xs)
        return xs


class SpatialAttentionBlockZhangJin(nn.Module):
    """
    Spatial Attention module of ZhangJin's model.

    Parameters
    ---------
    in_features: int
        The number of input features (channels).

    decrease_ratio: int
        Decrease rate of the number of units in the layers.
    """
    def __init__(self, in_features, decrease_ratio):
        super(SpatialAttentionBlockZhangJin, self).__init__()
        self.shared_dense = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=in_features // decrease_ratio),
            nn.ReLU(),
            nn.Linear(in_features=in_features // decrease_ratio, out_features=in_features),
            nn.ReLU()
        )

    """
    Forward function of the model. This function returns the operation of the
    neural network over data.

    Parameters
    ----------
    x : array-like
        Data to operate with.

    Returns
    -------
    se : torch.Tensor
        Result of the operation with the neural network.
    """
    def forward(self, x):
        # 1º Global average pooling for each feature
        x1 = torch.mean(x, dim=2)
        x1 = self.shared_dense(x1)
        x1 = x1.view((x1.size(0), x1.size(1), 1))

        # 2º Global max pooling
        x2 = torch.max(x, dim=2).values
        x2 = self.shared_dense(x2)
        x2 = x2.view((x2.size(0), x2.size(1), 1))

        result = F.sigmoid(torch.add(x1, x2))
        return result


class TemporalAttentionBlockZhangJin(nn.Module):
    """
    Temporal attention module of ZhangJin's Model.
    """
    def __init__(self):
        super(TemporalAttentionBlockZhangJin, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=7, padding='same')

    """
    Forward function of the model. This function returns the operation of the
    neural network over data.

    Parameters
    ----------
    x : array-like
        Data to operate with.

    Returns
    -------
    se : torch.Tensor
        Result of the operation with the neural network.
    """
    def forward(self, x):
        # Global max pooling for each TIMESTEP
        x1 = torch.max(x, dim=1).values
        x1 = x1.view((x1.size(0), 1, x1.size(1)))

        x2 = torch.mean(x, dim=1)
        x2 = x2.view((x2.size(0), 1, x2.size(1)))

        x = torch.cat((x1, x2), dim=1)
        x = self.conv1(x)
        x = F.sigmoid(x)

        return x


#
# def temporal_attention_block_ZhangJin(x):
#     """
#     Temporal attention module of ZhangJin's Model.
#     """
#     # Temporal attention module
#     x1 = layers.GlobalMaxPool1D(data_format='channels_first')(x)
#     x1 = layers.Reshape(target_shape=(x1.shape[1], 1))(x1)
#
#     x2 = layers.GlobalAveragePooling1D(data_format='channels_first')(x)
#     x2 = layers.Reshape(target_shape=(x2.shape[1], 1))(x2)
#
#     x = layers.Concatenate()([x1, x2])
#     x = layers.Conv1D(filters=1, kernel_size=7, padding="same")(x)
#     x = layers.Activation(activation=sigmoid)(x)
#     return x
