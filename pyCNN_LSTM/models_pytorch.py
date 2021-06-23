import os
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from typing import Callable, Optional, List, Tuple
from data import MIT_BIH
from torch.utils.data.sampler import SubsetRandomSampler
from blocks_pytorch import RTABlock, SqueezeAndExcitationModule, DenseNetDenseBlock, DenseNetTransitionBlock, \
    SpatialAttentionBlockZhangJin, TemporalAttentionBlockZhangJin
from pyCNN_LSTM.utils import flip_indices_for_conv_to_lstm


class pyCNN_LSTM_BaseModule(pl.LightningModule):
    """
    Base module for any pyTorch Lightning based algorithm in this library

    Parameters
    ----------
        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
            The loss function to use. It should accept two Tensors as inputs (predictions, targets) and return
            a Tensor with the loss.


        top_module: nn.Module, defaults=None
            The optional nn.Module to be used as additional top layers.

        optimizer:  torch.optim.Optimizer
            The pyTorch Optimizer to use. Note that this must be only the class type and not an instance of the class!!

        **kwargs: dict
            A dictionary with the parameters of the optimizer.
    """

    def __init__(self,
                 in_features: int,
                 top_module: Optional[nn.Module] = None,
                 loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = F.nll_loss,
                 optimizer: torch.optim.Optimizer = torch.optim.Adam,
                 **kwargs
                 ):
        super(pyCNN_LSTM_BaseModule, self).__init__()
        self.kwargs = kwargs
        self.in_features = in_features
        self.loss = loss
        self.optimizer = optimizer
        self.classifier = top_module
        pass

    def forward(self, x):
        pass

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        opt = self.optimizer(self.parameters(), **self.kwargs)
        return opt


class OhShuLih_Classifier(nn.Module):
    """
    Classifier of the OhShuLi model.
    """

    def __init__(self, in_features, n_classes):
        super(OhShuLih_Classifier, self).__init__()
        self.model = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=in_features, out_features=20),
            nn.ReLU(),
            nn.Linear(in_features=20, out_features=10),
            nn.ReLU(),
            nn.Linear(in_features=10, out_features=n_classes),
            nn.Softmax()
        )

    def forward(self, x):
        return self.model(x)


class OhShuLih(pyCNN_LSTM_BaseModule):
    """
    CNN+LSTM model for Arrythmia classification.

    Parameters
    ----------
        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
            The loss function to use. It should accept two Tensors as inputs (predictions, targets) and return
            a Tensor with the loss.


        top_module: nn.Module, defaults=OhShuLih_Classifier(5)
            The optional  nn.Module to be used as additional top layers.

        optimizer:  torch.optim.Optimizer
            The pyTorch Optimizer to use. Note that this must be only the class type and not an instance of the class!!

        **kwargs: dict
            A dictionary with the parameters of the optimizer.

    Returns
    -------
    `LightningModule`
        A pyTorch Lightning Module instance.

    References
    ----------
        `Oh, Shu Lih, et al. "Automated diagnosis of arrhythmia using combination of CNN and LSTM techniques with
        variable length heart beats." Computers in biology and medicine 102 (2018): 278-287.`
    """
    def __init__(self,
                 in_features: int,
                 top_module: Optional[nn.Module] = OhShuLih_Classifier(in_features=20, n_classes=5),
                 loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = F.nll_loss,
                 optimizer: torch.optim.Optimizer = torch.optim.Adam,
                 **kwargs
                 ):
        super(OhShuLih, self).__init__(in_features, top_module, loss, optimizer, **kwargs)

        conv_layers = []

        # The padding of (kernel_size - 1) is for full convolution.
        # We use a Lazy Conv1D layer here as we dont know the input channels beforehand.
        # NOTE: INPUT SHAPE MUST BE (N, C, L) where N is the batch size, C is the NUMBER OF CHANNEL (as opposite to keras)
        # and L is the number of timesteps of Length of the 1D signal.
        conv_layers.append(nn.Conv1d(in_channels=in_features, out_channels=3, kernel_size=20, bias=False, stride=1, padding=19))
        conv_layers.append(nn.ReLU())
        conv_layers.append(nn.MaxPool1d(kernel_size=2))

        # The remaining convolutional layers can be normal ones: we know the input size.
        conv_layers.append(nn.Conv1d(in_channels=3, out_channels=6, kernel_size=10, bias=False, padding=9))
        conv_layers.append(nn.ReLU())
        conv_layers.append(nn.MaxPool1d(kernel_size=2))

        conv_layers.append(nn.Conv1d(in_channels=6, out_channels=6, kernel_size=5, bias=False, padding=4))
        conv_layers.append(nn.ReLU())
        conv_layers.append(nn.MaxPool1d(kernel_size=2))

        self.convolutions = nn.Sequential(*conv_layers)

        # input_size is the number of DIMENSIONS of the data.
        # From the convolution above, the number of output channels is 6, so we have 6 dimensions in our data.
        # NOTE: In keras, this LSTM layer employs recurrent dropout. THIS IS NOT IMPLEMENTED IN PYTORCH!!
        # NOTE2: this LSTM Requires an input with shape (N, L, H_in) where N is the batch size, L is the LENGTH of the sequence
        # and H_in is the number of dimensions. From the convolutional layers we got a shape of (N, H_in, L), so
        # we have to RESHAPE our BEFORE plugging them into the LSTM.
        self.lstm = nn.LSTM(input_size=6, hidden_size=20, batch_first=True)


    def forward(self, x):
        out = self.convolutions(x)
        # Now, flip indices using a view for the LSTM as it requires a shape of (N, L, H_in = C)
        out = out.view(out.size(0), out.size(2), out.size(1))
        out, _ = self.lstm(out)

        if self.classifier is not None:
            out = self.classifier(out[:, -1, :])  # We only want the last step of the LSTM output

        return out


def en_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    YiboGao's custom loss function
    """
    epsilon = 1.e-7
    gamma = float(0.3)

    y_pred = torch.clip(y_pred, epsilon, 1.0 - epsilon)
    pos_pred = torch.pow(- torch.log(y_pred), gamma)
    neg_pred = torch.pow(- torch.log(1 - y_pred), gamma)
    y_t = torch.multiply(y_true, pos_pred) + torch.multiply(1 - y_true, neg_pred)
    loss = torch.mean(y_t)

    return loss


class YiboGaoClassifier(nn.Module):
    def __init__(self, in_features, n_classes):
        super(YiboGaoClassifier, self).__init__()
        self.n_classes = n_classes
        self.in_features = in_features
        self.module = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.7),
            nn.Linear(in_features=in_features, out_features=100),
            nn.ReLU(),
            nn.Dropout(p=0.7),
            nn.Linear(100, n_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.module(x)


class YiboGao(pyCNN_LSTM_BaseModule):
    """
    CNN using RTA blocks for end-to-end attrial fibrilation detection.

    Parameters
    ----------
        include_top: bool, default=True
            Whether to include the fully-connected layer at the top of the network.

        weights: str, default=None
            The path to the weights file to be loaded.

        input_tensor: keras.Tensor, defaults=None
            Optional Keras tensor (i.e. output of `layers.Input()`) to use as input for the model.

        input_shape: Tuple, defaults=None
            If `input_tensor=None`, a tuple that defines the input shape for the model.

        classes: int, defaults=5
            If `include_top=True`, the number of units in the top layer to classify data.

        classifier_activation: str or callable, defaults='softmax'
            The activation function to use on the "top" layer. Ignored unless `include_top=True`. Set
            `classifier_activation=None` to return the logits of the "top" layer.

        return_loss: bool, defaults=False
            Whether to return the custom loss function (en_loss) employed for training this model.

    Returns
    -------
        model: `keras.Model`
            A `keras.Model` instance.

    References
    ----------
        Gao, Y., Wang, H., & Liu, Z. (2021). An end-to-end atrial fibrillation detection by a novel residual-based
        temporal attention convolutional neural network with exponential nonlinearity loss.
        Knowledge-Based Systems, 212, 106589.

    Notes
    -----
        Code adapted from the original implementation available at:
            https://github.com/o00O00o/RTA-CNN
    """
    def __init__(self,
                 in_features: int,
                 top_module: Optional[nn.Module] = YiboGaoClassifier(128, 5),
                 loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = en_loss,
                 optimizer: torch.optim.Optimizer = torch.optim.Adam,
                 **kwargs
                 ):
        super(YiboGao, self).__init__(in_features, top_module, loss, optimizer, **kwargs)

        # Model definition
        self.module = nn.Sequential(
            RTABlock(in_features, 16, 32),
            nn.MaxPool1d(kernel_size=4),
            RTABlock(16, 32, 16),
            nn.MaxPool1d(kernel_size=4),
            RTABlock(32, 64, 9),
            nn.MaxPool1d(kernel_size=2),
            RTABlock(64, 64, 9),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(p=0.6),
            RTABlock(64, 128, 3),
            nn.MaxPool1d(kernel_size=2),
            RTABlock(128, 128, 3),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(p=0.6)
        )

    def forward(self, x):
        x = self.module(x)

        if self.classifier is not None:
            x = self.classifier(x)

        return x


class YaoQihangClassifier(nn.Module):
    def __init__(self, in_features, n_classes):
        super(YaoQihangClassifier, self).__init__()
        self.module = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=32),
            nn.Tanh(),
            nn.Linear(32, n_classes),
            # On each timestep it makes the softmax. Then, the final classification probablity is the average
            # throughout all the timesteps
            nn.Softmax(dim=2)
        )

    def forward(self, x):
        x1 = self.module(x)
        mean = torch.mean(x1, dim=1)
        return mean


class YaoQihang(pyCNN_LSTM_BaseModule):

    def __convBlock(self, in_features, filters) -> nn.Module:
        return nn.Sequential(
            nn.Conv1d(in_channels=in_features, out_channels=filters, kernel_size=3, padding='same'),
            nn.BatchNorm1d(num_features=filters),
            nn.ReLU()
        )

    def __init__(self,
                 in_features: int,
                 top_module: Optional[nn.Module] = YaoQihangClassifier(32, 5),
                 loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = F.nll_loss,
                 optimizer: torch.optim.Optimizer = torch.optim.Adam,
                 **kwargs):
        super(YaoQihang, self).__init__(in_features, top_module, loss, optimizer, **kwargs)

        self.convolutions = nn.Sequential(
            self.__convBlock(in_features, 64),
            self.__convBlock(64, 64),
            nn.MaxPool1d(kernel_size=3, stride=3),
            self.__convBlock(64, 128),
            self.__convBlock(128, 128),
            nn.MaxPool1d(kernel_size=3, stride=3),
            self.__convBlock(128, 256),
            self.__convBlock(256, 256),
            self.__convBlock(256, 256),
            nn.MaxPool1d(kernel_size=3, stride=3),
            self.__convBlock(256, 256),
            self.__convBlock(256, 256),
            self.__convBlock(256, 256),
            nn.MaxPool1d(kernel_size=3, stride=3),
            self.__convBlock(256, 256),
            self.__convBlock(256, 256),
            self.__convBlock(256, 256),
            nn.MaxPool1d(kernel_size=3, stride=3),
        )

        # Temporal Layers (2 stacked LSTM): REMEMBER TO SWAP DIMENSIONS ON THE FORWARD METHOD.
        # input is 256 as we expect the last convolution with 256 filters.
        self.lstm = nn.LSTM(input_size=256, hidden_size=32, num_layers=2, dropout=0.2, batch_first=True)


    def forward(self, x):
        x = self.convolutions(x)
        x = flip_indices_for_conv_to_lstm(x)
        x, _ = self.lstm(x)  # We don't care about hidden state.

        if self.classifier is not None:
            x = self.classifier(x)

        return x


class HtetMyetLynn(pyCNN_LSTM_BaseModule):
    def __init__(self,
                 in_features: int,
                 use_rnn: Optional[str] = 'gru',  # Options are 'gru' or 'lstm' or None
                 top_module: Optional[nn.Module] = nn.Sequential(nn.Linear(in_features=80, out_features=5), nn.Softmax()),
                 loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = F.nll_loss,
                 optimizer: torch.optim.Optimizer = torch.optim.Adam,
                 **kwargs
                 ):
        super(HtetMyetLynn, self).__init__(in_features, top_module, loss, optimizer, **kwargs)

        self.convLayers = nn.Sequential(
            nn.Conv1d(in_channels=in_features, out_channels=30, kernel_size=5, padding='same'),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=30, out_channels=30, kernel_size=2, padding='same'),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=30, out_channels=60, kernel_size=5, padding='same'),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=60, out_channels=60, kernel_size=2, padding='same'),
            nn.MaxPool1d(kernel_size=2),
        )

        if use_rnn is not None:
            if use_rnn.lower() == 'gru':
                self.rnn = nn.GRU(input_size=60, hidden_size=40, dropout=0.2, bidirectional=True, batch_first=True)
            else:
                self.rnn = nn.LSTM(input_size=60, hidden_size=40, dropout=0.2, bidirectional=True, batch_first=True)
        else:
            self.rnn = None

    def forward(self, x):
        x = self.convLayers(x)

        if self.rnn is not None:
            x = flip_indices_for_conv_to_lstm(x)
            x, _ = self.rnn(x)

        if self.classifier is not None:
            x = self.classifier(x[:, -1, :])  # get only the last timestep

        return x


class YildirimOzal(pyCNN_LSTM_BaseModule):

    def __init__(self,
                 input_shape: Tuple[int, int],    # (Channels, seq_length)
                 train_autoencoder: bool = True,  # This is for training the autoencoder or the LSTM-classifier
                 top_module: Optional[nn.Module] = None,
                 loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = F.nll_loss,
                 optimizer: torch.optim.Optimizer = torch.optim.Adam,
                 **kwargs
                 ):
        super(YildirimOzal, self).__init__(input_shape[0], top_module, loss, optimizer, **kwargs)
        self.train_autoencoder = train_autoencoder

        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=self.in_features, out_channels=16, kernel_size=5, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=16, out_channels=64, kernel_size=5, padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=64),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=1, kernel_size=3, padding='same'),
            nn.MaxPool1d(kernel_size=2)
        )
        output_length = input_shape[1] // 2 // 2 // 2  # seq_length is reduced by a half three times (3 maxpooling)

        self.decoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding='same'),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(in_channels=64, out_channels=16, kernel_size=5, padding='same'),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * output_length * 2 * 2, input_shape[1])  # Reconstruction (16 -> out_channels for conv, and 2 upsamples.
        )

        self.lstm = nn.LSTM(input_size=1, hidden_size=32, batch_first=True)

    def forward(self, x):
        reduction = self.encoder(x)
        if self.train_autoencoder:
            return self.decoder(reduction)
        else:
            x, _ = self.lstm(reduction)
            if self.classifier is not None:
                x = self.classifier(x)
            return x


class CaiWenjuan(pyCNN_LSTM_BaseModule):  # TODO: Squeeze and Activation units depends on input size.
    def __init__(self,
                 in_features: int,
                 reduction_ratio: float = 0.6,
                 top_module: Optional[nn.Module] = nn.Sequential(nn.Linear(67, 5), nn.Softmax()),
                 loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = F.nll_loss,
                 optimizer: torch.optim.Optimizer = torch.optim.Adam,
                 **kwargs
                 ):
        super(CaiWenjuan, self).__init__(in_features, top_module, loss, optimizer, **kwargs)

        # Convolutional inputs
        self.conv1 = nn.Conv1d(in_channels=in_features, out_channels=8, kernel_size=1, padding='same')
        self.conv2 = nn.Conv1d(in_channels=in_features, out_channels=16, kernel_size=3, padding='same')
        self.conv3 = nn.Conv1d(in_channels=in_features, out_channels=24, kernel_size=5, padding='same')
        # NOTE: Concatenate layers after that (cat or stack)
        self.in_features_after_concat = 8 + 16 + 24

        # dense-block construction
        dense_layers = [2, 4, 6, 4]
        in_feat = self.in_features_after_concat
        layers = []
        for i, n_block in enumerate(dense_layers):
            layers.append(SqueezeAndExcitationModule(in_feat, 32))
            layers.append(DenseNetDenseBlock(in_features=in_feat, layers=n_block, growth_rate=6))
            in_feat = layers[-1].output_features
            layers.append(SqueezeAndExcitationModule(in_feat, 32))
            if i < len(dense_layers) - 1:
                layers.append(DenseNetTransitionBlock(in_features=in_feat, reduction=reduction_ratio))
                in_feat = layers[-1].out_features

        self.dense_module = nn.Sequential(*layers)
        self.output_features = in_feat

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)

        x = torch.cat((x1, x2, x3), dim=1)
        x = self.dense_module(x)

        # Global average pooling (average value for each channel
        x = torch.mean(x, dim=2)

        if self.classifier is not None:
            x = self.classifier(x)

        return x


class ZhangJin_Classifier(nn.Module):
    def __init__(self, in_features, n_classes):
        super(ZhangJin_Classifier, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_features, n_classes),
            nn.Softmax()
        )

    def forward(self, x):
        x1 = torch.max(x, dim=1).values

        return self.linear(x1)


class ZhangJin(pyCNN_LSTM_BaseModule):
    def __convBlock(self, in_features, filters) -> nn.Module:
        return nn.Sequential(
            nn.Conv1d(in_channels=in_features, out_channels=filters, kernel_size=3, padding='same'),
            nn.BatchNorm1d(num_features=filters),
            nn.ReLU()
        )

    def __init__(self,
                 in_features: int,
                 decrease_ratio: int = 2,
                 top_module: Optional[nn.Module] = ZhangJin_Classifier(24, 5),
                 loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = F.nll_loss,
                 optimizer: torch.optim.Optimizer = torch.optim.Adam,
                 **kwargs
                 ):
        super(ZhangJin, self).__init__(in_features, top_module, loss, optimizer, **kwargs)
        self.decrease_ratio = decrease_ratio

        self.convolutions = nn.ModuleList(
            [
                nn.Sequential(
                    self.__convBlock(in_features, 64),
                    self.__convBlock(64, 64),
                    nn.MaxPool1d(kernel_size=3, stride=3),
                    nn.Dropout(p=0.2)
                ),
                nn.Sequential(      # Features after temporal attention is 1.
                    self.__convBlock(64, 128),
                    self.__convBlock(128, 128),
                    nn.MaxPool1d(kernel_size=3, stride=3),
                    nn.Dropout(p=0.2)
                ),
                nn.Sequential(
                    self.__convBlock(128, 256),
                    self.__convBlock(256, 256),
                    self.__convBlock(256, 256),
                    nn.MaxPool1d(kernel_size=3, stride=3),
                    nn.Dropout(p=0.2)
                ),
                nn.Sequential(
                    self.__convBlock(256, 256),
                    self.__convBlock(256, 256),
                    self.__convBlock(256, 256),
                    nn.MaxPool1d(kernel_size=3, stride=3),
                    nn.Dropout(p=0.2),
                ),
                nn.Sequential(
                    self.__convBlock(256, 256),
                    self.__convBlock(256, 256),
                    self.__convBlock(256, 256),
                    nn.MaxPool1d(kernel_size=3, stride=3),
                    nn.Dropout(p=0.2)
                )
            ]
        )
        self.spatial_attention = nn.ModuleList(
            [SpatialAttentionBlockZhangJin(f, decrease_ratio) for f in [64, 128, 256, 256, 256]]
        )
        self.temporal_attention = nn.ModuleList(
            [TemporalAttentionBlockZhangJin() for i in range(5)]
        )

        self.gru = nn.GRU(input_size=256, hidden_size=12, batch_first=True, bidirectional=True, dropout=0.2)

    def forward(self, x):
        for i in range(5):
            x = self.convolutions[i](x)
            x_spatial = self.spatial_attention[i](x)
            x = torch.multiply(x, x_spatial)
            x_temp = self.temporal_attention[i](x)
            x = torch.multiply(x, x_temp)

        x = flip_indices_for_conv_to_lstm(x)
        x, _ = self.gru(x)
        if self.classifier is not None:
            x = self.classifier(x)
        return x


####################################
#  SIMPLE TRAINING TEST
################################
def sparse_categorical_crossentropy(y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    ONLY FOR TESTING PURPOSES: This functions simulates the Keras spares_categorical_crossentropy in PyTorch
    """
    return F.nll_loss(torch.log(y_pred), y)


# Read the data as a DataLoader (I create a class for this in `data.py`)
mit_bih = MIT_BIH(path="physionet.org/files/mitdb/1.0.0/", return_hot_coded=False)
train_sampler = SubsetRandomSampler(range(len(mit_bih)))
train_loader = torch.utils.data.DataLoader(mit_bih, batch_size=256, sampler=train_sampler, num_workers=1)

# Example of model using a different optimizer and loss function than defaults.
# Default loss is nll_loss, here we use a modification for sparse_cce.
# Default optimizer is Adam, here we use SGD with momentum.
a = ZhangJin(in_features=1,
            optimizer=torch.optim.SGD,
            loss=nn.CrossEntropyLoss(),
             lr=0.001,  ## Additional params of the  SGD optimizer
             momentum=0.01)

# Dry-run for Lazy initialisation, see
# https://pytorch.org/docs/stable/generated/torch.nn.modules.lazy.LazyModuleMixin.html#torch.nn.modules.lazy.LazyModuleMixin
t = torch.randn(2, 1, 1000)
r = a(t)

# Train
trainer = pl.Trainer(gpus=1, max_epochs=50)
trainer.fit(a, train_dataloader=train_loader)

