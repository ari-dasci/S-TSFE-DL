import os
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from typing import Callable, Optional
from data import MIT_BIH
from torch.utils.data.sampler import SubsetRandomSampler
from blocks_pytorch import RTABlock
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
                 top_module: Optional[nn.Module] = None,
                 loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = F.nll_loss,
                 optimizer: torch.optim.Optimizer = torch.optim.Adam,
                 **kwargs
                 ):
        super(pyCNN_LSTM_BaseModule, self).__init__()
        self.kwargs = kwargs
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

    def __init__(self, n_classes):
        super(OhShuLih_Classifier, self).__init__()
        self.model = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=20, out_features=20),
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
                 top_module: Optional[nn.Module] = OhShuLih_Classifier(n_classes=5),
                 loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = F.nll_loss,
                 optimizer: torch.optim.Optimizer = torch.optim.Adam,
                 **kwargs
                 ):
        super(OhShuLih, self).__init__()

        self.loss = loss
        self.optimizer = optimizer   # Optimizer is a type, it must be instantiated later on the configure_optimizers
        self.kwargs = kwargs  # Save the parameters dict of the optimizer.


        conv_layers = []

        # The padding of (kernel_size - 1) is for full convolution.
        # We use a Lazy Conv1D layer here as we dont know the input channels beforehand.
        # NOTE: INPUT SHAPE MUST BE (N, C, L) where N is the batch size, C is the NUMBER OF CHANNEL (as opposite to keras)
        # and L is the number of timesteps of Length of the 1D signal.
        conv_layers.append(nn.LazyConv1d(out_channels=3, kernel_size=20, bias=False, stride=1, padding=19))
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

        self.classifier = top_module

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
    def __init__(self, n_classes):
        super(YiboGaoClassifier, self).__init__()
        self.n_classes = n_classes
        self.module = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.7),
            nn.LazyLinear(out_features=100),
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
                 top_module: Optional[nn.Module] = YiboGaoClassifier(5),
                 loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = en_loss,
                 optimizer: torch.optim.Optimizer = torch.optim.Adam,
                 **kwargs
                 ):
        super(YiboGao, self).__init__()

        # Model definition
        self.loss = loss
        self.optimizer = optimizer
        self.kwargs = kwargs

        rtaBlocks = [nn.Sequential(RTABlock(fil, ker), nn.MaxPool1d(kernel_size=pool))
                                   for fil, ker, pool in zip([16, 32, 64, 64],
                                                             [32, 16, 9, 9],
                                                             [4, 4, 2, 2])]

        self.module = nn.Sequential(
            nn.Sequential(*rtaBlocks),
            nn.Dropout(p=0.6),
            RTABlock(128, 3),
            nn.MaxPool1d(kernel_size=2),
            RTABlock(128, 3),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(p=0.6)
        )

        self.classifier = top_module

    def forward(self, x):
        x = self.module(x)

        if self.classifier is not None:
            x = self.classifier(x)

        return x


class YaoQihangClassifier(nn.Module):
    def __init__(self, n_classes):
        super(YaoQihangClassifier, self).__init__()
        self.module = nn.Sequential(
            nn.LazyLinear(out_features=32),
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

    def __init__(self,
                 top_module: Optional[nn.Module] = YaoQihangClassifier(5),
                 loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = F.nll_loss,
                 optimizer: torch.optim.Optimizer = torch.optim.Adam,
                 **kwargs):
        super(YaoQihang, self).__init__()
        self.optimizer = optimizer
        self.loss = loss
        self.kwargs = kwargs

        convLayers = []
        for conv_layers, filters in zip([2, 2, 3, 3, 3],
                                        [64, 128, 256, 256, 256]):  # 5-block layers
            for i in range(conv_layers):
                block = nn.Sequential(
                    nn.LazyConv1d(out_channels=filters, kernel_size=3, padding='same'),
                    nn.BatchNorm1d(num_features=filters),
                    nn.ReLU()
                )
                convLayers.append(block)
            convLayers.append(nn.MaxPool1d(kernel_size=3, stride=3))

        self.convolutions = nn.Sequential(*convLayers)

        # Temporal Layers (2 stacked LSTM): REMEMBER TO SWAP DIMENSIONS ON THE FORWARD METHOD.
        # input is 256 as we expect the last convolution with 256 filters.
        self.lstm = nn.LSTM(input_size=256, hidden_size=32, num_layers=2, dropout=0.2, batch_first=True)

        self.classifier = top_module

    def forward(self, x):
        x = self.convolutions(x)
        x = flip_indices_for_conv_to_lstm(x)
        x, _ = self.lstm(x)  # We don't care about hidden state.

        if self.classifier is not None:
            x = self.classifier(x)

        return x

class HtetMyetLynn(pyCNN_LSTM_BaseModule):
    def __init__(self,
                 use_rnn: Optional[str] = 'gru',  # Options are 'gru' or 'lstm' or None
                 top_module: Optional[nn.Module] = nn.Sequential(nn.LazyLinear(out_features=5), nn.Softmax()),
                 loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = F.nll_loss,
                 optimizer: torch.optim.Optimizer = torch.optim.Adam,
                 **kwargs
                 ):
        super(HtetMyetLynn, self).__init__()
        self.loss = loss
        self.optimizer = optimizer
        self.kwargs = kwargs

        self.convLayers = nn.Sequential(
            nn.LazyConv1d(out_channels=30, kernel_size=5, padding='same'),
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

        self.classifier = top_module

    def forward(self, x):
        x = self.convLayers(x)

        if self.rnn is not None:
            x = flip_indices_for_conv_to_lstm(x)
            x, _ = self.rnn(x)

        if self.classifier is not None:
            x = self.classifier(x[:, -1, :])  # get only the last timestep

        return x


# class ZhangJin(pyCNN_LSTM_BaseModule):  # TODO: Finish this model if possible
#     def __init__(self,
#                  decrease_ratio: int = 2,
#                  top_module: Optional[nn.Module] = None,
#                  loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = F.nll_loss,
#                  optimizer: torch.optim.Optimizer = torch.optim.Adam,
#                  **kwargs
#                  ):
#         attention_modules = []
#         convLayers[]
#         for conv_layers, filters in zip([2, 2, 3, 3, 3],
#                                         [64, 128, 256, 256, 256]):  # 5-block layers
#             for i in range(conv_layers):
#                 block = nn.Sequential(
#                     nn.LazyConv1d(out_channels=filters, kernel_size=3, padding='same'),
#                     nn.BatchNorm1d(num_features=filters),
#                     nn.ReLU()
#                 )
#                 convLayers.append(block)
#             convLayers.append(nn.MaxPool1d(kernel_size=3, stride=3))
#             convLayers.append(nn.Dropout(p=0.2))
#
#         self.convolutions = nn.Sequential(*convLayers)
#
# def ZhangJin(include_top=True,
#              weights=None,
#              input_tensor=None,
#              input_shape=None,
#              classes=5,
#              classifier_activation="softmax",
#              decrease_ratio=2):
#
#
#     # Model definition
#     for conv_layers, filters in zip([2, 2, 3, 3, 3],
#                                     [64, 128, 256, 256, 256]):  # 5-block layers
#         for i in range(conv_layers):
#             x = layers.Conv1D(filters=filters, kernel_size=3, padding="same")(x)
#             x = layers.BatchNormalization()(x)
#             x = layers.Activation(activation=relu)(x)
#         x = layers.MaxPooling1D(pool_size=3)(x)
#         x = layers.Dropout(rate=0.2)(x)
#
#         # Adds attention module after each convolutional block
#         x_spatial = spatial_attention_block_ZhangJin(decrease_ratio, x)
#         x = layers.multiply([x_spatial, x])
#         x_temporal = temporal_attention_block_ZhangJin(x)
#         x = layers.multiply([x_temporal, x])
#
#     x = layers.Bidirectional(layers.GRU(units=12, return_sequences=True))(x)
#     x = layers.Dropout(rate=0.2)(x)
#     if include_top:
#         x = layers.GlobalMaxPool1D()(x)
#         x = layers.Dense(units=classes, activation=classifier_activation)(x)
#
#     model = keras.Model(inputs=inp, outputs=x, name="ZhangJin")
#
#     if weights is not None:
#         model.load_weights(weights)
#
#     return model

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
train_loader = torch.utils.data.DataLoader(mit_bih, batch_size=256, sampler=train_sampler, num_workers=8)

# Example of model using a different optimizer and loss function than defaults.
# Default loss is nll_loss, here we use a modification for sparse_cce.
# Default optimizer is Adam, here we use SGD with momentum.
a = HtetMyetLynn(optimizer=torch.optim.SGD,
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

