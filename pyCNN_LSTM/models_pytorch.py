import os
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from typing import Callable, Optional
from data import MIT_BIH
from torch.utils.data.sampler import SubsetRandomSampler


class pyCNN_LSTM_BaseModule(pl.LightningModule):
    """
    Base module for any pyTorch Lightning based algorithm in this library

    Parameters
    ----------
        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
            The loss function to use. It should accept two Tensors as inputs (predictions, targets) and return
            a Tensor with the loss.

        classes: int, defaults=5
            Number of classes of the problem. Ignored if `include_top==True` and `top_module is not None`

        include_top: bool, defaults=True
            Whether to include a custom classification layer at the top of the network

        top_module: nn.Module, defaults=None
            If `include_top = True` the nn.Module to be used as top layers.
            If `None` use the default fully-connected layer.

        optimizer:  torch.optim.Optimizer
            The pyTorch Optimizer to use. Note that this must be only the class type and not an instance of the class!!

        optimzer_params: dict
            A dictionary with the parameters of the optimizer.
    """

    def __init__(self,
                 loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = F.nll_loss,
                 optimizer: torch.optim.Optimizer = torch.optim.Adam,
                 optimizer_params: dict = dict(lr=0.001),
                 classes: int = 5,
                 include_top: bool = True,
                 top_module: Optional[nn.Module] = None):
        super(pyCNN_LSTM_BaseModule, self).__init__()
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
        opt = self.optimizer(self.parameters(), **self.optimizer_params)
        return opt


class OhShuLih(pyCNN_LSTM_BaseModule):
    """
    CNN+LSTM model for Arrythmia classification.

    Parameters
    ----------
        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
            The loss function to use. It should accept two Tensors as inputs (predictions, targets) and return
            a Tensor with the loss.

        classes: int, defaults=5
            Number of classes of the problem. Ignored if `include_top==True` and `top_module is not None`

        include_top: bool, defaults=True
            Whether to include a custom classification layer at the top of the network

        top_module: nn.Module, defaults=None
            If `include_top = True` the nn.Module to be used as top layers.
            If `None` use the default fully-connected layer.

        optimizer:  torch.optim.Optimizer
            The pyTorch Optimizer to use. Note that this must be only the class type and not an instance of the class!!

        optimzer_params: dict
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
                 loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = F.nll_loss,
                 optimizer: torch.optim.Optimizer = torch.optim.Adam,
                 optimizer_params: dict = dict(lr=0.001),
                 classes: int = 5,
                 include_top: bool = True,
                 top_module: Optional[nn.Module] = None):
        super(OhShuLih, self).__init__()

        self.loss = loss
        self.optimizer = optimizer   # Optimizer is a type, it must be instantiated later on the configure_optimizers
        self.optimizer_params = optimizer_params  # Save the parameters dict of the optimizer.


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

        if include_top:
            if top_module is None:
                self.classifier = nn.Sequential(
                    nn.Dropout(p=0.2),
                    nn.Linear(in_features=20, out_features=20),
                    nn.ReLU(),
                    nn.Linear(in_features=20, out_features=10),
                    nn.ReLU(),
                    nn.Linear(in_features=10, out_features=classes),
                    nn.Softmax()
                )
            else:
                self.classifier = top_module
        else:
            self.classifier = None

    def forward(self, x):
        out = self.convolutions(x)
        # Now, flip indices using a view for the LSTM as it requires a shape of (N, L, H_in = C)
        out = out.view(out.size(0), out.size(2), out.size(1))
        out, _ = self.lstm(out)

        if self.classifier is not None:
            out = self.classifier(out[:, -1, :])  # We only want the last step of the LSTM output

        return out








####################################
#  SIMPLE TRAINING TEST
################################
def sparse_categorical_crossentropy(y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    ONLY FOR TESTING PURPOSES: This functions simulates the Keras spares_categorical_crossentropy in PyTorch
    """
    return F.nll_loss(torch.log(y_pred), y)


# Read the data as a DataLoader (I create a class for this in `data.py`)
mit_bih = MIT_BIH(path="physionet.org/files/mitdb/1.0.0/")
train_sampler = SubsetRandomSampler(range(len(mit_bih)))
train_loader = torch.utils.data.DataLoader(mit_bih, batch_size=256, sampler=train_sampler)

# Example of model using a different optimizer and loss function than defaults.
# Default loss is nll_loss, here we use a modification for sparse_cce.
# Default optimizer is Adam, here we use SGD with momentum.
a = OhShuLih(loss=sparse_categorical_crossentropy,
             optimizer=torch.optim.SGD,
             optimizer_params=dict(lr=0.001, momentum=0.01))

# Dry-run for Lazy initialisation, see
# https://pytorch.org/docs/stable/generated/torch.nn.modules.lazy.LazyModuleMixin.html#torch.nn.modules.lazy.LazyModuleMixin
t = torch.randn(1, 1, 1000)
r = a(t)

# Train
trainer = pl.Trainer(gpus=1, max_epochs=6)
trainer.fit(a, train_dataloader=train_loader)

