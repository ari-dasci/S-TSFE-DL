import os
import numpy as np
import torch
import torchmetrics
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from typing import Callable, Optional, Dict, Tuple
from TSFEDL.blocks_pytorch import RTABlock, SqueezeAndExcitationModule, DenseNetDenseBlock, DenseNetTransitionBlock, \
    SpatialAttentionBlockZhangJin, TemporalAttentionBlockZhangJin
from TSFEDL.utils import flip_indices_for_conv_to_lstm, flip_indices_for_conv_to_lstm_reshape
from TSFEDL.utils import TimeDistributed



class TSFEDL_BaseModule(pl.LightningModule):
    """
    Base module for any pyTorch Lightning based algorithm in this library

    Parameters
    ----------
        in_features: int,
            The number of input features.

        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
            The loss function to use. It should accept two Tensors as inputs (predictions, targets) and return
            a Tensor with the loss.

        metrics: List[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]
            A dictionary of metrics to be returned in the test phase. These metric are functions that accepts two tensors as
            inputs, i.e., predictions and targets, and return another tensor with the metric value. Defaults contains the accuracy

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
                 loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = nn.CrossEntropyLoss(),
                 metrics: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
                 optimizer: torch.optim.Optimizer = torch.optim.Adam,
                 **kwargs
                 ):
        super(TSFEDL_BaseModule, self).__init__()
        self.kwargs = kwargs
        self.in_features = in_features
        self.loss = loss
        self.optimizer = optimizer
        self.classifier = top_module
        self.metrics = metrics
        pass

    def forward(self, x):
        pass

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log('train_loss', loss)
        #self.log('train_acc', self.accuracy(y_hat, y))
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        # compute test loss
        loss = self.loss(y_hat, y)
        self.log('test_loss', loss, on_step=True, on_epoch=True)

        # compute the remaining test metrics
        for name, f in self.metrics.items():
            value = f(y_hat, y)
            self.log(str('test_' + name), value, on_step=True, on_epoch=True)


    def configure_optimizers(self):
        opt = self.optimizer(self.parameters(), **self.kwargs)
        return opt


class OhShuLih_Classifier(nn.Module):
    """
    Classifier of the OhShuLi model.

    Parameters
    ----------
        in_features: int
            Number of features of the input tensors

        n_classes: int
            Number of classes to predict at the end of the network

    Returns
    -------
    `LightningModule`
        A pyTorch Lightning Module instance.
    """

    def __init__(self, in_features, n_classes):
        super(OhShuLih_Classifier, self).__init__()
        self.model = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=in_features, out_features=20),
            nn.ReLU(),
            nn.Linear(in_features=20, out_features=10),
            nn.ReLU(),
            nn.Linear(in_features=10, out_features=n_classes)
        )

    def forward(self, x):
        return self.model(x)


class OhShuLih(TSFEDL_BaseModule):
    """
    CNN+LSTM model for Arrythmia classification.

    Parameters
    ----------
        in_features: int
            Number of features of the input tensors

        top_module: nn.Module, defaults=OhShuLih_Classifier(20,5)
            The optional  nn.Module to be used as additional top layers.

        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
            The loss function to use. It should accept two Tensors as inputs (predictions, targets) and return
            a Tensor with the loss.

        metrics: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]
            Dictionary with the name of the metric and a function to compute the metric from two tensors,
            prediction and true labels.

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
                 loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = nn.CrossEntropyLoss(),
                 metrics: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
                 optimizer: torch.optim.Optimizer = torch.optim.Adam,
                 **kwargs
                 ):
        super(OhShuLih, self).__init__(in_features, top_module, loss, metrics, optimizer, **kwargs)

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
    """
    Classifier of the Yibo Gao model.

    Parameters
    ----------
        in_features: int
            Number of features of the input tensors

        n_classes: int
            Number of classes to predict at the end of the network

    Returns
    -------
    `LightningModule`
        A pyTorch Lightning Module instance.
    """
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


class YiboGao(TSFEDL_BaseModule):
    """
    CNN using RTA blocks for end-to-end attrial fibrilation detection.

    Parameters
    ----------
        in_features: int
            Number of features of the input tensors

        top_module: nn.Module, defaults=YiboGaoClassifier(384, 5)
            The optional  nn.Module to be used as additional top layers.

        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
            The loss function to use. It should accept two Tensors as inputs (predictions, targets) and return
            a Tensor with the loss.

        metrics: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]
            Dictionary with the name of the metric and a function to compute the metric from two tensors,
            prediction and true labels.

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
                 top_module: Optional[nn.Module] = YiboGaoClassifier(384, 5),
                 loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = en_loss,
                 metrics: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
                 optimizer: torch.optim.Optimizer = torch.optim.Adam,
                 **kwargs
                 ):
        super(YiboGao, self).__init__(in_features, top_module, loss, metrics, optimizer, **kwargs)

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
    """
    Classifier of the Yao Qihang model.

    Parameters
    ----------
        in_features: int
            Number of features of the input tensors

        n_classes: int
            Number of classes to predict at the end of the network

    Returns
    -------
    `LightningModule`
        A pyTorch Lightning Module instance.
    """
    def __init__(self, in_features, n_classes):
        super(YaoQihangClassifier, self).__init__()
        self.module = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=32),
            nn.Tanh(),
            nn.Linear(32, n_classes),
            # On each timestep it makes the softmax. Then, the final classification probablity is the average
            # throughout all the timesteps
            # nn.Softmax(dim=2)
        )

    def forward(self, x):
        x1 = self.module(x)
        # NOTE:  This classifier must return the mean softmax over ALL TIMESTEPS. TAKE CARE OF THE LOSS FUNCTION!
        mean = torch.mean(F.softmax(x1, dim=2), dim=1)
        return mean


class YaoQihang(TSFEDL_BaseModule):
    """
    Attention-based time-incremental convolutional neural network (ATI-CNN)

    Parameters
    ----------
        in_features: int
            Number of features of the input tensors

        top_module: nn.Module, defaults=YaoQihangClassifier(32, 5)
            The optional  nn.Module to be used as additional top layers.

        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
            The loss function to use. It should accept two Tensors as inputs (predictions, targets) and return
            a Tensor with the loss.

        metrics: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]
            Dictionary with the name of the metric and a function to compute the metric from two tensors,
            prediction and true labels.

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
        Yao, Q., Wang, R., Fan, X., Liu, J., & Li, Y. (2020). Multi-class Arrhythmia detection from 12-lead varied-length
        ECG using Attention-based Time-Incremental Convolutional Neural Network. Information Fusion, 53, 174-182.

    """
    def __convBlock(self, in_features, filters) -> nn.Module:
        return nn.Sequential(
            nn.Conv1d(in_channels=in_features, out_channels=filters, kernel_size=3, padding='same'),
            nn.BatchNorm1d(num_features=filters),
            nn.ReLU()
        )

    def __init__(self,
                 in_features: int,
                 top_module: Optional[nn.Module] = YaoQihangClassifier(32, 5),
                 loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
                 metrics: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
                 optimizer: torch.optim.Optimizer = torch.optim.Adam,
                 **kwargs):
        super(YaoQihang, self).__init__(in_features, top_module, loss, metrics, optimizer, **kwargs)

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


class HtetMyetLynn(TSFEDL_BaseModule):
    """
    Hybrid CNN+Bidirectional RNN

    Parameters
    ----------
        in_features: int
            Number of features of the input tensors

        use_rnn: str, ["gru", "lstm", None], defaults="gru"
            This parameter controls wether to use RNN layers and what type to use

        top_module: nn.Module
            The optional  nn.Module to be used as additional top layers.

        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
            The loss function to use. It should accept two Tensors as inputs (predictions, targets) and return
            a Tensor with the loss.

        metrics: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]
            Dictionary with the name of the metric and a function to compute the metric from two tensors,
            prediction and true labels.

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
        Lynn, H. M., Pan, S. B., & Kim, P. (2019). A deep bidirectional GRU network model for biometric
        electrocardiogram classification based on recurrent neural networks. IEEE Access, 7, 145395-145405.
    """
    def __init__(self,
                 in_features: int,
                 use_rnn: Optional[str] = 'gru',  # Options are 'gru' or 'lstm' or None
                 top_module: Optional[nn.Module] = nn.Sequential(nn.Linear(in_features=80, out_features=5)),
                 loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = nn.CrossEntropyLoss(),
                 metrics: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
                 optimizer: torch.optim.Optimizer = torch.optim.Adam,
                 **kwargs
                 ):
        super(HtetMyetLynn, self).__init__(in_features, top_module, loss, metrics, optimizer, **kwargs)

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


class YildirimOzal(TSFEDL_BaseModule):
    """CAE-LSTM

    Parameters
    ----------
        input_shape: Tuple[int, int]
            Shape of the input tensors (number of channels, sequence length)

        train_autoencoder: bool, defaults=True
            This parameter controls wether to train the autoencoder or the classifier

        top_module: nn.Module
            The optional  nn.Module to be used as additional top layers.

        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
            The loss function to use. It should accept two Tensors as inputs (predictions, targets) and return
            a Tensor with the loss.

        metrics: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]
            Dictionary with the name of the metric and a function to compute the metric from two tensors,
            prediction and true labels.

        optimizer:  torch.optim.Optimizer
            The pyTorch Optimizer to use. Note that this must be only the class type and not an instance of the class!!

        **kwargs: dict
            A dictionary with the parameters of the optimizer.

    Returns
    -------
        autoenconder: `LightningModule`
            A pyTorch Lightning Module instance with the autoencoder.

        encoder: `LightningModule`
            A pyTorch Lightning Module instance with the encoder.

        model: `LightningModule`
            A pyTorch Lightning Module instance with the classifier.

    References
    ----------
        Yildirim, Ozal, et al. "A new approach for arrhythmia classification using deep coded features and LSTM
        networks." Computer methods and programs in biomedicine 176 (2019): 121-133.
     """
    def __init__(self,
                 input_shape: Tuple[int, int],    # (Channels, seq_length)
                 train_autoencoder: bool = True,  # This is for training the autoencoder or the LSTM-classifier
                 top_module: Optional[nn.Module] = None,
                 loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = nn.CrossEntropyLoss(),
                 metrics: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
                 optimizer: torch.optim.Optimizer = torch.optim.Adam,
                 **kwargs
                 ):
        super(YildirimOzal, self).__init__(input_shape[0], top_module, loss, metrics, optimizer, **kwargs)
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
            reconstruction = self.decoder(reduction)
            reconstruction = reconstruction.view(reconstruction.size(0), 1, reconstruction.size(1))
            return reconstruction
        else:
            x = flip_indices_for_conv_to_lstm(x)
            x, _ = self.lstm(x)
            if self.classifier is not None:
                x = self.classifier(x[:, -1, :])
            return x

    def training_step(self, batch, batch_idx):
        if self.train_autoencoder:
            x, _ = batch
            x_hat = self(x)
            loss = self.loss(x_hat, x)
            self.log('train_loss', loss)
            return loss
        else:
            x, y = batch
            y_hat = self(x)
            loss = self.loss(y_hat, y)
            self.log('train_loss', loss)
            return loss


class CaiWenjuan(TSFEDL_BaseModule):  # TODO: Squeeze and Activation units depends on input size.
    """
    DDNN network presented in:

    Parameters
    ----------
        in_features: int
            Number of features of the input tensors

        reduction_ratio: float, defaults=0.6
            This parameter controls the reduction of the dense transition block

        top_module: nn.Module
            The optional  nn.Module to be used as additional top layers.

        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
            The loss function to use. It should accept two Tensors as inputs (predictions, targets) and return
            a Tensor with the loss.

        metrics: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]
            Dictionary with the name of the metric and a function to compute the metric from two tensors,
            prediction and true labels.

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
        `Cai, Wenjuan, et al. "Accurate detection of atrial fibrillation from 12-lead ECG using deep neural network."
        Computers in biology and medicine 116 (2020): 103378.`
   """
    def __init__(self,
                 in_features: int,
                 reduction_ratio: float = 0.6,
                 top_module: Optional[nn.Module] = nn.Sequential(nn.Linear(67, 5)),
                 loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = nn.CrossEntropyLoss(),
                 metrics: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
                 optimizer: torch.optim.Optimizer = torch.optim.Adam,
                 **kwargs
                 ):
        super(CaiWenjuan, self).__init__(in_features, top_module, loss, metrics, optimizer, **kwargs)

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
    """
    Classifier of the Zhang Jin model.

    Parameters
    ----------
        in_features: int
            Number of features of the input tensors

        n_classes: int
            Number of classes to predict at the end of the network

    Returns
    -------
    `LightningModule`
        A pyTorch Lightning Module instance.
    """
    def __init__(self, in_features, n_classes):
        super(ZhangJin_Classifier, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_features, n_classes)
        )

    def forward(self, x):
        return self.linear(x[:, -1, :])


class ZhangJin(TSFEDL_BaseModule):
    """
    A CNN+Bi-Directional GRU with a spatio-temporal attention mechanism.

    Parameters
    ----------
        in_features: int
            Number of features of the input tensors

        decrease_ratio: int, defaults=2
            This parameter controls the decrease ratio of the spatial attention block

        top_module: nn.Module, defaults=ZhangJin_Classifier(24, 5)
            The optional  nn.Module to be used as additional top layers.

        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
            The loss function to use. It should accept two Tensors as inputs (predictions, targets) and return
            a Tensor with the loss.

        metrics: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]
            Dictionary with the name of the metric and a function to compute the metric from two tensors,
            prediction and true labels.

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
        Zhang, J., Liu, A., Gao, M., Chen, X., Zhang, X., & Chen, X. (2020). ECG-based multi-class arrhythmia detection
        using spatio-temporal attention-based convolutional recurrent neural network.
        Artificial Intelligence in Medicine, 106, 101856.
   """
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
                 loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = nn.CrossEntropyLoss(),
                 metrics: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
                 optimizer: torch.optim.Optimizer = torch.optim.Adam,
                 **kwargs
                 ):
        super(ZhangJin, self).__init__(in_features, top_module, loss, metrics, optimizer, **kwargs)
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


class KongZhengmin_Classifier(nn.Module):
    """
    Classifier of the Kong Zhengmin model.

    Parameters
    ----------
        in_features: int
            Number of features of the input tensors

        n_classes: int
            Number of classes to predict at the end of the network

        return_sequence: bool, defaults=True
            Parameter that controls wether the module returns the sequence or the prediction.

    Returns
    -------
    `LightningModule`
        A pyTorch Lightning Module instance.
    """
    def __init__(self, in_features, n_classes, return_sequence=False):
        super(KongZhengmin_Classifier, self).__init__()
        self.return_sequnce = return_sequence
        self.module = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=50),
            nn.ReLU(),
            nn.Linear(in_features=50, out_features=50),
            nn.ReLU(),
            nn.Linear(in_features=50, out_features=n_classes)
        )

    def forward(self, x):
        if self.return_sequnce:
            return self.module(x)
        else:
            return self.module(x[:, -1, :])


class KongZhengmin(TSFEDL_BaseModule):
    """
    CNN+LSTM

    Parameters
    ----------
        in_features: int
            Number of features of the input tensors

        top_module: nn.Module, defaults=KongZhengmin_Classifier(64, 5)
            The optional  nn.Module to be used as additional top layers.

        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
            The loss function to use. It should accept two Tensors as inputs (predictions, targets) and return
            a Tensor with the loss.

        metrics: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]
            Dictionary with the name of the metric and a function to compute the metric from two tensors,
            prediction and true labels.

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
        Kong, Zhengmin, et al. "Convolution and Long Short-Term Memory Hybrid Deep Neural Networks for Remaining
        Useful Life Prognostics." Applied Sciences 9.19 (2019): 4156.
   """
    def __init__(self,
                 in_features: int,
                 top_module: Optional[nn.Module] = KongZhengmin_Classifier(64, 5),
                 loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = nn.CrossEntropyLoss(),
                 metrics: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
                 optimizer: torch.optim.Optimizer = torch.optim.Adam,
                 **kwargs
                 ):
        super(KongZhengmin, self).__init__(in_features, top_module, loss, metrics, optimizer, **kwargs)

        self.convolution = nn.Sequential(
            nn.Conv1d(in_channels=in_features, out_channels=32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.lstm = nn.LSTM(batch_first=True, input_size=32, hidden_size=64, num_layers=2)

    def forward(self, x):
        x = self.convolution(x)

        x = flip_indices_for_conv_to_lstm(x)
        x, _ = self.lstm(x)

        if self.classifier is not None:
            x = self.classifier(x)
        return x


class WeiXiaoyan_Classifier(nn.Module):
    """
    Classifier of the Wei Xiaoyan model.

    Parameters
    ----------
        in_features: int
            Number of features of the input tensors

        n_classes: int
            Number of classes to predict at the end of the network

    Returns
    -------
    `LightningModule`
        A pyTorch Lightning Module instance.
    """
    def __init__(self, in_features, n_classes):
        super(WeiXiaoyan_Classifier, self).__init__()
        self.module = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            nn.Linear(1024, n_classes),
        )

    def forward(self, x):
        return self.module(x)


class WeiXiaoyan(TSFEDL_BaseModule):
    """
    CNN+LSTM model employed for 2-D images of ECG data. This model is adapted for 1-D time series.

    Parameters
    ----------
        in_features: int
            Number of features of the input tensors

        top_module: nn.Module, defaults=WeiXiaoyan_Classifier(512, 5)
            The optional  nn.Module to be used as additional top layers.

        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
            The loss function to use. It should accept two Tensors as inputs (predictions, targets) and return
            a Tensor with the loss.

        metrics: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]
            Dictionary with the name of the metric and a function to compute the metric from two tensors,
            prediction and true labels.

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
        Wei, Xiaoyan, et al. "Early prediction of epileptic seizures using a long-term recurrent convolutional
        network." Journal of neuroscience methods 327 (2019): 108395.
   """
    def __init__(self,
                 in_features: int,
                 top_module: Optional[nn.Module] = WeiXiaoyan_Classifier(512, 5),
                 loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = nn.CrossEntropyLoss(),
                 metrics: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
                 optimizer: torch.optim.Optimizer = torch.optim.Adam,
                 **kwargs
                 ):
        super(WeiXiaoyan, self).__init__(in_features, top_module, loss, metrics, optimizer, **kwargs)

        self.convolutions = nn.Sequential(
            nn.Conv1d(in_channels=in_features, out_channels=32, kernel_size=5),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.BatchNorm1d(num_features=32),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.BatchNorm1d(num_features=64),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.BatchNorm1d(num_features=128),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.BatchNorm1d(num_features=256),
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.BatchNorm1d(num_features=512),
        )
        self.lstm1 = nn.LSTM(input_size=512, hidden_size=512, batch_first=True)
        self.batchNorm = nn.BatchNorm1d(num_features=512)
        self.lstm2 = nn.LSTM(input_size=512, hidden_size=512, batch_first=True)

    def forward(self, x):
        x = self.convolutions(x)
        x = flip_indices_for_conv_to_lstm(x)
        x, _ = self.lstm1(x)
        x = x.reshape((x.size(0), x.size(2), x.size(1)))
        x = self.batchNorm(x)
        x = flip_indices_for_conv_to_lstm(x)
        x, _ = self.lstm2(x)

        if self.classifier is not None:
            x = self.classifier(x[:, -1, :])
        return x


class GaoJunLi_Classifier(nn.Module):
    """
    Classifier of the Gao Jun Li model.

    Parameters
    ----------
        in_features: int
            Number of features of the input tensors

        n_classes: int
            Number of classes to predict at the end of the network

        return_sequence: bool, defaults=True
            Parameter that controls wether the module returns the sequence or the prediction.

    Returns
    -------
    `LightningModule`
        A pyTorch Lightning Module instance.
    """
    def __init__(self, in_features, n_classes, return_sequence=False):
        super(GaoJunLi_Classifier, self).__init__()

        self.return_sequence = return_sequence
        self.module = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes)
        )

    def forward(self, x):
        if self.return_sequence:
            return self.module(x)
        else:
            return self.module(x[:,-1,:])


class GaoJunLi(TSFEDL_BaseModule):
    """
    LSTM network

    Parameters
    ----------
        in_features: int
            Number of features of the input tensors

        top_module: nn.Module, defaults=GaoJunLi_Classifier(64, 5)
            The optional  nn.Module to be used as additional top layers.

        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
            The loss function to use. It should accept two Tensors as inputs (predictions, targets) and return
            a Tensor with the loss.

        metrics: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]
            Dictionary with the name of the metric and a function to compute the metric from two tensors,
            prediction and true labels.

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
        Gao, Junli, et al. "An Effective LSTM Recurrent Network to Detect Arrhythmia on Imbalanced ECG Dataset."
        Journal of healthcare engineering 2019 (2019).
   """
    def __init__(self,
                 in_features: int,
                 top_module: Optional[nn.Module] = GaoJunLi_Classifier(64, 5),
                 loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = nn.CrossEntropyLoss(),
                 metrics: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
                 optimizer: torch.optim.Optimizer = torch.optim.Adam,
                 **kwargs
                 ):
        super(GaoJunLi, self).__init__(in_features, top_module, loss, optimizer, **kwargs)
        self.lstm = nn.LSTM(input_size=in_features, hidden_size=64, dropout=0.3)

    def forward(self, x):
        x = flip_indices_for_conv_to_lstm(x)
        x, _ = self.lstm(x)
        if self.classifier is not None:
            x = self.classifier(x)
        return x


class LiOhShu_Classifier(nn.Module):
    """
    Classifier of the Li Oh Shu model.

    Parameters
    ----------
        in_features: int
            Number of features of the input tensors

        n_classes: int
            Number of classes to predict at the end of the network

    Returns
    -------
    `LightningModule`
        A pyTorch Lightning Module instance.
    """
    def __init__(self, in_features, n_classes, return_sequence = False):
        super(LiOhShu_Classifier, self).__init__()
        self.return_sequence = return_sequence
        self.module = nn.Sequential(
            nn.Linear(in_features, 8),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(8, n_classes)
        )

    def forward(self, x):
        if self.return_sequence:
            return self.module(x)
        else:
            return self.module(x[:, -1, :])


class LihOhShu(TSFEDL_BaseModule):
    """
    CNN+LSTM model

    Parameters
    ----------
        in_features: int
            Number of features of the input tensors

        top_module: nn.Module, defaults=LiOhShu_Classifier(10, 5)
            The optional  nn.Module to be used as additional top layers.

        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
            The loss function to use. It should accept two Tensors as inputs (predictions, targets) and return
            a Tensor with the loss.

        metrics: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]
            Dictionary with the name of the metric and a function to compute the metric from two tensors,
            prediction and true labels.

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
        Lih, Oh Shu, et al. "Comprehensive electrocardiographic diagnosis based on deep learning." Artificial
        Intelligence in Medicine 103 (2020): 101789.
   """
    def __init__(self,
                 in_features: int,
                 top_module: Optional[nn.Module] = LiOhShu_Classifier(10, 5),
                 loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = nn.CrossEntropyLoss(),
                 metrics: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
                 optimizer: torch.optim.Optimizer = torch.optim.Adam,
                 **kwargs
                 ):
        super(LihOhShu, self).__init__(in_features, top_module, loss, metrics, optimizer, **kwargs)

        self.convolutions = nn.Sequential(
            nn.Conv1d(in_channels=in_features, out_channels=3, kernel_size=20, bias=False),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=3, out_channels=6, kernel_size=10, bias=False),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=6, out_channels=6, kernel_size=5, bias=False),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=6, out_channels=6, kernel_size=5, bias=False),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=6, out_channels=6, kernel_size=10, bias=False),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.lstm = nn.LSTM(input_size=6, hidden_size=10, batch_first=True)

    def forward(self, x):
        x = self.convolutions(x)
        x = flip_indices_for_conv_to_lstm(x)
        x, _ = self.lstm(x)

        if self.classifier is not None:
            x = self.classifier(x)
        return x

#####################################################################

class KhanZulfiqar_Classifier(nn.Module):
    """
    Classifier of the Khan Zulfiqar model.

    Parameters
    ----------
        in_features: int
            Number of features of the input tensors

        n_classes: int
            Number of classes to predict at the end of the network

        return_sequence: bool, defaults=True
            Parameter that controls wether the module returns the sequence or the prediction.

    Returns
    -------
    `LightningModule`
        A pyTorch Lightning Module instance.
    """
    def __init__(self, in_features, n_classes, return_sequence = False):
        super(KhanZulfiqar_Classifier, self).__init__()
        self.return_sequence = return_sequence
        self.module = nn.Sequential(
            nn.Linear(in_features, 8),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(8, n_classes)
        )

    def forward(self, x):
        if self.return_sequence:
            return self.module(x)
        else:
            return self.module(x[:, -1, :])


class KhanZulfiqar(TSFEDL_BaseModule):
    """
    CNN+GRU model for electricity forecasting

    Parameters
    ----------
        in_features: int
            Number of features of the input tensors

        top_module: nn.Module, defaults=KhanZulfiqar_Classifier(10, 5)
            The optional  nn.Module to be used as additional top layers.

        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
            The loss function to use. It should accept two Tensors as inputs (predictions, targets) and return
            a Tensor with the loss.

        metrics: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]
            Dictionary with the name of the metric and a function to compute the metric from two tensors,
            prediction and true labels.

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
        Sajjad, M., Khan, Z. A., Ullah, A., Hussain, T., Ullah, W., Lee, M. Y., & Baik, S. W. (2020). A novel
        CNN-GRU-based hybrid approach for short-term residential load forecasting. IEEE Access, 8, 143759-143768.
   """
    def __init__(self,
                 in_features: int,
                 top_module: Optional[nn.Module] = KhanZulfiqar_Classifier(10, 5),
                 loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = nn.CrossEntropyLoss(),
                 metrics: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
                 optimizer: torch.optim.Optimizer = torch.optim.Adam,
                 **kwargs
                 ):
        super(KhanZulfiqar, self).__init__(in_features, top_module, loss, metrics, optimizer, **kwargs)

        self.convolutions = nn.Sequential(
            nn.Conv1d(in_channels=in_features, out_channels=3, kernel_size=20, bias=False),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=3, out_channels=6, kernel_size=10, bias=False),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=6, out_channels=6, kernel_size=5, bias=False),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=6, out_channels=6, kernel_size=5, bias=False),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=6, out_channels=6, kernel_size=10, bias=False),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.lstm = nn.LSTM(input_size=6, hidden_size=10, batch_first=True)

    def forward(self, x):
        x = self.convolutions(x)
        x = flip_indices_for_conv_to_lstm(x)
        x, _ = self.lstm(x)

        if self.classifier is not None:
            x = self.classifier(x)
        return x

class ZhengZhenyu_Classifier(nn.Module):
    """
    Classifier of the Zheng Zhenyu model.

    Parameters
    ----------
        in_features: int
            Number of features of the input tensors

        n_classes: int
            Number of classes to predict at the end of the network

        return_sequence: bool, defaults=True
            Parameter that controls wether the module returns the sequence or the prediction.

    Returns
    -------
    `LightningModule`
        A pyTorch Lightning Module instance.
    """
    def __init__(self, in_features, n_classes, return_sequence = False):
        super(ZhengZhenyu_Classifier, self).__init__()
        self.return_sequence = return_sequence
        self.module = nn.Sequential(
            nn.Linear(in_features, 2048),
            nn.ELU(),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ELU(),
            nn.Linear(2048, n_classes)
        )

    def forward(self, x):
        if self.return_sequence:
            return self.module(x)
        else:
            return self.module(x[:, -1, :])


class ZhengZhenyu(TSFEDL_BaseModule):
    """
    CNN+LSTM network for arrythmia detection. This model initialy was designed to deal with 2D images. It was adapted
    to 1D time series.

    Parameters
    ----------
        in_features: int
            Number of features of the input tensors

        top_module: nn.Module, defaults=ZhengZhenyu_Classifier(256, 5)
            The optional  nn.Module to be used as additional top layers.

        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
            The loss function to use. It should accept two Tensors as inputs (predictions, targets) and return
            a Tensor with the loss.

        metrics: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]
            Dictionary with the name of the metric and a function to compute the metric from two tensors,
            prediction and true labels.

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
        Zheng, Z., Chen, Z., Hu, F., Zhu, J., Tang, Q., & Liang, Y. (2020). An automatic diagnosis of arrhythmias using
        a combination of CNN and LSTM technology. Electronics, 9(1), 121.
   """
    def __init__(self,
                 in_features: int,
                 top_module: Optional[nn.Module] = ZhengZhenyu_Classifier(256, 5),
                 loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = nn.CrossEntropyLoss(),
                 metrics: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
                 optimizer: torch.optim.Optimizer = torch.optim.Adam,
                 **kwargs
                 ):
        super(ZhengZhenyu, self).__init__(in_features, top_module, loss, metrics, optimizer, **kwargs)

        self.convolutions = nn.Sequential(
            nn.Conv1d(in_channels=in_features, out_channels=64, kernel_size=3, stride=1, bias=True, padding="same"),
            nn.ELU(),
            nn.BatchNorm1d(num_features=64),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, bias=True, padding="same"),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, bias=True, padding="same"),
            nn.ELU(),
            nn.BatchNorm1d(num_features=128),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=1, bias=True, padding="same"),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, bias=True, padding="same"),
            nn.ELU(),
            nn.BatchNorm1d(num_features=256),
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1, bias=True, padding="same"),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.lstm = nn.LSTM(input_size=256, hidden_size=256, batch_first=True)

    def forward(self, x):
        x = self.convolutions(x)
        x = flip_indices_for_conv_to_lstm(x)
        x, _ = self.lstm(x)

        if self.classifier is not None:
            x = self.classifier(x)
        return x

#
# class HouBoroui(TSFEDL_BaseModule):
#     def __init__(self,
#                  in_features: int,
#                  encoder_units: int = 100,
#                  top_module: Optional[nn.Module] = None,
#                  loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = nn.CrossEntropyLoss(),
#                  metrics: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
#                  optimizer: torch.optim.Optimizer = torch.optim.Adam,
#                  **kwargs
#                  ):
#         super(HouBoroui, self).__init__(in_features, top_module, loss, metrics, optimizer, **kwargs)
#
#         self.lstm1 = nn.LSTM(input_size=in_features, hidden_size=encoder_units, batch_first=True)
#         self.lstm2 = nn.LSTM(input_size=encoder_units, hidden_size=encoder_units, batch_first=True)
#         self.td = TimeDistributed(nn.Sequential(nn.Linear(encoder_units,1)))
#
#         # self.encoder = nn.Sequential(
#         #     nn.LSTM(input_size=in_features, hidden_size=encoder_units, batch_first=True),
#         #     nn.LSTM(input_size=encoder_units, hidden_size=encoder_units, batch_first=True),
#         #     TimeDistributed(nn.Sequential(nn.Linear(encoder_units,1)))
#         # )
#
#     def forward(self, x):
#         x = flip_indices_for_conv_to_lstm(x)
#         x, _ = self.lstm1(x)
#         x, _ = self.lstm2(x)
#         x = self.td(x)
#
#         if self.classifier is not None:
#             x = self.classifier(x)
#
#         return x

class WangKejun_Classifier(nn.Module):
    """
    Classifier of the Wang Kejun model.

    Parameters
    ----------
        in_features: int
            Number of features of the input tensors

        n_classes: int
            Number of classes to predict at the end of the network

        return_sequence: bool, defaults=True
            Parameter that controls wether the module returns the sequence or the prediction.

    Returns
    -------
    `LightningModule`
        A pyTorch Lightning Module instance.
    """
    def __init__(self, in_features, n_classes, return_sequence = False):
        super(WangKejun_Classifier, self).__init__()
        self.return_sequence = return_sequence
        self.module = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 2048),
            nn.ELU(),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ELU(),
            nn.Linear(2048, n_classes)
        )

    def forward(self, x):
        if self.return_sequence:
            return self.module(x)
        else:
            return self.module(x[:, -1, :])


class WangKejun(TSFEDL_BaseModule):
    """
    CNN 1D-LSTM

    Parameters
    ----------
        in_features: int
            Number of features of the input tensors

        top_module: nn.Module, defaults=WangKejun_Classifier(256, 5)
            The optional  nn.Module to be used as additional top layers.

        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
            The loss function to use. It should accept two Tensors as inputs (predictions, targets) and return
            a Tensor with the loss.

        metrics: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]
            Dictionary with the name of the metric and a function to compute the metric from two tensors,
            prediction and true labels.

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
        Wang, Kejun, Xiaoxia Qi, and Hongda Liu. "Photovoltaic power forecasting based LSTM-Convolutional Network."
        Energy 189 (2019): 116225.
   """
    def __init__(self,
                 in_features: int,
                 top_module: Optional[nn.Module] = WangKejun_Classifier(256, 5),
                 loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = nn.CrossEntropyLoss(),
                 metrics: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
                 optimizer: torch.optim.Optimizer = torch.optim.Adam,
                 **kwargs
                 ):
        super(WangKejun, self).__init__(in_features, top_module, loss, metrics, optimizer, **kwargs)

        self.convolutions = nn.Sequential(
            nn.Conv1d(in_channels=in_features, out_channels=64, kernel_size=3, stride=1, bias=True, padding="same"),
            nn.ELU(),
            nn.BatchNorm1d(num_features=64),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, bias=True, padding="same"),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, bias=True, padding="same"),
            nn.ELU(),
            nn.BatchNorm1d(num_features=128),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=1, bias=True, padding="same"),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, bias=True, padding="same"),
            nn.ELU(),
            nn.BatchNorm1d(num_features=256),
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1, bias=True, padding="same"),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.lstm = nn.LSTM(input_size=256, hidden_size=256, batch_first=True)

    def forward(self, x):
        x = self.convolutions(x)
        x = flip_indices_for_conv_to_lstm(x)
        x, _ = self.lstm(x)

        if self.classifier is not None:
            x = self.classifier(x)
        return x

class ChenChen_Classifier(nn.Module):
    """
    Classifier of the Chen Chen model.

    Parameters
    ----------
        in_features: int
            Number of features of the input tensors

        n_classes: int
            Number of classes to predict at the end of the network

        return_sequence: bool, defaults=True
            Parameter that controls wether the module returns the sequence or the prediction.

    Returns
    -------
    `LightningModule`
        A pyTorch Lightning Module instance.
    """
    def __init__(self, in_features, n_classes, return_sequence = False):
        super(ChenChen_Classifier, self).__init__()
        self.return_sequence = return_sequence
        self.module = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, n_classes),
            nn.Softmax()
        )

    def forward(self, x):
        if self.return_sequence:
            return self.module(x)
        else:
            return self.module(x[:, -1, :])


class ChenChen(TSFEDL_BaseModule):
    """
    CNN+LSTM model for arrythmia classification.

    Parameters
    ----------
        in_features: int
            Number of features of the input tensors

        top_module: nn.Module, defaults=ChenChen_Classifier(64, 5)
            The optional  nn.Module to be used as additional top layers.

        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
            The loss function to use. It should accept two Tensors as inputs (predictions, targets) and return
            a Tensor with the loss.

        metrics: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]
            Dictionary with the name of the metric and a function to compute the metric from two tensors,
            prediction and true labels.

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
        Chen, Chen, et al. "Automated arrhythmia classification based on a combination network of CNN and LSTM."
        Biomedical Signal Processing and Control 57 (2020): 101819.
   """
    def __init__(self,
                 in_features: int,
                 top_module: Optional[nn.Module] = ChenChen_Classifier(64, 5),
                 loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = nn.CrossEntropyLoss(),
                 metrics: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
                 optimizer: torch.optim.Optimizer = torch.optim.Adam,
                 **kwargs
                 ):
        super(ChenChen, self).__init__(in_features, top_module, loss, metrics, optimizer, **kwargs)

        self.convolutions = nn.Sequential(
            nn.Conv1d(in_channels=in_features, out_channels=251, kernel_size=5, stride=1, bias=True, padding="valid"),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=251, out_channels=150, kernel_size=5, stride=1, bias=True, padding="valid"),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=150, out_channels=100, kernel_size=10, stride=1, bias=True, padding="valid"),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=100, out_channels=81, kernel_size=20, stride=1, bias=True, padding="valid"),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=81, out_channels=61, kernel_size=20, stride=1, bias=True, padding="valid"),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=61, out_channels=14, kernel_size=10, stride=1, bias=True, padding="valid"),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        self.lstm1 = nn.LSTM(input_size=14, hidden_size=32, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=32, hidden_size=64, batch_first=True)

    def forward(self, x):
        x = self.convolutions(x)
        x = flip_indices_for_conv_to_lstm(x)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)

        if self.classifier is not None:
            x = self.classifier(x)
        return x


class KimTaeYoung_Classifier(nn.Module):
    """
    Classifier of the Kim Tae Young model.

    Parameters
    ----------
        in_features: int
            Number of features of the input tensors

        n_classes: int
            Number of classes to predict at the end of the network

        return_sequence: bool, defaults=True
            Parameter that controls wether the module returns the sequence or the prediction.

    Returns
    -------
    `LightningModule`
        A pyTorch Lightning Module instance.
    """
    def __init__(self, in_features, n_classes, return_sequence = False):
        super(KimTaeYoung_Classifier, self).__init__()
        self.return_sequence = return_sequence
        self.module = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes),
            nn.Softmax()
        )

    def forward(self, x):
        if self.return_sequence:
            return self.module(x)
        else:
            return self.module(x[:, -1, :])


class KimTaeYoung(TSFEDL_BaseModule):
    """
    CNN+LSTM model for arrythmia classification.

    Parameters
    ----------
        in_features: int
            Number of features of the input tensors

        top_module: nn.Module, defaults=KimTaeYoung_Classifier(64, 5)
            The optional  nn.Module to be used as additional top layers.

        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
            The loss function to use. It should accept two Tensors as inputs (predictions, targets) and return
            a Tensor with the loss.

        metrics: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]
            Dictionary with the name of the metric and a function to compute the metric from two tensors,
            prediction and true labels.

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
        Kim, Tae-Young, and Sung-Bae Cho. "Predicting residential energy consumption using CNN-LSTM neural networks."
        Energy 182 (2019): 72-81.

    Notes
    -----
        In the original paper, the time-series is windowed. Therefore, a TimeDistributed layer is employed before the
        LSTM to traverse all the generated windows. Here, we do not implement this as it is problem-specific.
        Please note that if you need this layer then you should add it manually.
   """
    def __init__(self,
                 in_features: int,
                 top_module: Optional[nn.Module] = KimTaeYoung_Classifier(64, 5),
                 loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = nn.CrossEntropyLoss(),
                 metrics: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
                 optimizer: torch.optim.Optimizer = torch.optim.Adam,
                 **kwargs
                 ):
        super(KimTaeYoung, self).__init__(in_features, top_module, loss, metrics, optimizer, **kwargs)

        self.convolutions = nn.Sequential(
            nn.Conv1d(in_channels=in_features, out_channels=64, kernel_size=2, stride=1, bias=True, padding="valid"),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=2, stride=1, bias=True, padding="valid"),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1)
        )

        self.lstm = nn.LSTM(input_size=64, hidden_size=64, batch_first=True)

    def forward(self, x):
        x = self.convolutions(x)
        x = flip_indices_for_conv_to_lstm(x)
        x, _ = self.lstm(x)

        if self.classifier is not None:
            x = self.classifier(x)
        return x


class GenMinxing_Classifier(nn.Module):
    """
    Classifier of the Gen Minxing model.

    Parameters
    ----------
        in_features: int
            Number of features of the input tensors

        n_classes: int
            Number of classes to predict at the end of the network

        return_sequence: bool, defaults=True
            Parameter that controls wether the module returns the sequence or the prediction.

    Returns
    -------
    `LightningModule`
        A pyTorch Lightning Module instance.
    """
    def __init__(self, in_features, n_classes, return_sequence = False):
        super(GenMinxing_Classifier, self).__init__()
        self.return_sequence = return_sequence
        self.module = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, n_classes),
            nn.Softmax()
        )

    def forward(self, x):
        if self.return_sequence:
            return self.module(x)
        else:
            return self.module(x[:, -1, :])


class GenMinxing(TSFEDL_BaseModule):
    """
    CNN+LSTM model for arrythmia classification.

    Parameters
    ----------
        in_features: int
            Number of features of the input tensors

        top_module: nn.Module, defaults=GenMinxing_Classifier(80, 5)
            The optional  nn.Module to be used as additional top layers.

        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
            The loss function to use. It should accept two Tensors as inputs (predictions, targets) and return
            a Tensor with the loss.

        metrics: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]
            Dictionary with the name of the metric and a function to compute the metric from two tensors,
            prediction and true labels.

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
        Geng, Minxing, et al. "Epileptic Seizure Detection Based on Stockwell Transform and Bidirectional Long
        Short-Term Memory." IEEE Transactions on Neural Systems and Rehabilitation Engineering 28.3 (2020): 573-580.
   """
    def __init__(self,
                 in_features: int,
                 top_module: Optional[nn.Module] = GenMinxing_Classifier(80, 5),
                 loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = nn.CrossEntropyLoss(),
                 metrics: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
                 optimizer: torch.optim.Optimizer = torch.optim.Adam,
                 **kwargs
                 ):
        super(GenMinxing, self).__init__(in_features, top_module, loss, metrics, optimizer, **kwargs)

        self.lstm = nn.LSTM(input_size=in_features, hidden_size=40, batch_first=True, bidirectional=True)

    def forward(self, x):
        x, _ = self.lstm(x)

        if self.classifier is not None:
            x = self.classifier(x)
        return x


class FuJiangmeng_Classifier(nn.Module):
    """
    Classifier of the Fu Jiangmeng model.

    Parameters
    ----------
        in_features: int
            Number of features of the input tensors

        n_classes: int
            Number of classes to predict at the end of the network

        return_sequence: bool, defaults=True
            Parameter that controls wether the module returns the sequence or the prediction.

    Returns
    -------
    `LightningModule`
        A pyTorch Lightning Module instance.
    """
    def __init__(self, in_features, n_classes, return_sequence = False):
        super(FuJiangmeng_Classifier, self).__init__()
        self.return_sequence = return_sequence
        self.module = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.Tanh(),
            nn.Dropout(p=0.3),
            nn.Linear(128, n_classes),
            nn.Softmax()
        )

    def forward(self, x):
        if self.return_sequence:
            return self.module(x)
        else:
            return self.module(x[:, -1, :])


class FuJiangmeng(TSFEDL_BaseModule):
    """
    CNN+LSTM model for arrythmia classification.

    Parameters
    ----------
        in_features: int
            Number of features of the input tensors

        top_module: nn.Module, defaults=FuJiangmeng_Classifier(256, 5)
            The optional  nn.Module to be used as additional top layers.

        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
            The loss function to use. It should accept two Tensors as inputs (predictions, targets) and return
            a Tensor with the loss.

        metrics: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]
            Dictionary with the name of the metric and a function to compute the metric from two tensors,
            prediction and true labels.

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
        Fu, Jiangmeng, et al. "A hybrid CNN-LSTM model based actuator fault diagnosis for six-rotor UAVs." 2019
        Chinese Control And Decision Conference (CCDC). IEEE, 2019.
   """
    def __init__(self,
                 in_features: int,
                 top_module: Optional[nn.Module] = FuJiangmeng_Classifier(256, 5),
                 loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = nn.CrossEntropyLoss(),
                 metrics: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
                 optimizer: torch.optim.Optimizer = torch.optim.Adam,
                 **kwargs
                 ):
        super(FuJiangmeng, self).__init__(in_features, top_module, loss, metrics, optimizer, **kwargs)

        self.convolutions = nn.Sequential(
            nn.Conv1d(in_channels=in_features, out_channels=32, kernel_size=1, stride=1, bias=True, padding="same"),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2, stride=1),
        )

        self.lstm = nn.LSTM(input_size=32, hidden_size=256, batch_first=True, dropout=0.3)

    def forward(self, x):
        x = self.convolutions(x)
        x = flip_indices_for_conv_to_lstm_reshape(x)
        x, _ = self.lstm(x)

        if self.classifier is not None:
            x = self.classifier(x)
        return x


class ShiHaotian_Classifier(nn.Module):
    """
    Classifier of the Shi Haotian model.

    Parameters
    ----------
        in_features: int
            Number of features of the input tensors

        n_classes: int
            Number of classes to predict at the end of the network

        in_features_original: int
            Number of features of the original tensors in the main architecture

        return_sequence: bool, defaults=True
            Parameter that controls wether the module returns the sequence or the prediction.

    Returns
    -------
    `LightningModule`
        A pyTorch Lightning Module instance.
    """
    def __init__(self, in_features, n_classes, in_features_original, return_sequence = False):
        super(ShiHaotian_Classifier, self).__init__()
        self.return_sequence = return_sequence
        self.module1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 518),
            nn.ReLU(),
            nn.Linear(518, 88),
            nn.ReLU()
        )
        self.module2 = nn.Sequential(
            nn.Linear(88+32, n_classes),
            nn.Softmax()
        )

    def forward(self, x):
        if self.return_sequence:
            x1 = self.module1(x)
            x_conc = torch.cat((x1, x[:,-1,:]), 1)
            return self.module2(x_conc)
        else:
            x1 = self.module1(x)
            x_conc = torch.cat((x1, x[:,-1,:]), 1)
            return self.module2(x_conc)


class ShiHaotian(TSFEDL_BaseModule):
    """
    CNN+LSTM model for arrythmia classification.

    Parameters
    ----------
        in_features: int
            Number of features of the input tensors

        top_module: nn.Module, defaults=None
            The optional  nn.Module to be used as additional top layers.

        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
            The loss function to use. It should accept two Tensors as inputs (predictions, targets) and return
            a Tensor with the loss.

        metrics: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]
            Dictionary with the name of the metric and a function to compute the metric from two tensors,
            prediction and true labels.

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
        Shi, Haotian, et al. "Automated heartbeat classification based on deep neural network with multiple input
        layers." Knowledge-Based Systems 188 (2020): 105036.
   """
    def __init__(self,
                 in_features: int,
                 top_module: Optional[nn.Module] = None,
                 loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = nn.CrossEntropyLoss(),
                 metrics: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
                 optimizer: torch.optim.Optimizer = torch.optim.Adam,
                 **kwargs
                 ):
        super(ShiHaotian, self).__init__(in_features, top_module, loss, metrics, optimizer, **kwargs)
        if self.classifier is None:
            self.classifier = ShiHaotian_Classifier(7904, 5, in_features)

        self.convolutions1 = nn.Sequential(
            nn.Conv1d(in_channels=in_features, out_channels=32, kernel_size=13, stride=2, bias=True, padding="valid"),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.convolutions2 = nn.Sequential(
            nn.Conv1d(in_channels=in_features, out_channels=32, kernel_size=13, stride=1, bias=True, padding="valid"),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.convolutions3 = nn.Sequential(
            nn.Conv1d(in_channels=in_features, out_channels=32, kernel_size=13, stride=2, bias=True, padding="valid"),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.lstm = nn.LSTM(input_size=32*3, hidden_size=32, batch_first=True)

    def forward(self, x):
        x1 = self.convolutions1(x)
        x2 = self.convolutions1(x)
        x3 = self.convolutions1(x)
        x_conc = torch.cat((x1, x2, x3), 1)
        x_conc = flip_indices_for_conv_to_lstm(x_conc)
        x, _ = self.lstm(x_conc)

        if self.classifier is not None:
            x = self.classifier(x)
        return x


class HuangMeiLing_Classifier(nn.Module):
    """
    Classifier of the Huang Mei Ling model.

    Parameters
    ----------
        in_features: int
            Number of features of the input tensors

        n_classes: int
            Number of classes to predict at the end of the network

        return_sequence: bool, defaults=True
            Parameter that controls wether the module returns the sequence or the prediction.

    Returns
    -------
    `LightningModule`
        A pyTorch Lightning Module instance.
    """
    def __init__(self, in_features, n_classes, return_sequence = False):
        super(HuangMeiLing_Classifier, self).__init__()
        self.return_sequence = return_sequence
        self.module = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, n_classes),
            nn.Softmax()
        )

    def forward(self, x):
        if self.return_sequence:
            return self.module(x)
        else:
            return self.module(x[:, -1, :])


class HuangMeiLing(TSFEDL_BaseModule):
    """
     CNN model, employed for 2-D images of ECG data. This model is adapted for 1-D time series

    Parameters
    ----------
        in_features: int
            Number of features of the input tensors

        top_module: nn.Module, defaults=None
            The optional  nn.Module to be used as additional top layers.

        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
            The loss function to use. It should accept two Tensors as inputs (predictions, targets) and return
            a Tensor with the loss.

        metrics: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]
            Dictionary with the name of the metric and a function to compute the metric from two tensors,
            prediction and true labels.

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
        Huang, Mei-Ling, and Yan-Sheng Wu. "Classification of atrial fibrillation and normal sinus rhythm based on
        convolutional neural network." Biomedical Engineering Letters (2020): 1-11.
   """
    def __init__(self,
                 in_features: int,
                 top_module: Optional[nn.Module] = HuangMeiLing_Classifier(81, 5),
                 loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = nn.CrossEntropyLoss(),
                 metrics: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
                 optimizer: torch.optim.Optimizer = torch.optim.Adam,
                 **kwargs
                 ):
        super(HuangMeiLing, self).__init__(in_features, top_module, loss, metrics, optimizer, **kwargs)

        self.convolutions = nn.Sequential(
            nn.ConstantPad1d(padding=1, value=0),
            nn.Conv1d(in_channels=in_features, out_channels=48, kernel_size=15, stride=6, bias=True, padding="valid"),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1),
            nn.ConstantPad1d(padding=3, value=0),
            nn.Conv1d(in_channels=48, out_channels=256, kernel_size=7, stride=2, bias=True, padding="valid"),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1),
        )

    def forward(self, x):
        x = self.convolutions(x)

        if self.classifier is not None:
            x = self.classifier(x)
        return x

class HongTan_Classifier(nn.Module):
    """
    Classifier of the HongTan model.

    Parameters
    ----------
        in_features: int
            Number of features of the input tensors

        n_classes: int
            Number of classes to predict at the end of the network

    Returns
    -------
    `LightningModule`
        A pyTorch Lightning Module instance.
    """

    def __init__(self, in_features, n_classes):
        super(HongTan_Classifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=n_classes),
            nn.Softmax()
        )

    def forward(self, x):
        return self.model(x)


class HongTan(TSFEDL_BaseModule):
    """
    Application of stacked convolutional and long short-term memory network
    for accurate identification of CAD ECG signals

    Parameters
    ----------
        in_features: int
            Number of features of the input tensors

        top_module: nn.Module, defaults=HongTan_Classifier(4,5)
            The optional  nn.Module to be used as additional top layers.

        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
            The loss function to use. It should accept two Tensors as inputs (predictions, targets) and return
            a Tensor with the loss.

        metrics: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]
            Dictionary with the name of the metric and a function to compute the metric from two tensors,
            prediction and true labels.

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
        TAN, Jen Hong, et al. Application of stacked convolutional and long short-term memory network for accurate
        identification of CAD ECG signals. Computers in biology and medicine, 2018, vol. 94, p. 19-26.
    """
    def __init__(self,
                 in_features: int,
                 top_module: Optional[nn.Module] = HongTan_Classifier(in_features=4, n_classes=5),
                 loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = nn.CrossEntropyLoss(),
                 metrics: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
                 optimizer: torch.optim.Optimizer = torch.optim.Adam,
                 **kwargs
                 ):
        super(HongTan, self).__init__(in_features, top_module, loss, metrics, optimizer, **kwargs)

        conv_layers = []

        conv_layers.append(nn.Conv1d(in_channels=in_features, out_channels=40, kernel_size=5, bias=False, stride=1))
        conv_layers.append(nn.ReLU())
        conv_layers.append(nn.MaxPool1d(kernel_size=2, stride=2))

        # The remaining convolutional layers can be normal ones: we know the input size.
        conv_layers.append(nn.Conv1d(in_channels=40, out_channels=32, kernel_size=3, bias=False, stride=1))
        conv_layers.append(nn.ReLU())
        conv_layers.append(nn.MaxPool1d(kernel_size=2, stride=2))

        self.convolutions = nn.Sequential(*conv_layers)

        self.lstm1 = nn.LSTM(input_size=32, hidden_size=32, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=32, hidden_size=16, batch_first=True)
        self.lstm3 = nn.LSTM(input_size=16, hidden_size=4, batch_first=True)


    def forward(self, x):
        out = self.convolutions(x)
        # Now, flip indices using a view for the LSTM as it requires a shape of (N, L, H_in = C)
        out = out.view(out.size(0), out.size(2), out.size(1))
        out, _ = self.lstm1(out, batch_first=True)
        out, _ = self.lstm2(out, batch_first=True)
        out, _ = self.lstm3(out, batch_first=True)

        if self.classifier is not None:
            out = self.classifier(out[:, -1, :])  # We only want the last step of the LSTM output

        return out
