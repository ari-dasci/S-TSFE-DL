import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlockYiboGao(nn.Module):
    """
    Convolutional block of YiboGao's model
    """
    def __init__(self, nb_filter, kernel_size):
        super(ConvBlockYiboGao, self).__init__()
        self.kernel_size = kernel_size
        self.nb_filter = nb_filter

        self.module = nn.Sequential(
            nn.LazyConv1d(out_channels=nb_filter, kernel_size=kernel_size, padding='same'),
            nn.BatchNorm1d(nb_filter),
            nn.ReLU()
        )

    def forward(self, x):
        return self.module(x)


class AttentionBranchYiboGao(nn.Module):
    """
    Attention bronch of YiboGao's model
    """

    def __init__(self, nb_filter, kernel_size):
        super(AttentionBranchYiboGao, self).__init__()
        self.nb_filter = nb_filter
        self.kernel_size = kernel_size
        self.convBlock1 = ConvBlockYiboGao(nb_filter, kernel_size)
        self.convBlock2 = nn.Sequential(
            nn.MaxPool1d(kernel_size=2),
            ConvBlockYiboGao(nb_filter, kernel_size),
            nn.Upsample(scale_factor=2)
        )
        self.convBlock3 = ConvBlockYiboGao(nb_filter, kernel_size)
        self.finalBlock = nn.Sequential(
            ConvBlockYiboGao(nb_filter, kernel_size),
            nn.Conv1d(in_channels=nb_filter, out_channels=nb_filter, kernel_size=1, padding='same'),
            nn.BatchNorm1d(nb_filter),
            nn.Sigmoid()
        )

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
    """
    def __init__(self, nb_filter, kernel_size):
        super(RTABlock, self).__init__()
        self.nb_filter = nb_filter
        self.kernel_size = kernel_size

        self.conv1 = ConvBlockYiboGao(nb_filter, kernel_size)
        self.conv2 = ConvBlockYiboGao(nb_filter, kernel_size)
        self.attention = AttentionBranchYiboGao(nb_filter, kernel_size)
        self.conv3 = ConvBlockYiboGao(nb_filter, kernel_size)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        attention_map = self.attention(x1)

        x = torch.multiply(x2, attention_map)
        x = torch.add(x, x1)

        return self.conv3(x)
