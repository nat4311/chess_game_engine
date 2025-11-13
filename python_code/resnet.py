"""
TODO:
    preallocate giant np array for inputs?
        119x64xN where N is max plausible number of chess turns - can allocate more if needed in rare cases
        is this actually faster?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np

# 14*8 (12 bitboards + 2 repetition, 1 current + 7 past) + 7 (1 turn + 1 total_moves + 4 castling + 1 halfmove)
feature_channels = 14*8 + 7

# the following came from https://www.chessprogramming.org/AlphaZero#Network_Architecture
default_filters = 256
default_kernel_size = 3
res_block_layers = 19

class ResBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv0 = nn.Conv2d(in_channels=default_filters, out_channels=default_filters, kernel_size=default_kernel_size, stride=1, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(default_filters)
        self.conv1 = nn.Conv2d(in_channels=default_filters, out_channels=default_filters, kernel_size=default_kernel_size, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(default_filters)

    def forward(self, x):
        res = x
        out = self.conv0(x)
        out = self.bn0(out)
        out = F.relu(out)
        out = self.conv1(out)
        out = self.bn1(out)
        out += res
        out = F.relu(out)

        return out

class OutputHeads(nn.Module):
    """
    NOTE: policy head illegal move masking should be done after the output of the network

    outputs policy and value heads
    forward returns p,v
    """

    def __init__(self):
        super().__init__()

        self.p_conv0 = nn.Conv2d(in_channels=default_filters, out_channels=default_filters, kernel_size=default_kernel_size, stride=1, padding=1, bias=False)
        self.p_bn0 = nn.BatchNorm2d(default_filters)
        self.p_conv1 = nn.Conv2d(in_channels=default_filters, out_channels=73, kernel_size=default_kernel_size, stride=1, padding=1, bias=False)
        self.p_lsm = nn.LogSoftmax(dim=1)

        self.v_conv = nn.Conv2d(in_channels=default_filters, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False)
        self.v_bn = nn.BatchNorm2d(1)
        self.v_lin0 = nn.Linear(in_features=64, out_features=256)
        self.v_lin1 = nn.Linear(in_features=256, out_features=1)

    def forward(self, x):
        """
        returns p, v
        """

        p = self.p_conv0(x)
        p = self.p_bn0(p)
        p = F.relu(p)
        p = self.p_conv1(p)
        p = self.p_lsm(p).exp()

        v = self.v_conv(x)
        v = self.v_bn(v)
        v = F.relu(v)
        v = self.v_lin0(v)
        v = F.relu(v)
        v = self.v_lin1(v)
        v = F.tanh(v)

        return p, v


class ResNet(nn.Module):
    """
    NOTE: policy head illegal move masking should be done after the output of the network

    outputs policy and value heads
    forward returns p,v
    """
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels=feature_channels, out_channels=default_filters, kernel_size=default_kernel_size, stride=1, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(default_filters)
        for layer in range(res_block_layers):
            setattr(self, f"res_block_{layer}", ResBlock())
        self.output = OutputHeads()

    def forward(self, x):
        """
        returns p, v
        """

        # input conv layer
        out = self.conv0(x)
        out = self.bn0(out)
        out = F.relu(out)

        # ResBlocks
        for layer in range(res_block_layers):
            out = getattr(self, f"res_block_{layer}")(out)

        # policy and value head
        return self.output(out)
















