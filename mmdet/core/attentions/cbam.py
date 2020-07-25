from __future__ import print_function, division, absolute_import
from .builder import ATTENTIONS
from mmcv.cnn import normal_init
import torch
import torch.nn as nn


@ATTENTIONS.register_module
class CBAM(nn.Module):
    def __init__(self, inplanes, reduction=16, bias_c=True, bias_s=True, kernel_size=7):
        super(CBAM, self).__init__()
        self.inplanes = inplanes
        self.reduction = reduction
        self.bias_c = bias_c
        self.bias_s = bias_s
        self.kernel_size=kernel_size

        self.init_layer()
        self.fp16_enabled = False

    def init_layer(self):
        self.channel_layer = ChannelAttention(self.inplanes, self.reduction, self.bias_c)
        self.spatial_layer = SpatialAttention(self.kernel_size, self.bias_s)

    def init_weights(self):
        ChannelAttention.init_weighs(self)
        SpatialAttention.init_weighs(self)

    def forward(self, x):
        residual = x
        x = self.channel_layer(x) * x
        x = self.spatial_layer(x) * x
        return x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7, bias=True):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=bias)
        self.sigmoid = nn.Sigmoid()

    def init_weighs(self):
        normal_init(self.spatial_layer.conv1, std=0.01)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class ChannelAttention(nn.Module):
    def __init__(self, inplanes, redection=16, bias=True):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(inplanes, inplanes // redection, 1, bias=bias)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(inplanes // redection, inplanes, 1, bias=bias)

        self.sigmoid = nn.Sigmoid()

    def init_weighs(self):
        normal_init(self.channel_layer.fc1, std=0.01)
        normal_init(self.channel_layer.fc2, std=0.01)

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

