from __future__ import print_function, division, absolute_import
from mmcv.cnn import normal_init
import torch.nn as nn
from .builder import ATTENTIONS
from functools import reduce

@ATTENTIONS.register_module
class BAM(nn.Module):
    def __init__(self, inplanes, reduction=16, bias=True):
        super(BAM, self).__init__()
        self.inplanes = inplanes
        self.outplanes = inplanes
        self.reduction = reduction
        self.bias = bias
        self.init_layers()
        self.fp16_enabled = False

    def init_layers(self):
        self.channel = channelWise(self.inplanes, self.reduction, self.bias)
        self.spatial = spatialWise(self.inplanes, self.reduction, dialation_num=2, dialation_val=4)
        self.sigmoid = nn.Sigmoid()

    def init_weights(self):
        self.channel.init_weights()
        self.spatial.init_weights()

    def forward(self, x):
        channel = self.channel(x)
        spatial = self.spatial(x)
        Mf = self.sigmoid(channel + spatial)

        output = x * Mf + x

        return output

#channel of BAM
class channelWise(nn.Module):
    def __init__(self, inplanes, r, bias=True):
        super(channelWise, self).__init__()
        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Sequential(
            nn.Conv2d(inplanes, inplanes // r, kernel_size=1, stride=1, bias=True),
            nn.BatchNorm2d(inplanes // r),
            nn.ReLU(inplace=True)
            )
        self.fc2 = nn.Sequential(
            nn.Conv2d(inplanes // r, inplanes, kernel_size=1, stride=1, bias=True),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True)
            )

    def init_weights(self):
        normal_init(self.fc1[0], std=0.01)
        normal_init(self.fc2[0], std=0.01)

    def forward(self, x):
        return self.fc2(self.fc1(self.GAP(x))).expand_as(x)


class spatialWise(nn.Module):
    def __init__(self, inplanes, r, dialation_num=2, dialation_val=4, bias=True):
        super(spatialWise, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Conv2d(inplanes, inplanes // r,
                      kernel_size=1, stride=1, bias=bias),
            nn.BatchNorm2d(inplanes // r),
            nn.ReLU(inplace=True)
            )
        for i in range(dialation_num):
            self.fc2 = nn.Sequential(
                nn.Conv2d(inplanes // r, inplanes // r, kernel_size=3,
                          padding=dialation_val, dilation=dialation_val),
                nn.BatchNorm2d(inplanes // r),
                nn.ReLU(inplace=True),
                )
        self.fc3 = nn.Conv2d(inplanes // r, 1, kernel_size=1, stride=1)

    def init_weights(self):
        normal_init(self.fc1[0], std=0.01)
        normal_init(self.fc2[0], std=0.01)
        normal_init(self.fc3, std=0.01)

    def forward(self, x):
        fc1 = self.fc1(x)
        fc2 = self.fc2(fc1)
        fc3 = self.fc3(fc2)
        return fc3.expand_as(x)
        return self.fc3(self.fc2(self.fc1(x))).expand_as(x)





