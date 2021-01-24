from __future__ import print_function, division, absolute_import
from .builder import ATTENTIONS
from mmcv.cnn import normal_init
import torch
import torch.nn as nn
import math


@ATTENTIONS.register_module
class MSCAM(nn.Module):
    def __init__(self, inplanes, r=16):
        super(MSCAM, self).__init__()
        self.inplanes = inplanes
        self.local_att = nn.Sequential(
            nn.Conv2d(self.inplanes, self.inplanes // r, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.inplanes // r),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inplanes // r, self.inplanes, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.inplanes),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.inplanes, self.inplanes // r, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.inplanes // r),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inplanes // r, self.inplanes, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.inplanes),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        return x * wei