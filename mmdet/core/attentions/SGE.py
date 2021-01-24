from __future__ import print_function, division, absolute_import
from mmcv.cnn import normal_init
import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
from .builder import ATTENTIONS

@ATTENTIONS.register_module
class SGE(nn.Module):
    def __init__(self, inplanes, groups=32):
        super(SGE, self).__init__()
        self.inplanes = inplanes
        self.groups = groups
        self.bias = Parameter(torch.ones(1, groups, 1, 1))
        self.weight = Parameter(torch.zeros(1, groups, 1, 1))
        self.init_layer()
        self.fp16_enabled = False

    def init_layer(self):
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        module_x = x
        b, c, h, w = x.size()
        x = x.view(b * self.groups, -1, h, w)  # (b*32, c', h, w)
        xn = x * self.avg_pool(x)
        xn = xn.sum(dim=1, keepdim=True)  # (b*32, 1, h, w)
        t = xn.view(b * self.groups, -1)
        t = t - t.mean(dim=1, keepdim=True)
        std = t.std(dim=1, keepdim=True) + 1e-5
        t = t / std
        t = t.view(b, self.groups, h, w)
        t = t * self.weight + self.bias
        t = t.view(b * self.groups, 1, h, w)
        x = x * self.sigmoid(t)
        x = x.view(b, c, h, w)
        return module_x * x



