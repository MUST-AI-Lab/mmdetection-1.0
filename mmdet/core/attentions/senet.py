from __future__ import print_function, division, absolute_import
from mmcv.cnn import normal_init
import torch.nn as nn
from .builder import ATTENTIONS

@ATTENTIONS.register_module
class SENet(nn.Module):
    def __init__(self, inplanes, reduction=16, bias=True):
        super(SENet, self).__init__()
        self.inplanes = inplanes
        self.reduction = reduction
        self.bias = bias
        self.init_layer()
        self.fp16_enabled = False

    def init_layer(self):
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc1 = nn.Conv2d(self.inplanes, self.inplanes // self.reduction,
                      kernel_size=(1, 1), stride=(1, 1), bias=self.bias)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(self.inplanes // self.reduction, self.inplanes,
                      kernel_size=(1, 1), stride=(1, 1), bias=self.bias)
        self.sigmoid = nn.Sigmoid()

    def init_weights(self):
        normal_init(self.fc1, std=0.01)
        normal_init(self.fc2, std=0.01)

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x + module_input


