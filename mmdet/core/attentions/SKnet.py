from __future__ import print_function, division, absolute_import
from mmcv.cnn import normal_init
import torch
import torch.nn as nn
from .builder import ATTENTIONS
from functools import reduce

@ATTENTIONS.register_module
class SKNet(nn.Module):
    def __init__(self, inplanes, M=2, G=32, r=16, bias=True):
        super(SKNet, self).__init__()
        self.inplanes = inplanes
        self.outplanes = inplanes
        self.M = M
        self.G = G
        self.r = r
        self.bias = bias
        self.init_layers()
        self.fp16_enabled = False

    def init_layers(self, L=32):
        self.d = max(self.inplanes // self.r, L)
        self.split()
        self.fuse(self.d)
        self.select()

    def init_weights(self):
        for i in self.splitting:
            normal_init(i[0], std=0.01)
        channelWise.init_weight(self)

    def split(self):
        self.splitting = nn.ModuleList()
        for i in range(self.M):
            self.splitting.append(
                nn.Sequential(
                    nn.Conv2d(self.inplanes, self.outplanes, kernel_size=3, stride=1,
                              padding=1+i, dilation=1+i, groups=self.G, bias=True),
                    nn.BatchNorm2d(self.outplanes),
                    nn.ReLU(inplace=True)
                )
            )

    def fuse(self, d):
        self.fusing = channelWise(self.inplanes, self.M, d)

    def select(self):
        self.selecting_softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.shape[0]

        #split
        split_conv = []
        for i, split in enumerate(self.splitting):
            split_conv.append(split(x))
        U = reduce(lambda x, y: x+y, split_conv)

        #fuse
        z = self.fusing(U)

        #select
        a_b = z.reshape(batch_size,self.M, self.outplanes, -1)
        a_b = self.selecting_softmax(a_b)
        a_b = list(a_b.chunk(self.M, dim=1))
        a_b = list(map(lambda x:x.reshape(batch_size, self.outplanes, 1,1),a_b))
        V=list(map(lambda x,y:x*y, split_conv, a_b))
        V=reduce(lambda x,y:x+y,V)

        return V


# channel of SKNet
class channelWise(nn.Module):
    def __init__(self, inplanes, M, d, bias=True):
        super(channelWise, self).__init__()

        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Sequential(
            nn.Conv2d(inplanes, d, kernel_size=(1, 1), stride=(1, 1), bias=bias),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Conv2d(d, inplanes*M, kernel_size=(1, 1), stride=(1, 1), bias=bias)

    def init_weight(self):
        normal_init(self.fusing.fc1[0], std=0.01)
        normal_init(self.fusing.fc2, std=0.01)

    def forward(self, x):
        return self.fc2(self.fc1(self.GAP(x)))








