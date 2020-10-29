from __future__ import print_function, division, absolute_import
from mmcv.cnn import normal_init
import torch.nn as nn
from .builder import ATTENTIONS
from functools import reduce

@ATTENTIONS.register_module
class MY(nn.Module):
    def __init__(self, inplanes, M=2, G=32, r=16, dialation_num=2, dialation_val=4, bias=True):
        super(MY, self).__init__()
        self.inplanes = inplanes
        self.outplanes = inplanes
        self.M = M
        self.G = G
        self.r = r
        self.dialation_num = dialation_num
        self.dialation_val = dialation_val
        self.bias = bias
        self.init_layers()
        self.fp16_enabled = False

    def init_layers(self, L=32):
        self.d = max(self.inplanes // self.r, L)
        self.split()
        self.fuse(self.d, self.r)
        self.select(self.M)

    def init_weights(self):
        for i in self.splitting:
            normal_init(i[0], std=0.01)
        channelWise.init_weight(self)
        spatialWise.init_weight(self)


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

    def fuse(self, d, r):
        self.fusing = nn.ModuleList()
        self.fusing.channel = channelWise(self.inplanes, d, self.M)
        self.fusing.spatial = spatialWise(self.inplanes, r, self.M, dialation_num=2, dialation_val=4)
        self.fusing.sigmoid = nn.Sigmoid()

    def select(self, M):
        self.selecting = nn.ModuleList()
        self.selecting.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.shape[0]
        # outputshape = x.reshape()

        #split
        split_conv = []
        for i, split in enumerate(self.splitting):
            split_conv.append(split(x))
        U = reduce(lambda x, y: x+y, split_conv)

        #fuse
        channel = self.fusing.channel(U)
        spatial = self.fusing.spatial(U)
        z = self.fusing.sigmoid(channel + spatial)
        # z = z * U + U

        #select
        a_b = z.reshape(batch_size, self.M, self.outplanes, -1)
        a_b = self.selecting.softmax(a_b)
        a_b = list(a_b.chunk(self.M, dim=1))
        a_b = list(map(lambda x: x.reshape(batch_size, self.outplanes, 1, 1), a_b))
        V = list(map(lambda x, y: x * y, split_conv, a_b))
        V = reduce(lambda x, y: x + y, V)

        return V


# channel of SKNet
class channelWise(nn.Module):
    def __init__(self, inplanes, d, M, bias=True):
        super(channelWise, self).__init__()

        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Sequential(
            nn.Conv2d(inplanes, d, kernel_size=(1, 1), stride=(1, 1), bias=bias),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Conv2d(d, inplanes*M, kernel_size=(1, 1), stride=(1, 1), bias=bias),
            nn.BatchNorm2d(inplanes*M),
            nn.ReLU(inplace=True)
        )

#channel of BAM
# class channelWise(nn.Module):
#     def __init__(self, inplanes, r, bias=True):
#         super(channelWise, self).__init__()
#         self.GAP = nn.AdaptiveAvgPool2d(1)
#         self.fc1 = nn.Sequential(
#             nn.Conv2d(inplanes, inplanes // r, kernel_size=1, stride=1, bias=True),
#             nn.BatchNorm2d(inplanes // r),
#             nn.ReLU(inplace=True)
#             )
#         self.fc2 = nn.Sequential(
#             nn.Conv2d(inplanes // r, inplanes, kernel_size=1, stride=1, bias=True),
#             nn.BatchNorm2d(inplanes),
#             nn.ReLU(inplace=True)
#             )

    def init_weight(self):
        normal_init(self.fusing.channel.fc1[0], std=0.01)
        normal_init(self.fusing.channel.fc2[0], std=0.01)

    def forward(self, x):
        return self.fc2(self.fc1(self.GAP(x)))


class spatialWise(nn.Module):
    def __init__(self, inplanes, r, M, dialation_num, dialation_val, bias=True):
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
        self.fc3 = nn.Sequential(
            nn.Conv2d(inplanes // r, inplanes*M, kernel_size=1, stride=1),
            nn.BatchNorm2d(inplanes*M),
            nn.ReLU(inplace=True)
            )
        self.GAP = nn.AdaptiveAvgPool2d(1)

    def init_weight(self):
        normal_init(self.fusing.spatial.fc1[0],std=0.01)
        for i in range(self.dialation_num):
            normal_init(self.fusing.spatial.fc2[0], std=0.01)
        normal_init(self.fusing.spatial.fc3[0], std=0.01)

    def forward(self, x):
        return self.fc3(self.fc2(self.fc1(x)))





