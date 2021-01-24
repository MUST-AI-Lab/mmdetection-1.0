from __future__ import print_function, division, absolute_import
from .builder import ATTENTIONS
from mmcv.cnn import normal_init
import torch
import torch.nn as nn
import math


@ATTENTIONS.register_module
class DNL(nn.Module):
    def __init__(self, inplanes, ratio, downsample=False, use_gn=False,
                 lr_mult=None, use_out=False, out_bn=False, whiten_type=['channle'],
                 temp=1.0, with_gc=False, with_2fc=False, double_conv=False):
        super(DNL, self).__init__()
        self.inplanes = inplanes
        self.outplanes = int(inplanes * ratio)

        self.conv_query = nn.Conv2d(self.inplanes, self.outplanes, kernel_size=1)
        self.conv_key = nn.Conv2d(self.inplanes, self.outplanes, kernel_size=1)

        if use_out:  # use_out=False
            self.conv_value = nn.Conv2d(self.inplanes, self.outplanes, kernel_size=1)
            self.conv_out = nn.Conv2d(self.outplanes, self.inplanes, kernel_size=1, bias=False)
        else:
            self.conv_value = nn.Conv2d(self.inplanes, self.inplanes, kernel_size=1, bias=False)
            self.conv_out = None

        if out_bn:  # out_bn = False
            self.out_bn = nn.BatchNorm2d(self.inplanes)
        else:
            self.out_bn = None

        self.with_2fc = with_2fc # with_2fc=False

        if with_gc:  # with_gc=True
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            if self.with_2fc:
                self.channel_mul_conv = nn.Sequential(
                    nn.Conv2d(self.inplanes, self.outplanes, kernel_size=1),
                    nn.LayerNorm([self.outplanes, 1, 1]),
                    nn.ReLU(inplace=True),  # yapf: disable
                    nn.Conv2d(self.outplanes, self.inplanes, kernel_size=1))
        if 'bn_affine' in whiten_type:  # whiten_type=['channel']
            self.key_bn_affine = nn.BatchNorm1d(self.outplanes)
            self.query_bn_affine = nn.BatchNorm1d(self.outplanes)
        if 'bn' in whiten_type:  # whiten_type=['channel']
            self.key_bn = nn.BatchNorm1d(self.outplanes, affine=False)
            self.query_bn = nn.BatchNorm1d(self.outplanes, affine=False)
        self.softmax = nn.Softmax(dim=2)
        self.downsample = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.gamma = nn.Parameter(torch.zeros(1))
        self.scale = math.sqrt(self.outplanes)
        self.whiten_type = whiten_type
        self.temp = temp
        self.with_gc = with_gc


    def forward(self, x):
        # [N, C, T, H, W]
        residual = x
        # [N, C, T, H', W']
        if self.downsample is not None:  # self.downsample=False
            input_x = self.downsample(x)
        else:
            input_x = x

        # [N, C', T, H, W]
        query = self.conv_query(x)
        # [N, C', T, H', W']
        key = self.conv_key(input_x)
        value = self.conv_value(input_x)

        # [N, C', H x W]
        query = query.view(query.size(0), query.size(1), -1)
        # [N, C', H' x W']
        key = key.view(key.size(0), key.size(1), -1)
        value = value.view(value.size(0), value.size(1), -1)

        if 'channel' in self.whiten_type:  # self.whiten_type = ['channel']
            key_mean = key.mean(2).unsqueeze(2)
            query_mean = query.mean(2).unsqueeze(2)
            key -= key_mean
            query -= query_mean
        if 'spatial' in self.whiten_type:
            key_mean = key.mean(1).unsqueeze(1)
            query_mean = query.mean(1).unsqueeze(1)
            key -= key_mean
            query -= query_mean
        if 'bn_affine' in self.whiten_type:
            key = self.key_bn_affine(key)
            query = self.query_bn_affine(query)
        if 'bn' in self.whiten_type:
            key = self.key_bn(key)
            query = self.query_bn(query)

        # [N, T x H x W, T x H' x W']
        sim_map = torch.bmm(query.transpose(1, 2), key)
        sim_map = sim_map / self.scale
        sim_map = sim_map / self.temp
        sim_map = self.softmax(sim_map)

        # [N, T x H x W, C']
        out_sim = torch.bmm(sim_map, value.transpose(1, 2))
        # [N, C', T x H x W]
        out_sim = out_sim.transpose(1, 2)
        # [N, C', T,  H, W]
        out_sim = out_sim.view(out_sim.size(0), out_sim.size(1), *x.size()[2:])
        # if self.norm is not None:
        #     out = self.norm(out)
        out_sim = self.gamma * out_sim

        # out = residual + out_sim

        if self.with_gc:
            if self.with_2fc:
                # [N, 1, H', W']
                mask = self.conv_mask(input_x)
                # [N, 1, H'x W']
                mask = mask.view(mask.size(0), mask.size(1), -1)
                mask = self.softmax(mask)
                # [N, 1, H'x W', 1]
                mask = mask.unsqueeze(-1)
                # [N, C, H'x W']
                input_x = input_x.view(input_x.size(0), input_x.size(1), -1)
                # [N, 1, C, H'x W']
                input_x = input_x.unsqueeze(1)
                # [N, 1, C, 1]
                out_gc = torch.matmul(input_x, mask)
                # [N, C, 1, 1]
                out_gc = out_gc.view(out_gc.size(0), out_gc.size(2), 1, 1)
                out_gc = self.channel_mul_conv(out_gc)
                out_sim = out_sim + out_gc
            else:
                # [N, 1, H', W']
                mask = self.conv_mask(input_x)
                # [N, 1, H'x W']
                mask = mask.view(mask.size(0), mask.size(1), -1)
                mask = self.softmax(mask)
                # [N, C, 1, 1]
                out_gc = torch.bmm(value, mask.permute(0, 2, 1)).unsqueeze(-1)
                out_sim = out_sim + out_gc

        out = out_sim + residual

        return out
