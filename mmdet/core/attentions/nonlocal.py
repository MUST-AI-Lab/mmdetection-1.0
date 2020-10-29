from __future__ import print_function, division, absolute_import
from mmcv.cnn import normal_init
import torch
import torch.nn as nn
from .builder import ATTENTIONS

@ATTENTIONS.register_module
class NonLocal(nn.Module):
    def __init__(self, inplanes, nonlocalType='Gaussian', inter_channels=None, sub_sample=False, bn_layer=False):
        super(NonLocal, self).__init__()
        self.inplanes = inplanes
        self.type = nonlocalType
        self.inter_channels = inter_channels
        self.sub_sample = sub_sample
        self.bn_layer = bn_layer

        self.init_layer()
        self.fp16_enabled = False

    def init_layer(self):
        if self.inter_channels is None:
            inter_channels = self.in_channels // 2
            if inter_channels == 0:
                inter_channels = 1

        self.g = nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0)

        if self.bn_layer:
            self.W = nn.Sequential(
                nn.Conv2d(self.inter_channels, self.in_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = nn.Conv2d(self.inter_channels, self.in_channels, kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)



        if self.nonlocalType == 'Gaussian':
            if self.sub_sample:
                self.g = nn.Sequential(self.g, nn.MaxPool2d(kernel_size=(2, 2)))
                self.phi = nn.MaxPool2d(kernel_size=(2, 2))

        if self.nonlocalType == 'EmbeddedGaussian':
            self.theta = nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0)
            self.phi = nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0)
            if self.sub_sample:
                self.g = nn.Sequential(self.g, nn.MaxPool2d(kernel_size=(2, 2)))
                self.phi = nn.Sequential(self.phi, nn.MaxPool2d(kernel_size=(2, 2)))

        if self.nonlocalType == 'Concatenation':
            self.theta = nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0)
            self.phi = nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0)
            self.concat_project = nn.Sequential(
                nn.Conv2d(self.inter_channels * 2, 1, 1, 1, 0, bias=False),
                nn.ReLU()
            )

            if self.sub_sample:
                self.g = nn.Sequential(self.g, nn.MaxPool2d(kernel_size=(2, 2)))
                self.phi = nn.Sequential(self.phi, nn.MaxPool2d(kernel_size=(2, 2)))

        if self.nonlocalType == 'DotProduct':
            self.theta = nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0)
            self.phi = nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0)
            if self.sub_sample:
                self.g = nn.Sequential(self.g, nn.MaxPool2d(kernel_size=(2, 2)))
                self.phi = nn.Sequential(self.phi, nn.MaxPool2d(kernel_size=(2, 2)))

    def init_weights(self):
        pass

    def forward(self, x):
        if self.nonlocalType == 'Gaussian':
            batch_size = x.shape[0]

            g_x = self.g(x).view(batch_size, self.inter_channels, -1)
            g_x = g_x.permute(0, 2, 1)

            theta_x = x.view(batch_size, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)

            if self.sub_sample:
                phi_x = self.phi(x).view(batch_size, self.in_channels, -1)
            else:
                phi_x = x.view(batch_size, self.in_channels, -1)

            f = torch.matmul(theta_x, phi_x)
            f_div_C = nn.Softmax(dim=-1)

            y = torch.matmul(f_div_C, g_x)
            y = y.permute(0, 2, 1).contiguous()
            y = y.view(batch_size, self.inter_channels, *x.size()[2:])
            W_y = self.W(y)
            z = W_y + x

            return z

        if self.nonlocalType == 'EmbeddedGaussian':
            batch_size = x.shape[0]

            g_x = self.g(x).view(batch_size, self.inter_channels, -1)
            g_x = g_x.permute(0, 2, 1)

            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
            f = torch.matmul(theta_x, phi_x)
            f_div_C = nn.Softmax(dim=-1)

            y = torch.matmul(f_div_C, g_x)
            y = y.permute(0, 2, 1).contiguous()
            y = y.view(batch_size, self.inter_channels, *x.size()[2:])
            W_y = self.W(y)
            z = W_y + x

            return z
        if self.nonlocalType == 'Concatenation':
            batch_size = x.shape[0]

            g_x = self.g(x).view(batch_size, self.inter_channels, -1)
            g_x = g_x.permute(0, 2, 1)

            # (b, c, N, 1)
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1, 1)
            # (b, c, 1, N)
            phi_x = self.phi(x).view(batch_size, self.inter_channels, 1, -1)

            h = theta_x.size(2)
            w = phi_x.size(3)
            theta_x = theta_x.repeat(1, 1, 1, w)
            phi_x = phi_x.repeat(1, 1, h, 1)

            concat_feature = torch.cat([theta_x, phi_x], dim=1)
            f = self.concat_project(concat_feature)
            b, _, h, w = f.size()
            f = f.view(b, h, w)

            N = f.size(-1)
            f_div_C = f / N

            y = torch.matmul(f_div_C, g_x)
            y = y.permute(0, 2, 1).contiguous()
            y = y.view(batch_size, self.inter_channels, *x.size()[2:])
            W_y = self.W(y)
            z = W_y + x

            return z
        if self.nonlocalType == 'DotProduct':
            batch_size = x.shape[0]

            g_x = self.g(x).view(batch_size, self.inter_channels, -1)
            g_x = g_x.permute(0, 2, 1)

            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
            f = torch.matmul(theta_x, phi_x)
            N = f.size(-1)
            f_div_C = f / N

            y = torch.matmul(f_div_C, g_x)
            y = y.permute(0, 2, 1).contiguous()
            y = y.view(batch_size, self.inter_channels, *x.size()[2:])
            W_y = self.W(y)
            z = W_y + x

            return z


