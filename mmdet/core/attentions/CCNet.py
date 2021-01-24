from __future__ import print_function, division, absolute_import
from mmcv.cnn import normal_init
import torch.nn as nn
import torch
from .builder import ATTENTIONS
import torch.autograd as autograd
from torch.autograd.function import once_differentiable

@ATTENTIONS.register_module
class CCA(nn.Module):
    """ Criss-Cross Attention Module"""

    def __init__(self, inplanes):
        super(CCA, self).__init__()
        self.inplanes = inplanes

        self.query_conv = nn.Conv2d(self.inplanes, self.inplanes // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(self.inplanes, self.inplanes // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(self.inplanes, self.inplanes, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        proj_query = self.query_conv(x)
        proj_key = self.key_conv(x)
        proj_value = self.value_conv(x)

        energy = ca_weight(proj_query, proj_key)
        attention = nn.Softmax(energy, 1)
        out = ca_map(attention, proj_value)
        out = self.gamma * out + x

        return out



class CA_Weight(autograd.Function):
    @staticmethod
    def forward(ctx, t, f):
        # Save context
        n, c, h, w = t.size()
        size = (n, h + w - 1, h, w)
        weight = torch.zeros(size, dtype=t.dtype, layout=t.layout, device=t.device)
        # Output
        ctx.save_for_backward(t, f)

        return weight

    @staticmethod
    @once_differentiable
    def backward(ctx, dw):
        t, f = ctx.saved_tensors

        dt = torch.zeros_like(t)
        df = torch.zeros_like(f)

        return dt, df


class CA_Map(autograd.Function):
    @staticmethod
    def forward(ctx, weight, g):
        # Save context
        out = torch.zeros_like(g)

        # Output
        ctx.save_for_backward(weight, g)

        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, dout):
        weight, g = ctx.saved_tensors

        dw = torch.zeros_like(weight)
        dg = torch.zeros_like(g)

        return dw, dg


ca_weight = CA_Weight.apply
ca_map = CA_Map.apply


