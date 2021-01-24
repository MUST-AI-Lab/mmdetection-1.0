from thop import profile
import torch
import torch.nn as nn
from torchvision.models import resnet50
from pthflops import count_ops

class TRIPLET_shared(nn.Module):
    def __init__(self, no_spatial=False):
        super(TRIPLET_shared, self).__init__()
        self.cw = AttentionGate()
        self.hc = AttentionGate()
        self.no_spatial=no_spatial
        if not no_spatial:
            self.hw = AttentionGate()

    def forward(self, w, x, y, z):
        w_perm1 = w.permute(0, 2, 1, 3).contiguous()
        w_out1 = self.cw(w_perm1)
        w_out11 = w_out1.permute(0, 2, 1, 3).contiguous()
        w_perm2 = w.permute(0, 3, 2, 1).contiguous()
        w_out2 = self.hc(w_perm2)
        w_out21 = w_out2.permute(0, 3, 2, 1).contiguous()
        if not self.no_spatial:
            w_out = self.hw(w)
            w_out = 1 / 3 * (w_out + w_out11 + w_out21)
        else:
            w_out = 1 / 2 * (w_out11 + w_out21)
        
        ww = w_out + w



        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.cw(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
        if not self.no_spatial:
            x_out = self.hw(x)
            x_out = 1 / 3 * (x_out + x_out11 + x_out21)
        else:
            x_out = 1 / 2 * (x_out11 + x_out21)
        xx = x_out + x


        y_perm1 = y.permute(0, 2, 1, 3).contiguous()
        y_out1 = self.cw(y_perm1)
        y_out11 = y_out1.permute(0, 2, 1, 3).contiguous()
        y_perm2 = y.permute(0, 3, 2, 1).contiguous()
        y_out2 = self.hc(y_perm2)
        y_out21 = y_out2.permute(0, 3, 2, 1).contiguous()
        if not self.no_spatial:
            y_out = self.hw(y)
            y_out = 1 / 3 * (y_out + y_out11 + y_out21)
        else:
            y_out = 1 / 2 * (y_out11 + y_out21)
        yy = y_out + y



        z_perm1 = z.permute(0, 2, 1, 3).contiguous()
        z_out1 = self.cw(z_perm1)
        z_out11 = z_out1.permute(0, 2, 1, 3).contiguous()
        z_perm2 = z.permute(0, 3, 2, 1).contiguous()
        z_out2 = self.hc(z_perm2)
        z_out21 = z_out2.permute(0, 3, 2, 1).contiguous()
        if not self.no_spatial:
            z_out = self.hw(z)
            z_out = 1 / 3 * (z_out + z_out11 + z_out21)
        else:
            z_out = 1 / 2 * (z_out11 + z_out21)
        zz = z_out + z


        return (ww, xx, yy, zz)



class TRIPLET_unshared(nn.Module):
    def __init__(self, no_spatial=False):
        super(TRIPLET_unshared, self).__init__()

        self.layer1 = nn.Sequential()
        self.layer2 = nn.Sequential()
        self.layer3 = nn.Sequential()
        self.layer4 = nn.Sequential()

        self.layer1.cw = AttentionGate()
        self.layer2.cw = AttentionGate()
        self.layer3.cw = AttentionGate()
        self.layer4.cw = AttentionGate()

        self.layer1.hc = AttentionGate()
        self.layer2.hc = AttentionGate()
        self.layer3.hc = AttentionGate()
        self.layer4.hc = AttentionGate()

        
        self.no_spatial=no_spatial
        if not no_spatial:
            self.layer1.hw = AttentionGate()
            self.layer2.hw = AttentionGate()
            self.layer3.hw = AttentionGate()
            self.layer4.hw = AttentionGate()

    def forward(self, w, x, y, z):
        w_perm1 = w.permute(0, 2, 1, 3).contiguous()
        w_out1 = self.layer1.cw(w_perm1)
        w_out11 = w_out1.permute(0, 2, 1, 3).contiguous()
        w_perm2 = w.permute(0, 3, 2, 1).contiguous()
        w_out2 = self.layer1.hc(w_perm2)
        w_out21 = w_out2.permute(0, 3, 2, 1).contiguous()
        if not self.no_spatial:
            w_out = self.layer1.hw(w)
            w_out = 1 / 3 * (w_out + w_out11 + w_out21)
        else:
            w_out = 1 / 2 * (w_out11 + w_out21)
        
        ww = w_out + w



        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.layer2.cw(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.layer2.hc(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
        if not self.no_spatial:
            x_out = self.layer2.hw(x)
            x_out = 1 / 3 * (x_out + x_out11 + x_out21)
        else:
            x_out = 1 / 2 * (x_out11 + x_out21)
        xx = x_out + x


        y_perm1 = y.permute(0, 2, 1, 3).contiguous()
        y_out1 = self.layer3.cw(y_perm1)
        y_out11 = y_out1.permute(0, 2, 1, 3).contiguous()
        y_perm2 = y.permute(0, 3, 2, 1).contiguous()
        y_out2 = self.layer3.hc(y_perm2)
        y_out21 = y_out2.permute(0, 3, 2, 1).contiguous()
        if not self.no_spatial:
            y_out = self.layer3.hw(y)
            y_out = 1 / 3 * (y_out + y_out11 + y_out21)
        else:
            y_out = 1 / 2 * (y_out11 + y_out21)
        yy = y_out + y



        z_perm1 = z.permute(0, 2, 1, 3).contiguous()
        z_out1 = self.layer4.cw(z_perm1)
        z_out11 = z_out1.permute(0, 2, 1, 3).contiguous()
        z_perm2 = z.permute(0, 3, 2, 1).contiguous()
        z_out2 = self.layer4.hc(z_perm2)
        z_out21 = z_out2.permute(0, 3, 2, 1).contiguous()
        if not self.no_spatial:
            z_out = self.layer4.hw(z)
            z_out = 1 / 3 * (z_out + z_out11 + z_out21)
        else:
            z_out = 1 / 2 * (z_out11 + z_out21)
        zz = z_out + z


        return (ww, xx, yy, zz)

class AttentionGate(nn.Module):
    def __init__(self):
        super(AttentionGate, self).__init__()
        kernel_size = 7
        self.compress = ZPool()
        self.conv = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale


class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x



shared_input = (torch.randn(2,256,200,200),torch.randn(2,256,100,100),torch.randn(2,256,50,50),torch.randn(2,256,25,25))
triplet_shared = TRIPLET_shared()
count_ops(triplet_shared,shared_input)
print('Total params: %.2f' % (sum(p.numel() for p in triplet_shared.parameters())))

unshared_input = (torch.randn(2,256,200,200),torch.randn(2,256,100,100),torch.randn(2,256,50,50),torch.randn(2,256,25,25))
triplet_shared = TRIPLET_unshared()
count_ops(triplet_shared,unshared_input)
print('Total params: %.2f' % (sum(p.numel() for p in triplet_shared.parameters())))