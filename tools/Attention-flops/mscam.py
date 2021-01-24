from thop import profile
import torch
import torch.nn as nn
from torchvision.models import resnet50
from pthflops import count_ops

class MSCAM_shared(nn.Module):
    def __init__(self, inplanes=256, r=16):
        super(MSCAM_shared, self).__init__()
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

    def forward(self, w,x,y,z):
        xl_w = self.local_att(w)
        xg_w = self.global_att(w)
        xlg_w = xl_w + xg_w
        wei_w = self.sigmoid(xlg_w)
        ww = w * wei_w + w
        
        xl_x = self.local_att(x)
        xg_x = self.global_att(x)
        xlg_x = xl_x + xg_x
        wei_x = self.sigmoid(xlg_x)
        xx = x * wei_x + x

        xl_y = self.local_att(y)
        xg_y = self.global_att(y)
        xlg_y = xl_y + xg_y
        wei_y = self.sigmoid(xlg_y)
        yy = y * wei_y + y

        xl_z = self.local_att(z)
        xg_z = self.global_att(z)
        xlg_z = xl_z + xg_z
        wei_z = self.sigmoid(xlg_z)
        zz = z * wei_z + z

        return (ww, xx, yy, zz)


class MSCAM_unshared(nn.Module):
    def __init__(self, inplanes=256, r=16):
        super(MSCAM_unshared, self).__init__()
        self.inplanes = inplanes


        self.layer1 = nn.Sequential()
        self.layer1.local_att = nn.Sequential(
            nn.Conv2d(self.inplanes, self.inplanes // r, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.inplanes // r),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inplanes // r, self.inplanes, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.inplanes),
        )
        self.layer1.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.inplanes, self.inplanes // r, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.inplanes // r),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inplanes // r, self.inplanes, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.inplanes),
        )
        self.layer1.sigmoid = nn.Sigmoid()


        self.layer2 = nn.Sequential()
        self.layer2.local_att = nn.Sequential(
            nn.Conv2d(self.inplanes, self.inplanes // r, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.inplanes // r),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inplanes // r, self.inplanes, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.inplanes),
        )
        self.layer2.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.inplanes, self.inplanes // r, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.inplanes // r),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inplanes // r, self.inplanes, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.inplanes),
        )
        self.layer2.sigmoid = nn.Sigmoid()


        self.layer3 = nn.Sequential()
        self.layer3.local_att = nn.Sequential(
            nn.Conv2d(self.inplanes, self.inplanes // r, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.inplanes // r),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inplanes // r, self.inplanes, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.inplanes),
        )
        self.layer3.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.inplanes, self.inplanes // r, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.inplanes // r),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inplanes // r, self.inplanes, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.inplanes),
        )
        self.layer3.sigmoid = nn.Sigmoid()

        self.layer4 = nn.Sequential()
        self.layer4.local_att = nn.Sequential(
            nn.Conv2d(self.inplanes, self.inplanes // r, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.inplanes // r),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inplanes // r, self.inplanes, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.inplanes),
        )
        self.layer4.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.inplanes, self.inplanes // r, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.inplanes // r),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inplanes // r, self.inplanes, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.inplanes),
        )
        self.layer4.sigmoid = nn.Sigmoid()

    def forward(self, w,x,y,z):
        xl_w = self.layer1.local_att(w)
        xg_w = self.layer1.global_att(w)
        xlg_w = xl_w + xg_w
        wei_w = self.layer1.sigmoid(xlg_w)
        ww = w * wei_w + w
        
        xl_x = self.layer2.local_att(x)
        xg_x = self.layer2.global_att(x)
        xlg_x = xl_x + xg_x
        wei_x = self.layer2.sigmoid(xlg_x)
        xx = x * wei_x + x

        xl_y = self.layer3.local_att(y)
        xg_y = self.layer3.global_att(y)
        xlg_y = xl_y + xg_y
        wei_y = self.layer3.sigmoid(xlg_y)
        yy = y * wei_y + y

        xl_z = self.layer4.local_att(z)
        xg_z = self.layer4.global_att(z)
        xlg_z = xl_z + xg_z
        wei_z = self.layer4.sigmoid(xlg_z)
        zz = z * wei_z + z

        return (ww, xx, yy, zz)



shared_input = (torch.randn(2,256,200,200),torch.randn(2,256,100,100),torch.randn(2,256,50,50),torch.randn(2,256,25,25))
mscam_shared = MSCAM_shared()
count_ops(mscam_shared,shared_input)
print('Total params: %.2f' % (sum(p.numel() for p in mscam_shared.parameters())))

unshared_input = (torch.randn(2,256,200,200),torch.randn(2,256,100,100),torch.randn(2,256,50,50),torch.randn(2,256,25,25))
mscam_unshared = MSCAM_unshared()
count_ops(mscam_unshared,unshared_input)
print('Total params: %.2f' % (sum(p.numel() for p in mscam_unshared.parameters())))