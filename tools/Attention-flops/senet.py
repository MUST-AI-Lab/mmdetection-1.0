from thop import profile
import torch
import torch.nn as nn
# from torchvision.models import resnet50
from pthflops import count_ops

class SENet_shared(nn.Module):
    def __init__(self, inplanes=256, reduction=16, bias=False):
        super(SENet_shared, self).__init__()
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

    def forward(self, w, x, y, z):
        ww = w
        w = self.avg_pool(w)
        w = self.fc1(w)
        w = self.relu(w)
        w = self.fc2(w)
        w = self.sigmoid(w)
        w = ww * w + w

        xx = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        x = xx * x + x

        yy = y
        x = self.avg_pool(x)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        y = yy * y + y

        zz = z
        z = self.avg_pool(z)
        z = self.fc1(z)
        z = self.relu(z)
        z = self.fc2(z)
        z = self.sigmoid(z)
        z = xx * z + z

        return (w, x, y, z)

class SENet_unshared(nn.Module):
    def __init__(self, inplanes=256, reduction=16, bias=False):
        super(SENet_unshared, self).__init__()
        self.inplanes = inplanes
        self.reduction = reduction
        self.bias = bias
        self.init_layer()
        self.fp16_enabled = False

    def init_layer(self):
        self.layer1 = nn.Sequential()
        self.layer1.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.layer1.fc1 = nn.Conv2d(self.inplanes, self.inplanes // self.reduction,
                             kernel_size=(1, 1), stride=(1, 1), bias=self.bias)
        self.layer1.relu = nn.ReLU(inplace=True)
        self.layer1.fc2 = nn.Conv2d(self.inplanes // self.reduction, self.inplanes,
                             kernel_size=(1, 1), stride=(1, 1), bias=self.bias)
        self.layer1.sigmoid = nn.Sigmoid()

        self.layer2 = nn.Sequential()
        self.layer2.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.layer2.fc1 = nn.Conv2d(self.inplanes, self.inplanes // self.reduction,
                             kernel_size=(1, 1), stride=(1, 1), bias=self.bias)
        self.layer2.relu = nn.ReLU(inplace=True)
        self.layer2.fc2 = nn.Conv2d(self.inplanes // self.reduction, self.inplanes,
                             kernel_size=(1, 1), stride=(1, 1), bias=self.bias)
        self.layer2.sigmoid = nn.Sigmoid()

        self.layer3 = nn.Sequential()
        self.layer3.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.layer3.fc1 = nn.Conv2d(self.inplanes, self.inplanes // self.reduction,
                             kernel_size=(1, 1), stride=(1, 1), bias=self.bias)
        self.layer3.relu = nn.ReLU(inplace=True)
        self.layer3.fc2 = nn.Conv2d(self.inplanes // self.reduction, self.inplanes,
                             kernel_size=(1, 1), stride=(1, 1), bias=self.bias)
        self.layer3.sigmoid = nn.Sigmoid()

        self.layer4 = nn.Sequential()
        self.layer4.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.layer4.fc1 = nn.Conv2d(self.inplanes, self.inplanes // self.reduction,
                             kernel_size=(1, 1), stride=(1, 1), bias=self.bias)
        self.layer4.relu = nn.ReLU(inplace=True)
        self.layer4.fc2 = nn.Conv2d(self.inplanes // self.reduction, self.inplanes,
                             kernel_size=(1, 1), stride=(1, 1), bias=self.bias)
        self.layer4.sigmoid = nn.Sigmoid()

    def forward(self, w, x, y, z):
        ww = w
        w = self.layer1.avg_pool(w)
        w = self.layer1.fc1(w)
        w = self.layer1.relu(w)
        w = self.layer1.fc2(w)
        w = self.layer1.sigmoid(w)
        w = ww * w + w

        xx = x
        x = self.layer2.avg_pool(x)
        x = self.layer2.fc1(x)
        x = self.layer2.relu(x)
        x = self.layer2.fc2(x)
        x = self.layer2.sigmoid(x)
        x = xx * x + x

        yy= y
        x = self.layer3.avg_pool(x)
        y = self.layer3.fc1(y)
        y = self.layer3.relu(y)
        y = self.layer3.fc2(y)
        y = self.layer3.sigmoid(y)
        y = yy * y + y

        zz = z
        z = self.layer4.avg_pool(z)
        z = self.layer4.fc1(z)
        z = self.layer4.relu(z)
        z = self.layer4.fc2(z)
        z = self.layer4.sigmoid(z)
        z = xx * z + z

        return (w,x,y,z)



iinput = torch.randn(1,3,200,200)



# thop

# model = resnet50()
# print(model, '\n')
# flop, params = profile(model, inputs=(iinput,))
# print(flop, params)


# iinput = torch.randn(2,256,200,200)
# senet = SENet()
# print(senet, '\n')
# # flop, params = profile(senet, inputs=(iinput,), custom_ops={SENet: forward(senet, iinput)})
# flop, params = profile(senet, inputs=(iinput))
# print(flop, params)


shared_input = (torch.randn(2,256,200,200),torch.randn(2,256,100,100),torch.randn(2,256,50,50),torch.randn(2,256,25,25))
senet_shared = SENet_shared()
count_ops(senet_shared,shared_input)
print('Total params: %.2f' % (sum(p.numel() for p in senet_shared.parameters())))

unshared_input = (torch.randn(2,256,200,200),torch.randn(2,256,100,100),torch.randn(2,256,50,50),torch.randn(2,256,25,25))
senet_unshared = SENet_unshared()
count_ops(senet_unshared,unshared_input)
print('Total params: %.2f' % (sum(p.numel() for p in senet_unshared.parameters())))



