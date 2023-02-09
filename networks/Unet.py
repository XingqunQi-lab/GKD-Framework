"""
zwx
2021 4/15
"""


import torch
import torch.nn.functional as F
from torch import nn
from sync_batchnorm.batchnorm import SynchronizedBatchNorm2d


multi_gpu = True
BatchNorm = SynchronizedBatchNorm2d if multi_gpu else nn.BatchNorm2d

__all__ = ["Unet"]


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
            BatchNorm(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
            BatchNorm(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpBlock, self).__init__()
        self.upsample = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):  #x2是encoder过程中的feature map
        x1 = self.upsample(x1)
        if ((x1.size(2) != x2.size(2)) or (x1.size(3) != x2.size(3))):
            diffY = x2.size()[2]-x1.size()[2]
            diffX = x2.size()[3]-x1.size()[3]
            if(diffX%2!=0):
              if(diffY%2!=0):
                x1 = nn.functional.pad(x1, (diffX//2+1, diffX//2, diffY//2+1, diffY//2))
              else:
                x1 = nn.functional.pad(x1, (diffX//2+1, diffX//2, diffY//2, diffY//2))
            else:
              if(diffY%2!=0):
                  x1 = nn.functional.pad(x1, (diffX//2, diffX//2, diffY//2+1, diffY//2))
              else:
                x1 = nn.functional.pad(x1, (diffX//2, diffX//2, diffY//2, diffY//2))
            #x1 = nn.functional.pad(x1, (diffX//2, diffX//2, diffY//2, diffY//2))
        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)


class Unet(nn.Module):
    def __init__(self, in_ch=1, out_ch=2):
        super(Unet, self).__init__()
        self.conv1 = DoubleConv(in_ch, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        self.conv5 = DoubleConv(512, 1024)
        
        self.up1 = UpBlock(1024, 512)
        self.up2 = UpBlock(512, 256)
        self.up3 = UpBlock(256, 128)
        self.up4 = UpBlock(128, 64)
        self.conv14 = nn.Sequential(nn.Dropout2d(0.5, False),
                                   nn.Conv2d(64, out_ch, 1))
        self._init_weight()
          
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        c1 = self.conv1(x)

        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)

        c6 = self.up1(c5, c4)

        c7 = self.up2(c6, c3)
        c8 = self.up3(c7, c2)
        c9 = self.up4(c8, c1)
        y = self.conv14(c9)

        return c4,c7,y


if __name__ == "__main__":
    batch_size = 8
    num_classes = 2
    h = 64
    w = 64
    x = torch.randn((batch_size, 3, h, w), requires_grad=True)
    Unet = Unet(in_ch=3, out_ch=num_classes)
    middle_feature1,middle_feature2,y = Unet(x)
    print(middle_feature1.size())
    print(middle_feature2.size())
    
    print(y.size())