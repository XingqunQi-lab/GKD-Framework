# -*- coding:utf-8 -*-
# Author : lkq
# Data : 2019/3/6 15:11
import torch
import torch.nn.functional as F
from torch import nn
import math

from sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

multi_gpu = True
BatchNorm = SynchronizedBatchNorm2d if multi_gpu else nn.BatchNorm2d

__all__ = ['UnetSKA2']

class SKConv(nn.Module):
    def __init__(self, channel, reduction=2):
        super(SKConv, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(channel, channel, 5, padding=2, bias=True),
            BatchNorm(channel),
            nn.ReLU(inplace=True))
    
        self.conv2 = nn.Sequential(
            nn.Conv2d(channel, channel, 3, padding=1, bias=True),
            BatchNorm(channel),
            nn.ReLU(inplace=True))

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv_se = nn.Sequential(
            nn.Conv2d(channel, channel//reduction, 1, padding=0, bias=True),
            BatchNorm(channel//reduction),
            nn.ReLU(inplace=True)
        )
        self.conv_ex1 = nn.Sequential(nn.Conv2d(channel//reduction, channel, 1, padding=0, bias=True))
        self.conv_ex2 = nn.Sequential(nn.Conv2d(channel//reduction, channel, 1, padding=0, bias=True))
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        conv1 = self.conv1(x).unsqueeze(dim=1)
        conv2 = self.conv2(x).unsqueeze(dim=1)
        features = torch.cat([conv1, conv2], dim=1)
        U = torch.sum(features, dim=1)
        S = self.pool(U)
        Z = self.conv_se(S)
        attention_vector = torch.cat([self.conv_ex1(Z).unsqueeze(dim=1), self.conv_ex2(Z).unsqueeze(dim=1)],dim=1)
        attention_vector = self.softmax(attention_vector)
        V = (features * attention_vector).sum(dim=1)
        
        return V


class SKBlock(nn.Module):
    def __init__(self,inplanes,planes,stride=1,downsample=None):
        super(SKBlock,self).__init__()
        self.conv1=nn.Sequential(nn.Conv2d(inplanes,planes,kernel_size=1,stride=stride,bias=False),
                                 BatchNorm(planes),
                                 nn.ReLU(inplace=True))
        self.conv2=nn.Sequential(SKConv(planes),
                                 BatchNorm(planes),
                                 nn.ReLU(inplace=True))
        self.conv3=nn.Sequential(nn.Conv2d(planes,planes,1,bias=False),
                                 BatchNorm(planes))
        self.relu=nn.ReLU(inplace=True)
        self.downsample=downsample
    def forward(self, input):
        shortcut=input
        output=self.conv1(input)
        output=self.conv2(output)
        output=self.conv3(output)
        if self.downsample is not None:
            shortcut=self.downsample(input)
        output+=shortcut
        return self.relu(output)

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super(DoubleConv, self).__init__()
        downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch,kernel_size=1, stride=stride, bias=False),
                BatchNorm(out_ch)
            )
        self.conv = nn.Sequential(
            SKBlock(in_ch,out_ch,downsample=downsample),
            SKBlock(out_ch,out_ch)
        )

    def forward(self, input):
        return self.conv(input)

class PAM_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        #self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        #self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        #self.gamma = 1

        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.query_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        #scaled = math.sqrt(C)
        #energy = energy/scaled
        attention = self.softmax(energy)
        proj_value = self.query_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out

class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out

class StripPooling(nn.Module):
    """
    Reference:
    """
    def __init__(self, in_channels, pool_size, norm_layer=BatchNorm):
        super(StripPooling, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(pool_size[0])
        self.pool2 = nn.AdaptiveAvgPool2d(pool_size[1])
        self.pool3 = nn.AdaptiveAvgPool2d((1, None))
        self.pool4 = nn.AdaptiveAvgPool2d((None, 1))

        inter_channels = int(in_channels)
        self.conv1_1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv1_2 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv2_0 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels))
        self.conv2_1 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels))
        self.conv2_2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels))
        self.conv2_3 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (1, 3), 1, (0, 1), bias=False),
                                norm_layer(inter_channels))
        self.conv2_4 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (3, 1), 1, (1, 0), bias=False),
                                norm_layer(inter_channels))
        self.conv2_5 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv2_6 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv2d(inter_channels*2, in_channels, 1, bias=False),
                                norm_layer(in_channels))
        self.gamma = nn.Parameter(torch.zeros(1))
        

    def forward(self, x):
        _, _, h, w = x.size()
        x1 = self.conv1_1(x)
        x2 = self.conv1_2(x)
        x2_1 = self.conv2_0(x1)
        x2_2 = F.interpolate(self.conv2_1(self.pool1(x1)), (h, w))
        x2_3 = F.interpolate(self.conv2_2(self.pool2(x1)), (h, w))
        x2_4 = F.interpolate(self.conv2_3(self.pool3(x2)), (h, w))
        x2_5 = F.interpolate(self.conv2_4(self.pool4(x2)), (h, w))
        x1 = self.conv2_5(F.relu_(x2_1 + x2_2 + x2_3))
        x2 = self.conv2_6(F.relu_(x2_5 + x2_4))
        out = self.conv3(torch.cat([x1, x2], dim=1))
        #out = self.conv3(x2)
        return F.relu_(x + out)

class StripPooling2(nn.Module):
    """
    Reference:
    """
    def __init__(self, in_channels, pool_size, norm_layer=BatchNorm):
        super(StripPooling2, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(pool_size[0])
        self.pool2 = nn.AdaptiveAvgPool2d(pool_size[1])
        self.pool3 = nn.AdaptiveAvgPool2d((1, None))
        self.pool4 = nn.AdaptiveAvgPool2d((None, 1))

        inter_channels = int(in_channels)
        self.conv1_1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv1_2 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv2_0 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels))
        self.conv2_1 = nn.Sequential(nn.Conv2d(inter_channels, int(inter_channels/2), 3, 1, 1, bias=False),
                                norm_layer(int(inter_channels/2)))
        self.conv2_2 = nn.Sequential(nn.Conv2d(inter_channels, int(inter_channels/2), 3, 1, 1, bias=False),
                                norm_layer(int(inter_channels/2)))
        self.conv2_3 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (1, 3), 1, (0, 1), bias=False),
                                norm_layer(inter_channels))
        self.conv2_4 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (3, 1), 1, (1, 0), bias=False),
                                norm_layer(inter_channels))
        self.conv2_5 = nn.Sequential(nn.Conv2d(inter_channels*2, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv2_6 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv2d(inter_channels*2, in_channels, 1, bias=False),
                                norm_layer(in_channels))
        self.gamma = nn.Parameter(torch.zeros(1))
        

    def forward(self, x):
        _, _, h, w = x.size()
        x1 = self.conv1_1(x)
        x2 = self.conv1_2(x)
        x2_1 = self.conv2_0(x1)
        x2_2 = F.interpolate(self.conv2_1(self.pool1(x1)), (h, w))
        x2_3 = F.interpolate(self.conv2_2(self.pool2(x1)), (h, w))
        x2_4 = F.interpolate(self.conv2_3(self.pool3(x2)), (h, w))
        x2_5 = F.interpolate(self.conv2_4(self.pool4(x2)), (h, w))
        x1 = self.conv2_5(F.relu_(torch.cat([x2_1,x2_2,x2_3],dim=1)))
        x2 = self.conv2_6(F.relu_(x2_5 + x2_4))
        x2 = x*x2
        out = self.conv3(torch.cat([x1, x2], dim=1))
        #out = self.conv3(x2)
        return F.relu_(x + out)

class UnetSKA2(nn.Module):
    def __init__(self, in_ch=1, out_ch=2):
        super(UnetSKA2, self).__init__()
        self.conv1 = DoubleConv(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)
        #self.pool1 = nn.Conv2d(64, 64, kernel_size=3, stride=2,padding=1, bias=False)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        #self.pool2 = nn.Conv2d(128, 128, kernel_size=3, stride=2,padding=1, bias=False)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        #self.pool3 = nn.Conv2d(256, 256, kernel_size=3, stride=2,padding=1, bias=False)
        self.conv4 = DoubleConv(256, 512)
        
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = DoubleConv(512, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = DoubleConv(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = DoubleConv(128, 64)
        self.conv10 = nn.Sequential(nn.Dropout2d(0.5, False), nn.Conv2d(64, out_ch, 1))
        #self.conv10 = nn.Conv2d(64, out_ch, 1)
        self.conv11 = nn.Sequential(nn.Dropout2d(0.5, False), nn.Conv2d(64, out_ch, 1))
        #self.conv11 = nn.Conv2d(64, out_ch, 1)
        
        self.pam1 = StripPooling(256, (3,6))
        self.pam2 = StripPooling(128, (12,18))
        self.pam3 = StripPooling2(64, (3,6))
        #self.pam4 = StripPooling2(64, (12,20))
        
        #self.pam1 = PAM_Module(256)
        #self.pam2 = PAM_Module(128)
        #self.pam3 = PAM_Module(64)
        self._init_weight()
        
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                torch.nn.init.xavier_uniform_(m.weight, gain=1)
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
        
        up_7 = self.up7(c4)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)
        c7 = self.pam1(c7)
        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        c8 = self.pam2(c8)
        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)
        c11 = self.conv11(c9)
        c9 = self.pam3(c9)
        #c9 = self.pam4(c9)
        c10 = self.conv10(c9)
        
        """
        torch.Size([8, 512, 32, 32])
        torch.Size([8, 256, 64, 64])
        torch.Size([8, 2, 256, 256])


        """
        return c3,up_7,c10


if __name__ == "__main__":
    batch_size = 8
    num_classes = 2
    h = 256
    w = 256
    x = torch.randn((batch_size, 1, h, w), requires_grad=True)
    UnetSKA2 = UnetSKA2(in_ch=1, out_ch=num_classes)
    middle_feature1,middle_feature2,y = UnetSKA2(x)
    print(middle_feature1.size())
    print(middle_feature2.size())
    print(y.size())