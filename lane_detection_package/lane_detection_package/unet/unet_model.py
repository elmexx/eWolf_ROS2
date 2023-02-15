""" Full assembly of the parts to form the complete network """

import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet_parts import Up, Down, DoubleConv, OutConv

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x
    
class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
       
        self.up2 = Up(512, 256 // factor, bilinear)
        
        self.up3 = Up(256, 128 // factor, bilinear)
        
        self.up4 = Up(128, 64, bilinear)
        
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class Att_UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(Att_UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.att1 = Attention_block(F_g=512,F_l=512,F_int=256)
        self.up_conv1 = conv_block(ch_in=1024, ch_out=512)
        
        self.up2 = Up(512, 256 // factor, bilinear)
        self.att2 = Attention_block(F_g=256,F_l=256,F_int=128)
        self.up_conv2 = conv_block(ch_in=512, ch_out=256)
        
        self.up3 = Up(256, 128 // factor, bilinear)
        self.att3 = Attention_block(F_g=128,F_l=128,F_int=64)
        self.up_conv3 = conv_block(ch_in=256, ch_out=128)
        
        self.up4 = Up(128, 64, bilinear)
        self.att4 = Attention_block(F_g=64,F_l=64,F_int=32)
        self.up_conv4 = conv_block(ch_in=128, ch_out=64)
        
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # x = self.up1(x5, x4)
        d5 = self.up1(x5, x4)
        x4 = self.att1(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)        
        d5 = self.up_conv1(d5)
        
        # x = self.up2(x, x3)
        d4 = self.up2(d5, x3)
        x3 = self.att2(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.up_conv2(d4)
        
        # x = self.up3(x, x2)
        d3 = self.up3(d4, x2)
        x2 = self.att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.up_conv3(d3)
        
        # x = self.up4(x, x1)
        d2 = self.up4(d3, x1)
        x1 = self.att4(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.up_conv4(d2)
        
        logits = self.outc(d2)
        return logits
    
