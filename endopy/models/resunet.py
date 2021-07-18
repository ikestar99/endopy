#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
Deep Residual Learning for Image Recognition
arXiv:1512.03385
@author: ike
"""
import torch
import torch.nn as nn
import torchvision.transforms.functional as ttf


class Encoder(nn.Module):
    """
    nn.Module, creates block for a ResUNet encoder layer.

    Parameters:
    ----------
    dowmtype (str) : reduce dimensions with "conv" for convolution or "max" for maxpool.
    mode (int) : block type. 0 for a basic block, 1 for a deeper block.
    Cin (int) : expected input channels for first convolutional operation.
    Cout (int) : expected ouput channels for entire block.
    stride (int) : stride of first convolution operation. The default is 1.

    Returns:
    -------
    None.
    """
    
    def __init__(self, down, mode, Cin, Cout, stride=1):
        super(Encoder, self).__init__()
        self.needShortcut = (True if (stride != 1 or Cin != Cout) else False)
        self.relu = nn.ReLU()
        Cmid = (Cin + Cout) // 2
        
        if down == "conv":
            if mode == 0:
                self.layers = nn.Sequential(
                    nn.Conv2d(Cin, Cout, kernel_size=3, stride=stride,
                              padding=1, bias=False),
                    nn.BatchNorm2d(Cout), nn.ReLU(),
                    nn.Conv2d(Cout, Cout, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(Cout))
                
            elif mode == 1:
                self.layers = nn.Sequential(
                    nn.Conv2d(Cin, Cmid, kernel_size=1, bias=False),
                    nn.BatchNorm2d(Cmid), nn.ReLU(),
                    nn.Conv2d(Cmid, Cmid, kernel_size=3, stride=stride,
                              padding=1, bias=False),
                    nn.BatchNorm2d(Cmid), nn.ReLU(),
                    nn.Conv2d(Cmid, Cout, kernel_size=1, bias=False),
                    nn.BatchNorm2d(Cout))
                
        elif down == "max":
            if mode == 0:
                self.layers = (nn.Sequential(
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(Cin, Cout, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(Cout))
                    if stride == 2 else nn.Sequential(
                            nn.Conv2d(Cin, Cout, kernel_size=3, padding=1, bias=False),
                            nn.BatchNorm2d(Cout)))
                
            elif mode == 1:
                self.layers = (nn.Sequential(
                    nn.Conv2d(Cin, Cmid, kernel_size=1, bias=False),
                    nn.BatchNorm2d(Cmid), nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(Cmid, Cout, kernel_size=1, bias=False),
                    nn.BatchNorm2d(Cout))
                    if stride == 2 else nn.Sequential(
                            nn.Conv2d(Cin, Cmid, kernel_size=1, bias=False),
                            nn.BatchNorm2d(Cmid), nn.ReLU(),
                            nn.Conv2d(Cmid, Cout, kernel_size=1, bias=False),
                            nn.BatchNorm2d(Cout)))
        
        if self.needShortcut:
            self.shortcut = nn.Sequential(
                nn.Conv2d(Cin, Cout, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(Cout))

    def forward(self, x):
        out = self.layers(x)
        out = out + (self.shortcut(x) if self.needShortcut else x)
        out = self.relu(out)
        return out


class Decoder(nn.Module):
    """
    nn.Module, creates block for a ResUNet decoder layer.

    Parameters:
    ----------
    Cin (int) : expected input channels for first convolutional operation.
    Cout (int) : expected ouput channels for entire block.
    padding (bool) : pad input spatial dimensions by 1. The default is True.

    Returns:
    -------
    None.
    """
    
    def __init__(self, Cin, Cout, padding=True):
        super(Decoder, self).__init__()
        Cmid = (Cin + Cout) // 2
        self.tconv = nn.Sequential(
            nn.ConvTranspose2d(Cin, Cout, kernel_size=2, stride=2),
            nn.BatchNorm2d(Cout))
        self.layers = nn.Sequential(
            nn.Conv2d(Cin + Cout, Cmid, kernel_size=3, padding=int(padding)),
            nn.BatchNorm2d(Cmid), nn.ReLU(),
            nn.Conv2d(Cmid, Cout, kernel_size=3, padding=int(padding)),
            nn.BatchNorm2d(Cout), nn.ReLU())

    def forward(self, x, bridge):
        up = self.tconv(x)
        crop = ttf.center_crop(bridge, up.shape[-2:])
        out = self.layers(torch.cat([up, crop], 1))
        return out


class ResUNet(nn.Module):
    """
    nn.Module, creates a ResUNet convolutional neural network.

    Parameters:
    ----------
    organelle (str) : mito or ER, organelle that model was/will be trained to segment.
    ver (int) : model version. The default is 18, others are 34, 50, 101, 152.
    dowmtype (str) : reduce dimensions with "conv" for convolution or "max" for maxpool.
    Cin (int) : expected input channels for first convolutional operation.
    nClasses (int) : expected number of class labels in ground truth.

    Returns:
    -------
    None.
    """
    
    def __init__(self, params):
        super(ResUNet, self).__init__()
        stats = {
            18: (0, 2, 2, 2, 2), 34: (0, 3, 4, 6, 3),
            50: (1, 3, 4, 6, 3), 101: (1, 3, 4, 23, 3),
            152: (1, 3, 8, 36, 3)}
        
        self.stats, down = stats[params("vers")], params("reduce")
        Cin, Cout = params("Cin"), params("Cout")
        scale = 4 if self.stats[0] == 1 else 1
        self.Cs = (Cin, 64, 64*scale, 128*scale, 256*scale, 512*scale)

        self.main = Encoder(down, self.stats[0], self.Cs[0], self.Cs[1])
        self.enc1 = self.makeEncoder(1, down=down, stride=1)
        self.enc2 = self.makeEncoder(2, down=down, stride=2)
        self.enc3 = self.makeEncoder(3, down=down, stride=2)
        self.enc4 = self.makeEncoder(4, down=down, stride=2)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.dec4 = Decoder(self.Cs[5], self.Cs[4])
        self.dec3 = Decoder(self.Cs[4], self.Cs[3])
        self.dec2 = Decoder(self.Cs[3], self.Cs[2])
        self.dec1 = Decoder(self.Cs[2], self.Cs[1])
        self.last = nn.Sequential(
            nn.Conv2d(
                self.Cs[1], ((self.Cs[1] + Cout) // 2), kernel_size=3, padding=1),
            nn.BatchNorm2d(((self.Cs[1] + Cout) // 2)),
            nn.ReLU(),
            nn.Conv2d(
                ((self.Cs[1] + Cout) // 2), Cout, kernel_size=1, stride=1),
            nn.ReLU())

    def makeEncoder(self, encoder, down, stride):
        layers = [Encoder(
            down, self.stats[0], self.Cs[encoder], self.Cs[encoder + 1], stride)]
        layers += [Encoder(
            down, self.stats[0], self.Cs[encoder + 1], self.Cs[encoder + 1])
            for x in range(self.stats[encoder] - 1)]
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.main(x)  # [N, Cs[1], H, W]
        lay1 = self.enc1(out)  # [N, Cs[2], H, W]
        lay2 = self.enc2(lay1)  # [N, Cs[3], H/2, W/2]
        lay3 = self.enc3(lay2)  # [N, Cs[4], H/4, W/4]
        lay4 = self.enc4(lay3)  # [N, Cs[5], H/8, W/8]
        
        out = self.pool(lay4)  # [N, Cs[5], H/16, W/16]
        
        out = self.dec4(out, lay4)  # [N, Cs[4], H/8, W/8]
        out = self.dec3(out, lay3)  # [N, Cs[3], H/4, W/4]
        out = self.dec2(out, lay2)  # [N, Cs[2], H/2, W/2]
        out = self.dec1(out, lay1)  # [N, Cs[1], H, W]
        out = self.last(out)  # [N, nClasses, H, W]
        return out
    

class ResUNetClassifier(nn.Module):
    def __init__(self, cfg):
        super(ResUNetClassifier, self).__init__()
        stats = {
            18: (0, 2, 2, 2, 2), 34: (0, 3, 4, 6, 3),
            50: (1, 3, 4, 6, 3), 101: (1, 3, 4, 23, 3),
            152: (1, 3, 8, 36, 3)}
        
        self.stats, down = stats[cfg["Version"]], cfg["Reduce"]
        Cs = (cfg["Cin"], 4, 4, 8, 16, 32)
        lin = (cfg["Hin"] * cfg["Win"] * Cs[-1]) // 256
        ls = (lin, 512, 50, cfg["Cout"])
        
        self.main = Encoder(down, self.stats[0], Cs[0], Cs[1])
        self.enc1 = self.makeEncoder(1, Cs[1], Cs[2], down=down, stride=1)
        self.enc2 = self.makeEncoder(2, Cs[2], Cs[3], down=down, stride=2)
        self.enc3 = self.makeEncoder(3, Cs[3], Cs[4], down=down, stride=2)
        self.enc4 = self.makeEncoder(4, Cs[4], Cs[5], down=down, stride=2)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.lin1 = self.makeLinear(ls[0], ls[1])
        self.lin2 = self.makeLinear(ls[1], ls[2])
        self.lin3 = self.makeLinear(ls[2], ls[3], False)

    def makeEncoder(self, num, Cin, Cout, down, stride):
        layers = [Encoder(down, self.stats[0], Cin, Cout, stride)]
        layers += [Encoder(down, self.stats[0], Cout, Cout)
                   for x in range(self.stats[num] - 1)]
        return nn.Sequential(*layers)

    def makeLinear(self, Cin, Cout, norm=True):
        layers = ([nn.Linear(Cin, Cout), nn.BatchNorm1d(Cout), nn.ReLU()]
                  if norm else [nn.Linear(Cin, Cout), nn.ReLU()])
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.main(x)  # [N, Cs[1], H, W]
        out = self.enc1(out)  # [N, Cs[2], H, W]
        out = self.enc2(out)  # [N, Cs[3], H/2, W/2]
        out = self.enc3(out)  # [N, Cs[4], H/4, W/4]
        out = self.enc4(out)  # [N, Cs[5], H/8, W/8]
        
        out = self.pool(out)  # [N, Cs[5], H/16, W/16]
        out = torch.flatten(out, start_dim=1)

        out = self.lin1(out)
        out = self.lin2(out)
        out = self.lin3(out)
        return out
