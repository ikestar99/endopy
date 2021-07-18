#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
@author: ike
Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
Deep Residual Learning for Image Recognition
arXiv:1512.03385
"""
import numpy as np

import torch
import torch.nn as nn


class EncoderLayer(nn.Module):
    """
    nn.Module, creates entire EndoNet encoder layer.

    Parameters:
    ----------
    Din (int) : input spatial dimensions. 2 for time series, 3 for volumetric.
    dowmtype (str) : reduce dimensions with "conv" for convolution or "max" for maxpool.
    depth(int) : number of blocks in each encoder layer.
    Cin (int) : expected input channels for first convolutional operation.
    Cout (int) : expected ouput channels for entire layer.
    stride (int) : stride of first convolution operation. The default is 1.

    Returns:
    -------
    None.
    """
    
    def __init__(self, Din, down, mode, depth, Cin, Cout, stride=1):
        super(EncoderLayer, self).__init__()
        self.info = (Din, down, mode, depth)
        
        self.blocks = nn.ModuleList()
        self.relu = nn.ReLU()
        self.getBlock(Cin, Cout, stride)
        for x in range(1, self.info[3]):
            self.getBlock(Cout, Cout)
    
    def getBlock(self, Cin, Cout, stride=1):
        conv = nn.Conv2d if self.info[0] == 2 else nn.Conv3d
        bnorm = nn.BatchNorm2d if self.info[0] == 2 else nn.BatchNorm3d
        mpool = nn.MaxPool2d if self.info[0] == 2 else nn.MaxPool3d
        Cmid = (Cin + Cout) // 2
        
        if self.info[1] == "conv":
            if self.info[2] == 0:
                self.blocks.append(nn.Sequential(
                    conv(Cin, Cmid, kernel_size=3, stride=stride,
                         padding=1, bias=False),
                    bnorm(Cmid), nn.ReLU(),
                    conv(Cmid, Cout, kernel_size=3, padding=1, bias=False),
                    bnorm(Cout)))
                
            elif self.info[2] == 1:
                self.blocks.append(nn.Sequential(
                    conv(Cin, Cmid, kernel_size=1, bias=False),
                    bnorm(Cmid), nn.ReLU(),
                    conv(Cmid, Cmid, kernel_size=3, stride=stride,
                         padding=1, bias=False),
                    bnorm(Cmid), nn.ReLU(),
                    conv(Cmid, Cout, kernel_size=1, bias=False),
                    conv(Cout)))
                
        elif self.info[1] == "max":
            if self.info[2] == 0:
                self.blocks.append((nn.Sequential(
                    mpool(kernel_size=2, stride=2),
                    conv(Cin, Cout, kernel_size=3, padding=1, bias=False),
                    bnorm(Cout))
                    if stride == 2 else nn.Sequential(
                            conv(Cin, Cout, kernel_size=3, padding=1, bias=False),
                            bnorm(Cout))))
                
            elif self.info[2] == 1:
                self.blocks.append((nn.Sequential(
                    conv(Cin, Cmid, kernel_size=1, bias=False),
                    bnorm(Cmid), nn.ReLU(),
                    mpool(kernel_size=2, stride=2),
                    conv(Cmid, Cout, kernel_size=1, bias=False),
                    bnorm(Cout))
                    if stride == 2 else nn.Sequential(
                            conv(Cin, Cmid, kernel_size=1, bias=False),
                            bnorm(Cmid), nn.ReLU(),
                            conv(Cmid, Cout, kernel_size=1, bias=False),
                            bnorm(Cout))))
    
    def forward(self, x):
        out = self.relu(self.blocks[0](x))
        for i in range(1, self.info[3]):
            out = self.relu(self.blocks[i](out) + out) 
        return out
    

class DecoderLayer(nn.Module):
    """
    nn.Module, creates entire EndoNet encoder layer.

    Parameters:
    ----------
    Dout (int) : output spatial dimensions. 2 for single mask, 3 for volumetric.
    depth(int) : number of blocks in each encoder layer.
    Cin (int) : expected input channels for first convolutional operation.
    Cout (int) : expected ouput channels for entire layer.

    Returns:
    -------
    None.
    """
    
    def __init__(self, Dout, depth, Cin, Cout):
        super(DecoderLayer, self).__init__()
        self.info = (Dout, depth)
        
        self.blocks = nn.ModuleList()
        self.relu = nn.ReLU()
        self.getBlock(Cin)
        self.getBlock(Cin, Cout)
        for x in range(2, self.info[1]):
            self.getBlock(Cout, Cout)
    
    def getBlock(self, Cin, Cout=None):
        conv = nn.Conv2d if self.info[0] == 2 else nn.Conv3d
        tconv = nn.ConvTranspose2d if self.info[0] == 2 else nn.ConvTranspose2d
        bnorm = nn.BatchNorm2d if self.info[0] == 2 else nn.BatchNorm3d
        
        if Cout is None:
            self.blocks.append(nn.Sequential(
                tconv(Cin, Cin, kernel_size=2, stride=2),
                bnorm(Cin)))
                
        elif Cout is not None:
            self.blocks.append(nn.Sequential(
                conv(Cin, Cout, kernel_size=3, padding=1, bias=False),
                bnorm(Cout)))
    
    def forward(self, x, skip):
        def adjustDims(skip, target): # target dimensions are after tconv
            skip = torch.mean(skip, dim=2) if skip.dim() != target.dim() else skip
            yx = np.asarray(target.size()[2:])
            dyx = (yx - np.asarray(skip.size()[2:])) // 2
            yx += dyx
            if target.dim() == 5:
                return skip[:,:, dyx[0]:yx[0], dyx[1]:yx[1], dyx[2]:yx[2]]
            return skip[:,:, dyx[0]:yx[0], dyx[1]:yx[1]]
        
        out = self.relu(self.blocks[0](x))
        skip = adjustDims(skip, out) if (skip.size() != out.size()) else skip
        out = self.relu(self.blocks[1](out + skip))
        for i in range(2, self.info[1]):
            out = self.relu(self.blocks[i](out) + out)  
        return out


class EndoNet(nn.Module):
    """
    nn.Module, creates an EndoNet Convolutional Network.

    Parameters:
    ----------
    organelle (str) : mito or ER, organelle that model was/will be trained to segment.
    ver (int) : model version. The default is 18, others are 34, 50, 101, 152.
    Din (int) : input spatial dimensions. 2 for time series, 3 for volumetric.
    Dout (int) : output spatial dimensions. 2 for single mask, 3 for volumetric.
    dowmtype (str) : reduce dimensions with "conv" for convolution or "max" for maxpool.
    Cin (int) : expected input channels for first convolutional operation.
    nClasses (int) : expected number of class labels in ground truth.
    timeDepth (int) : expected time dimension length. Must be divisible by 16.

    Returns:
    -------
    None.
    """
    
    def __init__(self, params):
        super(EndoNet, self).__init__()
        stats = {
            18: (0, 2, 2, 2, 2), 101: (1, 3, 4, 23, 3),
            34: (0, 3, 4, 6, 3), 152: (1, 3, 8, 36, 3),
            50: (1, 3, 4, 6, 3)}

        stats, down = stats[params("vers")], params("reduce")
        Din, Dout = params("Din"), params("Dout")
        Cin, Cout = params("Cin"), params("Cout")
        scale = 4 if (stats[0] == 1) else 1
        Cs = ([Cin, 4, 4*scale, 8*scale, 16*scale, 32*scale] if Din == 3 else
              [Cin, 64, 64*scale, 128*scale, 256*scale, 512*scale])
        
        self.extraDim = True if Dout == 2 else False

        self.main = EncoderLayer(
            Din=Din, down=down, mode=stats[0], depth=1,
            Cin=Cs[0], Cout=Cs[1])
        self.enc1 = EncoderLayer(
            Din=Din, down=down, mode=stats[0], depth=stats[1],
            Cin=Cs[1], Cout=Cs[2])
        self.enc2 = EncoderLayer(
            Din=Din, down=down, mode=stats[0], depth=stats[2],
            Cin=Cs[2], Cout=Cs[3], stride=2)
        self.enc3 = EncoderLayer(
            Din=Din, down=down, mode=stats[0], depth=stats[3],
            Cin=Cs[3], Cout=Cs[4], stride=2)
        self.enc4 = EncoderLayer(
            Din=Din, down=down, mode=stats[0], depth=stats[4],
            Cin=Cs[4], Cout=Cs[5], stride=2)

        self.pool = (nn.MaxPool3d(kernel_size=2, stride=2) if Dout == 3 else
                     nn.MaxPool2d(kernel_size=2, stride=2))

        self.dec4 = DecoderLayer(
            Dout=Dout, depth=stats[4], Cin=Cs[5], Cout=Cs[4])
        self.dec3 = DecoderLayer(
            Dout=Dout, depth=stats[3], Cin=Cs[4], Cout=Cs[3])
        self.dec2 = DecoderLayer(
            Dout=Dout, depth=stats[2], Cin=Cs[3], Cout=Cs[2])
        self.dec1 = DecoderLayer(
            Dout=Dout, depth=stats[1], Cin=Cs[2], Cout=Cs[1])
        self.last = EncoderLayer(
            Din=Dout, down=down, mode=stats[0], depth=1, Cin=Cs[1], Cout=Cout)

    def forward(self, x):            
        out = self.main(x)  # [N, Cs[1], H, W]
        lay1 = self.enc1(out)  # [N, Cs[2], H, W]
        lay2 = self.enc2(lay1)  # [N, Cs[3], H/2, W/2]
        lay3 = self.enc3(lay2)  # [N, Cs[4], H/4, W/4]
        lay4 = self.enc4(lay3)  # [N, Cs[5], H/8, W/8]
        
        if self.extraDim and lay4.dim() > 4:
            lay4 = torch.mean(lay4, dim=2)
        elif not self.extraDim and lay4.dim() < 5:
            lay4 = torch.unsqueeze(lay4, dim=2)

        out = self.pool(lay4)  # [N, Cs[5], H/16, W/16]
        
        out = self.dec4(out, lay4)  # [N, Cs[4], H/8, W/8]
        out = self.dec3(out, lay3)  # [N, Cs[3], H/4, W/4]
        out = self.dec2(out, lay2)  # [N, Cs[2], H/2, W/2]
        out = self.dec1(out, lay1)  # [N, Cs[1], H, W]
        out = self.last(out)  # [N, nClasses, H, W]
        return out
