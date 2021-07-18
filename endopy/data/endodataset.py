#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 21:28:00 2020


@author: ike
"""

import numpy as np

import torch

from ..utils.base import BaseDataset
from ..utils.pathutils import firstGlob, recursiveGlob
from ..utils.multipagetiff import MultipageTiff


class FlyGuys(BaseDataset):
    def __init__(self, cfg, masks=True):
        super(FlyGuys, self).__init__(cfg, masks)

        self.samples = None
        self.mask = None
        self.multipageTiff = sum(
            [MultipageTiff(f) for f in recursiveGlob(
                self.cfg["DataDir"], "**", self.cfg["Channel"], ext="tif")])
        if masks:
            self.mask = np.load(firstGlob(
                self.cfg["DataDir"], "**", self.cfg["MaskPath"]))

        self.parseSamples()
        if len(self) == 0:
            raise IndexError(
                ("FlyGuys dataset could not find raw data ",
                 "in: \n    {}".format(self.cfg["DataDir"])))
        
    def parseSamples(self):
        self.multipageTiff.pad(self.cfg["Hin"], self.cfg["Win"])
        self.samples = list(range(len(self.multipageTiff)))
        self.samples = np.array(self.adjustKeys(
            self.samples)).reshape(-1, self.cfg["SampleSize"])

    def __getitem__(self, idx):        
        sample = np.array(
            [self.multipageTiff[mdx] for _, mdx in enumerate(
                self.samples[idx])])
        mySize = (
            (self.cfg["Cin"], sample.shape[-2], sample.shape[-1])
            if self.cfg["Din"] == 2 else
            (self.cfg["Cin"], self.cfg["Depth"], sample.shape[-2],
             sample.shape[-1]))
        sample = dict(In=torch.tensor(
            np.reshape(sample, newshape=mySize)))
        if self.masks:
            sample["GT"] = torch.tensor(self.mask)

        return sample
