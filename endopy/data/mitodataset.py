#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 19:09:37 2020

@author: ike
"""
    

import numpy as np
from glob import glob

import torch

from ..utils.base import BaseDataset
from ..utils.pathutils import getPath, getPathName, csvDict, namePathDict


class MitoGuys(BaseDataset):    
    def __init__(self, cfg, rawData, masks=True):
        super(MitoGuys, self).__init__(cfg, masks)
        
        self.parseSamples(rawData)
        if len(self) == 0:
            raise IndexError ("MitoGuys dataset could not find raw data")
        
    def parseSamples(self, mitoImages):
        def getLabels():
            csvFiles = glob(getPath(self.cfg["CSVDir"], "*.csv"))
            labels = {}
            for file in csvFiles:
                new = csvDict(file, self.cfg["ClassMap"])
                labels.update(
                    {key: new[key] for key in new if key not in labels})
                for key in new:
                    if labels[key] != new[key]:
                        del labels[key]
            
            return labels
        
        def getMasks():
            masks = glob(getPath(self.cfg["MaskDir"], "*.tif*"))
            masks = namePathDict(masks)
            return masks
    
        mitoImages = namePathDict(mitoImages)
        if self.masks:
            reference = (getLabels() if self.num else getMasks())
        else:
            reference = {key: key for key in mitoImages}
        
        keys = sorted([key for key in mitoImages if key in reference])
        keys = self.adjustKeys(keys)
        self.samples = np.array(
            [(mitoImages[keys[x]], reference[keys[x]])
             for x in range(len(keys))])
        
    def getWeights(self):
        weights = np.zeros((self.cfg["Cout"]))
        for x in range(len(self)):
            if self.num:
                weights[int(self.samples[x,1])] += 1
            else:
                mask = self.getImage(self.samples[x,1], color="grayscale")
                cdx, frequency = np.unique(mask, return_num=True)
                for x in range(cdx.size):
                    weights[cdx[x]] += frequency[x]
        
        weights = ((1 / weights) / np.sum((1 / weights))).tolist()
        return weights
    
    def __call__(self, idx):
        return getPathName(self.samples[idx,0])
    
    def __getitem__(self, idx):
        sample = self.getImage(self.samples[idx,0])[np.newaxis]
        if self.masks:
            if self.num:
                mask = int(self.samples[idx,1])
            else:
                if ".np" in self.samples[idx,1]:
                    mask = np.load(self.samples[idx,0])
                else:
                    mask = self.getImage(self.samples[idx,1], color="grayscale")
            
            return dict(In=torch.tensor(sample), GT=torch.tensor(mask))
        else:
            return dict(In=torch.tensor(sample), FN=np.array([idx]))