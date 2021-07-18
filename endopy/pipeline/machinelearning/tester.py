#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 21:28:00 2020


@author: ike
"""

import numpy as np
import os.path as op

import torch
import torch.nn as nn

from ...main import tqdm
from ...utils.pathutils import changeExt
from ...models.modelsmain import getModel
from ...utils.visualization import plotIterations, wait


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Tester(nn.Module):
    def __init__(self, cfg):
        super(Tester, self).__init__()
        self.cfg = cfg
        self.model, _ = getModel(cfg, load=True)
        if self.cfg["Weights"] == "None":
            self.criterion = nn.CrossEntropyLoss()
        else:
            weights = torch.tensor(self.cfg["Weights"]).double().to(DEVICE)
            self.criterion = nn.CrossEntropyLoss(weight=weights)
        
        self.track = {}
        wait("{} Tester prepared".format(self.cfg["Name"]))

    def save(self):        
        np.savez_compressed(
            changeExt(self.cfg["Test"], ext="npz"), **self.track)
        
    def testLoop(self, loader):        
        def getStats(GT, out, loss, mode):
            out = np.argmax(out.cpu().detach().numpy(), axis=1)
            GT = GT.cpu().detach().numpy()
            stats = [(np.sum(out == GT) / GT.size), loss]
            if self.cfg["OutputType"] != "classification":
                stats = [np.sum(
                    (GT == idx) * (out == idx))  / max(1, np.sum(GT == idx))
                    for idx in range(self.cfg["Cout"])] + stats
            stats = np.array(stats)[np.newaxis]
            self.track[mode] = (
                stats if not (mode in self.track) else np.concatenate(
                    (self.track[mode], stats), axis=0))
        
        def testStep(batch):
            if batch["GT"][0] is None:
                return
            
            In = batch["In"].double().to(DEVICE)
            GT = batch["GT"].long().to(DEVICE)
            out = self.model(In) #[N, classes, <D>, H, W]
            loss = self.criterion(out, GT)
            getStats(GT, out, loss.item(), "Test")
            del In, GT, out; torch.cuda.empty_cache()
        
        if op.isfile(changeExt(self.cfg["Test"], ext="npz")):
            return
        else:
            self.model.eval(); torch.set_grad_enabled(False)
            [testStep(i) for i in tqdm(loader["Test"])]
            plotIterations(
                bSize=self.cfg["BatchSize"], save=self.cfg["Test"],
                **self.track)  
            self.save()
            wait("Test sequence for {} completed successfully".format(
                self.cfg["Name"]))