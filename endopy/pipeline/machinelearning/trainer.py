#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 21:28:00 2020


@author: ike
"""

import sys
import numpy as np
import pandas as pd
import os.path as op

import torch
import torch.nn as nn

from ...main import tqdm
from ...utils.base import BaseMachine
from ...models.modelsmain import getModel
from ...utils.visualization import plotIterations, wait


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Trainer(BaseMachine):
    sHead = ["mean", "STD", "N"]

    def __init__(self, mCFG):
        super(Trainer, self).__init__(mCFG)

    def __call__(self):
        wait(" \nCommencing training loop\n  ")
        params = self.getParameters()
        self.model, checkpoint = getModel(self.getParameters())
        self.optim = torch.optim.Adam(
            params=self.model.parameters(), lr=params("LR"))
        self.decay = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optim, gamma=params("Gamma"))
        self.criterion = nn.CrossEntropyLoss()
        # else:
        #     weights = torch.tensor(self.cfg["Weights"]).double().to(DEVICE)
        #     self.criterion = nn.CrossEntropyLoss(weight=weights)
        self.dfs = pd.DataFrame()
        epoch = 0
        if checkpoint is not None:
            if "Epoch" not in checkpoint:
                sys.exit("{} is already trained".format(params("Name")))

            self.optim.load_state_dict(checkpoint["Optimizer"])
            self.decay.load_state_dict(checkpoint["Scheduler"])
            self.dfs = pd.read_csv(params, usecols=self.sHead)
            epoch = checkpoint["Epoch"]

    def saveCheckpoint(self, epoch):        
        stateDicts = dict(
            Model=self.model.state_dict())
        if type(epoch) is not None:
            stateDicts.update(dict(
                Optimizer=self.optim.state_dict(),
                Scheduler=self.decay.state_dict(),
                Epoch=epoch))
            
        torch.save(stateDicts, self.cfg["Pause"])
        np.savez_compressed(self.cfg["Stats"], **self.track)
        if epoch is not None:
            wait("Checkpoint for {} saved successfully".format(
                self.cfg["Name"]))         

    def trainingLoop(self, loaders):        
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
        
        def trainStep(batch):
            self.optim.zero_grad()
            In = batch["In"].double().to(DEVICE)
            GT = batch["GT"].long().to(DEVICE)
            out = self.model(In) #[N, classes, <D>, H, W]
            loss = self.criterion(out, GT)
            loss.backward(); self.optim.step()
            getStats(GT, out, loss.item(), "Train")
            del In, GT, out; torch.cuda.empty_cache()
            
        def validStep(batch):
            In = batch["In"].double().to(DEVICE)
            GT = batch["GT"].long().to(DEVICE)
            out = self.model(In)
            loss = self.criterion(out, GT)
            getStats(GT, out, loss.item(), "Valid")
            del In, GT, out; torch.cuda.empty_cache()

        if op.isfile(self.cfg["Pause"]) and not self.load:
            wait("{} has a save on record, resume training instead".format(
                self.cfg["Name"]))
        else:  
            for epoch in range(self.epoch, self.cfg["Epochs"]):
                wait("Epoch [{}/{}]".format(epoch + 1, self.cfg["Epochs"]))
                
                self.model.train(); torch.set_grad_enabled(True)
                [trainStep(i) for i in tqdm(loaders["Train"])]
                
                self.model.eval(); torch.set_grad_enabled(False)
                [validStep(j) for j in tqdm(loaders["Valid"])]
                self.decay.step(); self.saveCheckpoint(epoch)
                
                plotIterations(
                    bSize=self.cfg["BatchSize"], save=self.cfg["TGraph"],
                    **self.track)
            
            self.saveCheckpoint(epoch=None)
            wait("Training for {} completed successfully".format(
                self.cfg["Name"]))