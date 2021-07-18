#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 21:28:00 2020


@author: ike
"""

import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.transforms as tt


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class EndoSearcher(nn.Module):
    """
    nn.Module, parameter search, optimizes initial learning rate and decay.

    Parameters:
    ----------
    model (nn.Module) : model to be evaluated. Instantiate in package __init__.py.
    loaders (tuple) : tuple (trainLoader, validLoader).
    modelDir (str) : directory in which to store data.
    weights (torch.tensor) : weights for loss function. Must have len = nClasses.
    lr (tuple) : initial learning rate parameter space. (start, stop, step).
    gamma (float): exponential decay factor parameter space. (start, stop, step).

    Returns:
    -------
    None.
    """

    def __init__(self, model, loaders, modelDir, epochs, window):
        super(EndoSearcher, self).__init__()
        self.reference = model
        self.crop = tt.CenterCrop(window)
        self.loaders, self.epochs = loaders, epochs
        self.saveDir = "{}/{} [epochs: {}, window: {}]".format(
            modelDir, model.name, epochs, window)  
        if not os.path.isdir(self.saveDir):
            os.makedirs(self.saveDir)
    
    def learnSearch(self, weights, lrs, gammas, parallel):
        for lr in lrs:
            for gamma in gammas:
                self.trainingLoop(weights, lr, gamma, parallel)
    
    def initSearch(self, weights, lr, gamma):
        pass
    
    def trainingLoop(self, weights, lr, gamma, parallel):
        """
        trains a model.
        
        Parameters:
        ----------
        weights (torch.tensor) : weights for loss function. Must have len = nClasses.
        lrs (list) : learning rates to evaluate.
        gamma (list): exponential decay factors to evaluate.

        Returns:
        -------
        None.
        """
        
        def getStats(Y, out, loss):            
            nClasses = out.size()[0]
            out = np.argmax(out.cpu().detach().numpy(), axis=0)
            Y = Y.cpu().detach().numpy()
            stats = [(np.sum((out == j) * (Y == j))) / np.sum(Y == j)
                     if np.sum(Y == j) else (1 - (np.sum(out == j) / Y.size))
                     for j in range(nClasses)]
            stats += [(np.sum(out == Y) / Y.size), loss]
            return stats
        
        def trainStep(batch):
            self.optim.zero_grad()
            X = self.crop(batch[0]).double().to(DEVICE)
            Y = self.crop(batch[1]).long().to(DEVICE)
            out = self.model(X) #[N, classes, <D>, H, W]
            loss = self.criterion(out, Y)
            loss.backward()
            self.optim.step()
            stats = getStats(Y[0], out[0], loss.item())
            del X, Y, out; torch.cuda.empty_cache()
            return stats
            
        def validStep(batch):
            X = self.crop(batch[0]).double().to(DEVICE)
            Y = self.crop(batch[1]).long().to(DEVICE)
            out = self.model(X)
            loss = self.criterion(out, Y)
            stats = getStats(Y[0], out[0], loss.item())
            del X, Y, out; torch.cuda.empty_cache()
            return stats
        
        coordinate = "weights {}, lr {}, gamma {}".format(
            (weights.cpu().detach().numpy() if weights is not None else weights), lr, gamma)
        if os.path.isfile("{}/{}.txt".format(self.saveDir, coordinate)):
            return
        
        self.model = self.reference.double().to(DEVICE)
        self.criterion = (nn.CrossEntropyLoss(weight=weights.double().to(DEVICE))
                     if weights is not None else nn.CrossEntropyLoss())
        self.optim = torch.optim.Adam(params=self.model.parameters(), lr=lr)
        lrScheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optim, gamma=gamma)

        currentEpoch = 0
        check = "{}/checkpoint{}.pt".format(self.saveDir, parallel)
        if os.path.isfile(check):
            checkpoint = torch.load(check, map_location=DEVICE)
            if checkpoint["coordinate"] == coordinate:
                if checkpoint["epoch"] == 9:
                    return
                
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.optim.load_state_dict(checkpoint["optimizer_state_dict"])
                lrScheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                currentEpoch = checkpoint["epoch"] + 1

        for epoch in range(currentEpoch, self.epochs):
            print("Epoch [{}/{}]".format(epoch, self.epochs - 1))
            
            self.model.to(DEVICE).train(); torch.set_grad_enabled(True)
            trainStats = np.array([trainStep(i) for i in tqdm(self.loaders["Train"])])
            lrScheduler.step()
            
            self.model.eval(); torch.set_grad_enabled(False)
            validStats = np.array([validStep(j) for j in tqdm(self.loaders["Valid"])])
            
            means = (np.mean(trainStats, axis=0), np.mean(validStats, axis=0))
            means = np.concatenate(means, axis=0)
            stdvs =(np.std(trainStats, axis=0), np.std(validStats, axis=0))
            stdvs = np.concatenate(stdvs, axis=0)
            statistics = [[means[x], stdvs[x]] for x in range(len(means))]
            statistics = [stat for sublist in statistics for stat in sublist]
            statistics = np.array([[epoch] + statistics])
            
            if epoch > 0:
                old = torch.load(check, map_location=DEVICE)["statistics"]
                statistics = np.concatenate((old, statistics), axis=0)
            
            torch.save(
                {"model_state_dict": self.model.to(torch.device("cpu")).state_dict(),
                  "optimizer_state_dict": self.optim.state_dict(),
                  "scheduler_state_dict": lrScheduler.state_dict(),
                  "statistics": statistics,
                  "coordinate": coordinate,
                  "epoch": epoch}, check)
            
            if epoch == self.epochs - 1:
                np.savetxt("{}/{}.txt".format(self.saveDir, coordinate), statistics)