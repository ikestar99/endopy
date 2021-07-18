#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
Deep Residual Learning for Image Recognition
arXiv:1512.03385
@author: ike
"""
import os
import numpy as np

import torch
import torch.nn as nn
"""import pytorch_lightning as pl"""


class PLNet("""pl.LightningModule"""):
    """
    pl.LightningModule, creates a lightning-wrapped Convolutional Network.

    Parameters:
    ----------
    model (nn.Module) : model to be trained. Instantiate in package __init__.py.
    weights (torch.tensor) : weights for loss function. Must have len = nClasses.
    modelDir (str) : usr/.../dir with different models
    lr (float) : initial learning rate. The default is 0.01.
    gamma (float): exponential decay factor. The default is 0.794328.

    Returns:
    -------
    None.
    """
    
    def __init__(self, model, weights, modelDir, lr, gamma):
        # super(PLNet, self).__init__()
        self.model = model.double()
        self.criterion = nn.CrossEntropyLoss(weight=weights.double())
        self.lr, self.gamma = lr, gamma
        
        self.epoch = {"Train": 0, "Valid": 0}
        self.statistics = {"Train": None, "Valid": None}
        self.saveDir = "{}/{}".format(modelDir, "PL{}".format(self.model.name))  
        if not os.path.exists(self.saveDir):
            os.makedirs(self.saveDir)
        print("Instantiated: PL{}".format(self.model.name))
    
    def configure_optimizers(self):
        """
        get optimizer and exponential decay learning rate scheduler.

        Returns:
        -------
        optim (torch.optim.adam.Adam) : Adam optimizer.
        lrScheduler (torch.optim.lr_scheduler.ExponentialLR) : decaying learning rate.
        """
        
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer, gamma=self.gamma)
        return [optimizer], [scheduler]
    
    def getStats(self, Y, out, loss):
        """
        helper function to get iteration accuracy and loss values.

        Parameters:
        ----------
        Y (torch.tensor) : ground truth. Remove batch dimension first.
        out (torch.tensor) : model prediction. Remove batch dimension first.
        loss (float) : loss.item().

        Returns:
        -------
        stats (list): [class 1 accuracy, ..., class n accuracy, total, loss].
        """
        
        nClasses = out.size()[0]
        out = np.argmax(out.cpu().detach().numpy(), axis=0)
        Y = Y.cpu().detach().numpy()        
        stats = (
            [np.sum((out == j) * (Y == j)) / np.sum(Y == j)
             for j in range(nClasses)] + [(np.sum(out == Y) / Y.size), loss])
        return stats

    def training_step(self, batch, idx):
        print("Train epoch: {}, batch: {}".format(self.epoch["Train"], idx))
        X, Y = batch[0].double(), batch[1].long() # [N, Cin, <D>, H, W]
        out = self.model(X) #[N, classes, <D>, H, W]
        loss = self.criterion(out, Y)
        return self.getStats(Y[0], out[0], loss.item())
    
    def training_epoch_end(self, trainOutputs):
        torch.save(self.model.state_dict(), "{}/{}.pt".format(self.saveDir, self.model.name))
        data = np.array(trainOutputs)
        self.statistics["Train"] = (
            data[np.newaxis] if self.statistics["Train"] is None else
            np.concatenate((self.statistics["Train"], data[np.newaxis]), axis=0))
        np.save("{}/Train Statistics.npy".format(self.saveDir), self.statistics["Train"])
        self.epoch["Train"] += 1
        
    def validation_step(self, batch, idx):
        print("Valid epoch: {}, batch: {}".format(self.epoch["Valid"], idx))
        X, Y = batch[0].double(), batch[1].long() # [N, Cin, <D>, H, W]
        out = self.model(X) #[N, classes, <D>, H, W]
        return self.getStats(Y[0], out[0], self.criterion(out, Y).item())

    def validation_epoch_end(self, validOutputs):
        data = np.array(validOutputs)
        self.statistics["Valid"] = (
            data[np.newaxis] if self.statistics["Valid"] is None else
            np.concatenate((self.statistics["Valid"], data[np.newaxis]), axis=0))
        np.save("{}/Valid Statistics.npy".format(self.saveDir), self.statistics["Valid"])
        self.epoch["Valid"] += 1