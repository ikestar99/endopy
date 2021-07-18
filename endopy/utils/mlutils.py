#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 03:04:49 2021

@author: ike
"""


import numpy as np

import torch


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# def normalizeProbArray(probArray):
#     negMins = -np.min(probArray, axis=0)
#     probArray += (negMins * (negMins > 0))
#     rowSums = np.sum(probArray, axis=0) + (np.sum(probArray, axis=0) == 0)
#     return probArray / rowSums


def makePrediction(model, batch, pbar=None):
    out = model(batch["In"].double().to(DEVICE)).cpu().detach().numpy()
    if pbar is not None:
        pbar.update(1)

    return np.mean(out, axis=0)


def getAveragePrediction(model, loader, pbar):
    model.eval(); torch.set_grad_enabled(False)
    runAvg = np.mean(np.array(
        [makePrediction(model, batch, pbar) for batch in loader]), axis=0)
    return runAvg
