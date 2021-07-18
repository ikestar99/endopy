#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 09:39:44 2021

@author: ike
"""


import os.path as op

import torch

from .endonet import EndoNet
from .resunet import ResUNet, ResUNetClassifier


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODELS = dict(R=ResUNet, E=EndoNet, RC=ResUNetClassifier)


def getModel(params):
    model = MODELS[params("model")](params).double().to(DEVICE)
    checkpoint = None
    if op.isfile(params("mSave")):
        checkpoint = torch.load(params("mSave"), map_location=DEVICE)
        model.load_state_dict(checkpoint["Model"])

    return model, checkpoint