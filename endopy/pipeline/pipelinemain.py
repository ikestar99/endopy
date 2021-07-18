#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 26 18:45:52 2021

@author: ike
"""


import sys

from .treesorter import TreeSorter
from .machinelearning.trainer import Trainer
from .visualstimuli.imageprocessor import ImageProcessor
from .visualstimuli.responsemeasurer import ResponseMeasurer
from .visualstimuli.receptivefieldmapper import ReceptiveFieldMapper

from ..utils.configurator import Configurator


"""
TODO: Add options to use Jordan's alignment code on Z stacks'
"""


def runSection(cfg, FunctionalClass, stage):
    cfg.proceed(stage)
    functional = FunctionalClass()
    functional()


def processLeicaData():
    pCFG = Configurator("P")
    pCFG.proceed("***sort: does your data match the template?***")
    sorter = TreeSorter(pCFG)
    sorter()
    if sorter.inVivo():
        runSection(pCFG, ImageProcessor, stage="***Pre-processing***")
        runSection(pCFG, ResponseMeasurer, stage="***Measure Responses***")
        runSection(pCFG, ReceptiveFieldMapper,
                   stage="***Map Receptive Field Centers***")
    else:
        sys.exit("Image analysis pipeline for ex-vivo data completed")


def trainMachineLearning():
    mCFG = Configurator("M")
    trainer = Trainer(mCFG)
    trainer()


# def useMachineLearning():
#     mCFG = MachineCFG()