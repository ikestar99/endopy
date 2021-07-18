#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 09:34:52 2021

@author: ike
"""


from glob import glob

from torch.utils.data import DataLoader

from .endodataset import FlyGuys
from .mitodataset import MitoGuys
from ..utils.pathutils import getPath
from ..utils.visualization import wait


DSETS = {
    "FlyGuys": FlyGuys,
    "MitoGuys": MitoGuys}


def getRawData(cfg):
    if cfg["Dataset"] == "FlyGuys":
        rawData = glob(getPath(cfg["DataDir"], "*"))
        rawData = [folder for folder in rawData if (
            len(glob(getPath(folder, "**", cfg["RGECODirs"]), recursive=True)) *
            len(glob(getPath(folder, "**", cfg["ER210Dirs"]), recursive=True)) > 0)]
    elif cfg["Dataset"] == "MitoGuys":
        rawData = glob(getPath(cfg["ImageDir"], "**", "*", ext="tif*"), recursive=True)
    
    return rawData


def toTrain(cfg):
    rawData = getRawData(cfg)
    train = DSETS[cfg["Dataset"]](cfg, rawData[::2])
    valid = DSETS[cfg["Dataset"]](cfg, rawData[1::2])
    if cfg["Weights"] != "None":
        cfg["Weights"] = train.getWeights()
        cfg.save(weights=True)
        wait("\n".join(
            ["\n", "Computed classwise weights"] +
            ["    Class {}: {}".format(x, cfg["Weights"][x]) for x in range(
                len(cfg["Weights"]))] + ["\n"]))
    
    loaders = dict(
        Train=DataLoader(train, batch_size=cfg["BatchSize"], shuffle=True),
        Valid=DataLoader(valid, batch_size=cfg["BatchSize"], shuffle=True))
    wait("\n".join((
        "Training datasets acquired",
        "T samples: {}, V samples: {}".format(len(train), len(valid)))))
    return loaders


def toTest(cfg):
    rawData = getRawData(cfg)
    test = DSETS[cfg["Dataset"]](cfg, rawData)
    loader = dict(
        Test=DataLoader(test, batch_size=cfg["BatchSize"], shuffle=True))
    wait("\n".join((
        "Test dataset acquired", "Test samples: {}".format(len(test)))))
    return loader


def toPredict(cfg, imageShape=None):
    rawData = getRawData(cfg)
    if cfg["Dataset"] == "FlyGuys":
        for folder in rawData:
            if imageShape is not None:
                cfg.resize("Hin", imageShape[0])
                cfg.resize("Win", imageShape[1])

            loader = dict(
                Folder=folder,
                Predict= DataLoader(FlyGuys(cfg, [folder], masks=False)), batch_size=cfg["BatchSize"])
            yield loader
        
    elif cfg["Dataset"] == "MitoGuys":
        predict = MitoGuys(cfg, rawData, masks=False)
        loader = dict(
            Predict=DataLoader(predict, batch_size=cfg["BatchSize"]),
            MitoGuy=predict)
        wait("\n".join((
            "Prediction mito dataset acquired",
            "P samples: {}".format(len(predict)))))
        yield loader; return


def toPipeline(flyCFG):
    predict = FlyGuys(flyCFG, masks=False)
    predict = DataLoader(predict, batch_size=flyCFG["BatchSize"])
    return predict