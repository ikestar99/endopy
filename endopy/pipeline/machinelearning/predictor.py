#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 21:28:00 2020


@author: ike
"""

import os
import numpy as np
import os.path as op
from PIL import Image as PIM

import torch
import torch.nn as nn

from ...main import tqdm
from ...utils.base import BasePipeline
from ...data.datamain import toPipeline
from ...utils.pathutils import getPath, csvSave, getPathName
from ...models.modelsmain import getModel
from ...utils.visualization import wait


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Predictor(nn.Module):
    cfg = None
    model = None
#
#     def __init__(self, cfg=None):
#         super(Predictor, self).__init__()
#
#         if cfg is not None:
#             Predictor.cfg = cfg
#             Predictor.model, _ = getModel(cfg=cfg, load=True)
#
#     def predictionLoop(self,):
#         def cdx(batch):
#             In = batch["In"].double().to(DEVICE)
#             out = self.model(In).cpu().detach().numpy()
#             return out
#
#         saveFile = getPath(
#             loader["Folder"], BasePipeline.predDir, "{} Prediction".format(self.cfg["Name"]),
#             ext="npy")
#         if op.isfile(saveFile):
#             return
#         elif not op.isdir(saveFile[:saveFile.rindex("/")]):
#             os.makedirs(saveFile[:saveFile.rindex("/")])
#
#         out = np.concatenate([cdx(batch) for batch in loader["Predict"]], axis=0)
#         out = np.mean(out, axis=0)
#         out = out + np.amin(out, axis=0)[np.newaxis]
#         out = out / np.sum(out, axis=0)[np.newaxis]
#         np.save(saveFile, out)
#
#     def predictionLoop(self):
#         def mitoSegmentation(batch):
#             fdx = np.reshape(batch["FN"].cpu().detach().numpy(), newshape=-1)
#             self.triples["Name"] += [self.dset(fdx[x]) for x in range(len(fdx))]
#             if self.cfg["OutputType"] == "mask":
#                 if op.isfile(getPath(self.mitoSave, fdx[0])):
#                     return
#
#             In = batch["In"].double().to(DEVICE)
#             out = self.model(In).cpu().detach().numpy()
#             out = np.argmax(out, axis=1)
#             if self.cfg["OutputType"] == "classification":
#                 out = np.reshape(out, newshape=-1)
#                 labels = np.array(
#                     [self.cfg["ClassMap"][str(out[x])] for x in
#                      range(len(out))])
#                 self.triples["Prediction"] = (
#                     out if self.triples["Prediction"] is None else
#                     np.concatenate((self.triples["Prediction"], labels), axis=0))
#                 self.triples["Class"] = (
#                     out if self.triples["Class"] is None else
#                     np.concatenate((self.triples["Class"], out), axis=0))
#             else:
#                 for x in range(out.shape[0]):
#                     mask = out[x] / max(np.amax(out[x]), 1)
#                     mask = (mask * 255).astype(np.uint8)
#                     mask = PIM.fromarray(mask)
#                     mask.save(
#                         getPath(self.mitoSave, self.dset(fdx[x])),
#                         compression="tiff_deflate")
#
#         if self.cfg["Dataset"] == "FlyGuys":
#
#         else:
#             self.dset = None
#             identifier = getPathName(self.cfg["ImageDir"])
#             self.mitoSave = getPath(self.cfg["DataDir"], "{} by {}".format(
#                 identifier, self.cfg["Name"]))
#             if self.cfg["OutputType"] == "classification":
#                 self.mitoSave = getPath(self.mitoSave, ext="csv")
#                 self.triples = dict(Name=[], Prediction=None, Class=None)
#             else:
#                 if not op.isdir(self.mitoSave):
#                     os.makedirs(self.mitoSave)
#
#             if self.cfg["OutputType"] == "classification":
#                 if op.isfile(self.mitoSave):
#                     return
#
#             for item in toPredict(self.cfg):
#                 loader, self.dset = item["Predict"], item["MitoGuy"]
#                 break
#
#             [mitoSegmentation(batch) for batch in tqdm(loader)]
#             if self.cfg["OutputType"] == "classification":
#                 csvSave(self.mitoSave, self.triples)
