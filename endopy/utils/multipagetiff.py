#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  7 13:06:21 2021
Core code obtained from Vicky Mak, Barnhart Lab
Class structure and relative imports from Ike Ogbonna, Barnhart Lab
@author: ike
"""

import io
from PIL import Image, ImageSequence
from glob import glob
import numpy as np
import os.path as op
import seaborn as sns
import scipy.ndimage as scn
import skimage.measure as skm
import matplotlib.pyplot as plt

from .pathutils import getPath, changeExt, makeParentDirectory


class MultipageTiff(object):
    figures = list()

    def __init__(self, imagePages):
        if (type(imagePages) == str) and op.isdir(imagePages):
            imagePages = glob(getPath(imagePages, "*", ext="tif"))
        elif (type(imagePages) == list) and (len(imagePages) == 1):
            imagePages = imagePages[0]

        if type(imagePages) == str:
            multipage = Image.open(imagePages)
            self.imageArray = np.array(
                [np.array(page) for page in ImageSequence.Iterator(multipage)],
                dtype=np.uint8)
        elif type(imagePages) == list:
            self.imageArray = np.array(
                [self.getImage(imageFile) for imageFile in sorted(imagePages)],
                dtype=np.uint8)
        else:
            raise ValueError ("Trying to make an empty image")

    @staticmethod
    def getImage(imageFile, mask=False):
        image = (np.array(Image.open(imageFile)) > 0).astype(int)
        image = (np.ma.masked_equal(image, 0) if mask else image)
        return image

    @staticmethod
    def unit8Image(image, normalize=False):
        image = image + abs(min(0, np.min(image)))
        oldMax = (np.max(image) if np.max(image) != 0 else 1)
        newMax = (255 if normalize else min(np.max(image), 255))
        image = np.rint((image / oldMax) * newMax).astype(np.uint8)
        return image

    @staticmethod
    def savePillowArray(pilArray, saveFile):
        saveFile = changeExt(saveFile, ext="tif")
        makeParentDirectory(saveFile)
        pilArray[0].save(
            saveFile, compression="tiff_deflate",
            save_all=True, append_images=pilArray[1:])

    @classmethod
    def saveImage(cls, image, save, normalize=False):
        if image.dtype != "uint8" or normalize:
            image = cls.unit8Image(image, normalize)

        image = Image.fromarray(image, mode="L")
        save = changeExt(save, ext="tif")
        image.save(save, compression="tiff_deflate")

    @classmethod
    def addFigure(
            cls, Y, name, yLabel, color="gray", axLabel=None, X=None, dY=None,
            lightOn=None, bkg=None):
        fig, ax = plt.subplots(figsize=(4, 3), dpi=300)
        sns.set_theme(font_scale=0.5)
        sns.set_style("ticks")
        sns.despine()
        X = (list(range(1, Y.shape[0] + 1)) if X is None else X)
        if bkg is not None:
            ax.plot(X, bkg, color="black", label="background")
        if type(color) == list:
            for x in range(len(color)):
                ax.plot(X, Y[x], color=color[x], label=yLabel[x])
        else:
            ax.plot(X, Y, color=color, label=yLabel)

        ax.set_title(name)
        if axLabel is not None:
            ax.set_xlabel(axLabel[0])
            ax.set_ylabel(axLabel[1])
        if dY is not None:
            ax.fill_between(X, (Y - dY), (Y + dY), color=color, alpha=0.2)
        if lightOn is not None:
            plt.axvspan(
                lightOn[0], lightOn[1], color="blue", lw=1, alpha=0.1)

        ax.legend(loc="upper right")
        ax.locator_params(axis="x", nbins=10)
        ax.locator_params(axis="y", nbins=6)
        ax.set_xlim(left=float(X[0]), right=float(X[-1]))
        # ax.set_ylim(bottom=np.min(Y), top=np.max(Y))
        # ax.spines['left'].set_position("zero")
        # if bkg is not None:
        #     ax.spines['bottom'].set_position("zero")

        buffer = io.BytesIO()
        fig.savefig(buffer)
        buffer.seek(0)
        cls.figures += [Image.open(buffer)]
        plt.close("all")

    @classmethod
    def saveFigures(cls, saveFile):
        if len(cls.figures) != 0:
            cls.savePillowArray(cls.figures, saveFile)
            cls.figures = list()

    def __len__(self):
        return self.imageArray.shape[0]

    def __add__(self, other):
        if not isinstance(other, MultipageTiff):
            return self

        y = min(self.imageArray.shape[1], other.imageArray.shape[1])
        x = min(self.imageArray.shape[2], other.imageArray.shape[2])
        self.imageArray = np.concatenate(
            (self.imageArray[:, :y, :x], other.imageArray[:, :y, :x]),
            axis=0)
        return self

    def __radd__(self, other):
        return self + other

    def __getitem__(self, idx):
        return self.imageArray[idx]

    def __setitem__(self, idx, value):
        self.imageArray[idx] = value

    def __call__(self, other, mode):
        if (not isinstance(other, MultipageTiff) or
                (self.imageArray.shape[1:] != other.imageArray.shape[1:])):
            pass
        elif mode == "concatenate":
            self.imageArray = np.concatenate(
                (self.imageArray, other.imageArray), axis=0)
        elif mode == "sum":
            idx = min(self.imageArray.shape[0], other.imageArray.shape[0])
            self.imageArray = self.imageArray[:idx] + other.imageArray[:idx]

        return self

    def averageProjection(self):
        averageProjection = np.mean(self.imageArray, axis=0)
        return averageProjection

    def correct(self, backFile):
        bkg = self.getImage(backFile, True)[np.newaxis]
        bkg = np.mean((self.imageArray * bkg), axis=(-2, -1))
        return bkg

    def getMaskedROIs(self, maskFile):
        mask, regions = scn.label(self.getImage(maskFile, True))
        self.saveImage(np.ma.filled(mask, fill_value=0), maskFile)
        ROI, sizes = np.unique(mask[np.nonzero(mask)], return_counts=True)
        avgs = np.array(
            [np.mean(np.copy(self.imageArray[:, (mask == l)]), axis=-1)
             for l in range(regions)])
        return avgs, ROI, sizes

    def blockReduce(self, kSize):
        self.imageArray = skm.block_reduce(
            self.imageArray, block_size=(1, kSize, kSize), func=np.sum)

    def pad(self, Hin, Win):
        height, width = self.imageArray.shape[1:]
        dy, dx = Hin - height, Win - width
        self.imageArray = np.pad(
            self.imageArray, pad_width=(
                (0, 0), (0, max(0, dy)), (0, max(0, dx))),
            mode="reflect")[:, :Hin, :Win]

    def update(self, imageArray, normalize=False):
        self.imageArray = self.unit8Image(imageArray, normalize=normalize)

    def save(self, saveFile, mode="L", normalize=False):
        self.imageArray = self.unit8Image(self.imageArray, normalize=normalize)
        pilArray = [
            Image.fromarray(self.imageArray[idx], mode=mode)
            for idx in range(len(self))]
        self.savePillowArray(pilArray, saveFile)
