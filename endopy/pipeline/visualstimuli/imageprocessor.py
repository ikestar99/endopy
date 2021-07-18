#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Original code by Erin Barnhart, Vicky Mak
Python port by Ike Ogbonna
Barnhart Lab, Columbia University 2021
"""


import sys
import numpy as np
import pandas as pd
import os.path as op
import scipy.stats as scs
import skimage.util as sku
import scipy.ndimage as scn
import skimage.feature as skf
import skimage.transform as skt
import skimage.segmentation as sks

from pystackreg import StackReg

from ...main import tqdm
from ...utils.base import BasePipeline
from ...data.datamain import toPipeline
from ...utils.mlutils import getAveragePrediction
from ...utils.pathutils import getPath, getPathName, changeExt
from ...models.modelsmain import getModel
from ...utils.visualization import wait
from ...configs.configsmain import getFlyCFGs
from ...utils.multipagetiff import MultipageTiff


class ImageProcessor(BasePipeline):
    tReg = StackReg(StackReg.RIGID_BODY)
    ccdx = 2
    ain4 = 2.437588
    ain4str = "AIN4"
    kSize = 2
    margin = 1.2
    mtpTiff = None
    threshold = 1

    def __init__(self):
        super(ImageProcessor, self).__init__()

    def __call__(self):
        wait(" \nAligning images")
        self.parseAndAlignFolders()

        wait(" \nCounting stimulus frames")
        self.countStimulusFrames()

        wait(" \nBinning images")
        self.binChannelFolders()

        wait(" \nCorrelating masks with machine learning")
        self.maskAndCorrelateFolders()

    @classmethod
    def parseAndAlignFolders(cls):
        def alignSample(row, cMax, pbar, ref=None):
            cns = [cMax] + [c for c in row("channels") if c != cMax]
            tif = [MultipageTiff(
                getPath(row("cPath", channel=c), ext="tif")) for c in cns]
            pbar.total += len(tif[0]) * 3
            for _ in range(3):
                ref = (tif[0].averageProjection() if ref is None else ref)
                for x in range(len(tif[0])):
                    mat = cls.tReg.register(ref, tif[0][x])
                    for y in range(len(tif)):
                        tif[y][x] = sku.img_as_ubyte(skt.warp(
                            tif[y][x], mat, order=0, mode="constant", cval=0))

                    pbar.update(1)

            for y in range(len(tif)):
                tif[y].save(getPath(row("cPath", channel=cns[y]), ext="tif"))

            avg = {c: t.averageProjection() for c, t in zip(cns, tif)}
            return avg

        dims = dict()
        with tqdm(total=0) as pbar:
            for sample, ids in cls.getUnique("idList"):
                row = cls.getRow(sample, ids[0])
                pbar.total, pbar.n, pbar.last_print_n = 0, 0, 0
                pbar.set_description(getPathName(sample), refresh=True)
                if op.isfile(row("avpTif")):
                    dims[sample] = MultipageTiff.getImage(
                        (row("avpTif"))).shape
                    continue

                rID = [i for i in ids if cls.sNames[-1] in i]
                rID = (rID[0] if len(rID) > 0 else sorted(ids)[0])
                row = cls.getRow(sample, rID)
                avgs = np.array([
                    np.mean(MultipageTiff(getPath(row(
                        "cPath", channel=c), ext="tif")).averageProjection())
                    for c in row("channels")])
                cMax = row("channels")[np.argmax(avgs)]
                run = alignSample(row, cMax, pbar=pbar)
                ref = np.copy(run[cMax])
                for sID in ids:
                    row = cls.getRow(sample, sID)
                    new = alignSample(row, cMax, pbar=pbar, ref=ref)
                    for k in new:
                        run[k] = (new[k] if k not in run else run[k] + new[k])

                dims[sample] = ref.shape
                for key in run:
                    base = "{} {}".format(changeExt(row("avpTif")), key)
                    MultipageTiff.saveImage(run[key], getPath(base, ext="tif"))

                ref = np.sum(np.array([v for v in run.values()]), axis=0)
                MultipageTiff.saveImage(ref, row("avpTif"))

        pbar.close(); del pbar
        if not cls.imDims[0] in cls.dfs:
            dfs = cls.dfs
            dfs[cls.imDims[0]] = dfs[cls.lHead3[0]].apply(lambda x: dims[x][0])
            dfs[cls.imDims[1]] = dfs[cls.lHead3[0]].apply(lambda x: dims[x][1])
            cls.save(dfs)

    @classmethod
    def countStimulusFrames(cls):
        for sample, ids in tqdm(cls.getUnique("idList")):
            for id in ids:
                row = cls.getRow(sample, id)
                if op.isfile(row("frmStim")) or ("frames" in row("newStim")):
                    continue

                dfs = pd.read_csv(row("newStim"))
                vts = np.squeeze(dfs[cls.ain4str].to_numpy())
                vts = ((vts > cls.threshold).astype(int) * cls.ain4)

                # Calculate change in voltage signal for each stimulus frame
                vts[0] = 0
                vts[1:] = (vts[1:] - vts[:-1])

                # count imaging frames from the change in voltage signal
                frames = np.zeros((vts.size, 2))
                for n in range(1, len(vts) - 1, 1):
                    frames[n] = frames[n - 1]
                    if all(((vts[n] > vts[n - 1]), (vts[n] > vts[n + 1]),
                            (vts[n] > cls.threshold))):
                        frames[n, 0] += 1
                    elif all(((vts[n] < vts[n - 1]), (vts[n] < vts[n + 1]),
                              (vts[n] < 0 - cls.threshold))):
                        frames[n, 1] -= 1

                dfs[cls.imDims[2]] = (
                        frames[:, 0] * np.sum(frames, axis=-1)).astype(int)
                MultipageTiff.addFigure(
                    Y=dfs[cls.imDims[2]].to_numpy(), name=id, yLabel="Trigger")

                # sync microscope frames to voltage signal
                dfs = dfs.sort_values(cls.imDims[2], ascending=True)
                dfs = dfs[dfs[cls.imDims[2]] >= 1]
                dfs.to_csv(row("frmStim"), encoding="utf-8", index=False)

            MultipageTiff.saveFigures(row("cntfFig"))

    @classmethod
    def binChannelFolders(cls):
        def binImages(mtpTiff, dfs, row):
            bndHz = None
            for _, b in enumerate(row("binTime")):
                frm = dfs[(dfs[cls.rrHead[1]] >= b) & (dfs[cls.rrHead[1]] < (
                        b + row("binT")))].copy()
                frm = frm[cls.imDims[2]].astype(int).tolist()
                if len(frm) == 0:
                    continue

                bnd = np.mean(np.array([mtpTiff[f - 1] for f in frm]), axis=0)
                bnd = bnd[np.newaxis]
                bndHz = (bnd if bndHz is None else np.concatenate(
                    (bndHz, bnd), axis=0))

            # bndEp = None
            # if row("manyEp"):
            #     dfs = dfs.sort_values(
            #         cls.rrHead[1], ascending=True, ignore_index=True)[
            #         cls.rrHead[2]].astype(int).to_numpy()
            #     rem = row("timing").size - (dfs.size % row("timing").size)
            #     dfs = (
            #         np.pad(
            #         dfs, (0, rem), mode="reflect") if rem != 0 else dfs)
            #     dfs = np.array(dfs).reshape(row("timing").size, -1)
            #     bndEp = np.array([
            #         np.mean([mtpTiff[dfs[r, c] - 1] for c in range(
            #             dfs.shape[-1])], axis=0)
            #         for r in range(dfs.shape[0])])

            return bndHz

        nFrames = dict()
        for row in tqdm(cls.iterRows(), total=cls.len()):
            columns = ([cls.rrHead[-1]] if not row("sCheck") else [])
            columns += cls.rrHead[1:3]
            dff = pd.read_csv(row("frmStim"), usecols=columns)
            dff = dff.groupby(columns[:-1]).median().reset_index()
            if cls.imDims[2] in cls.dfs:
                continue

            for c in row("channels"):
                mtpTiff = MultipageTiff(
                    getPath(row("cPath", channel=c), ext="tif"))
                nFrames[getPathName(row("newStim"))] = len(mtpTiff)
                if op.isfile(row("binTif", c)):
                    continue
                elif row("sCheck"):
                    binArray = binImages(mtpTiff, dff, row)
                else:
                    dff = dff.sort_values(
                        cls.rrHead[-1], ascending=True, ignore_index=True)
                    binArray = None
                    for e in sorted(dff[cls.rrHead[-1]].unique().tolist()):
                        dfs = dff[dff[cls.rrHead[-1]] == e].copy()
                        holder = binImages(mtpTiff, dfs, row)
                        binArray = (
                            holder if binArray is None else np.concatenate(
                                (binArray, holder), axis=0))

                mtpTiff.update(binArray)
                mtpTiff.save(row("binTif", c))

        if len(nFrames) > 0:
            dfs = cls.dfs
            dfs[cls.imDims[2]] = dfs[cls.lHead3[3]].apply(
                lambda x: nFrames[getPathName(x)])
            BasePipeline.dfs = dfs
            cls.save(dfs)

    @classmethod
    def maskAndCorrelateFolders(cls):
        def watershedMask(waterMap, maskSave, name):
            newMask = scn.binary_erosion(sks.watershed(
                -waterMap, mask=(waterMap != 0), watershed_line=True))
            MultipageTiff.saveImage(
                newMask, getPath(maskSave, name, ext="tif"))

        def getPixelSeries(row, col):
            return cls.mtpTiff[..., row // cls.kSize, col // cls.kSize]

        def assignLabel(mask, corrRefs, seed, Y, X):
            for row in (Y - 1, Y, Y + 1):
                for col in (X - 1, X, X + 1):
                    if all(((row < mask.shape[0]), (col < mask.shape[1]),
                            (mask[row, col] == 1))):
                        if all(((0 < row < mask.shape[0] - 1),
                                (0 < col < mask.shape[1] - 1))):
                            tile = mask[row - 1:row + 2, col - 1:col + 2]
                            if np.unique((tile * (tile > 1))).size > 2:
                                continue

                        corr = [scs.pearsonr(corrRefs[x], getPixelSeries(
                            row, col))[0] for x in range(len(corrRefs))]
                        if (corr[seed - 2] * cls.margin) >= max(corr):
                            mask[row, col] = seed
                            corrRefs[seed - 2] += getPixelSeries(row, col)
                            mask, corrRefs = assignLabel(
                                mask, corrRefs, seed, row, col)

            return mask, corrRefs

        # flyCFGs = getFlyCFGs()
        # with tqdm(total=0) as pbar:
        #     for flyCFG in flyCFGs.values():
        #         model, _ = getModel(flyCFG)
        #         for sample, ids in cls.getUnique("idList"):
        #             row = cls.getRow(sample, ids[0])
        #             save = getPath(row("predDir"), flyCFG["Name"], ext="npy")
        #             if op.isfile(save):
        #                 continue
        #
        #             flyCFG["DataDir"] = sample
        #             flyCFG.Data["Channel"] = getPathName(
        #                 changeExt(row("binTif", row("channels")[-1])))
        #             flyCFG.resize("Hin", row("shape")[0])
        #             flyCFG.resize("Win", row("shape")[1])
        #             loader = toPipeline(flyCFG)
        #             pbar.total += len(loader)
        #             pbar.set_description(flyCFG["Name"], refresh=True)
        #             avgPred = getAveragePrediction(
        #                 model, loader, pbar=pbar)[
        #                       :, :row("shape")[0], :row("shape")[1]]
        #             np.save(save, avgPred)

        # pbar.close()
        # del pbar
        for row in tqdm(cls.iterRows(), total=cls.len()):
            if op.isfile(row("bkgTif")):
                continue

            sys.exit(("Masking with machine learning is still under "
                      "development. Please make 'background.tif' and "
                      "ROI mask files for each of your samples and return"))
            # BNet1 = np.load(getPath(
            #     row("predDir"), flyCFGs["B1"]["Name"], ext="npy"))
            # CNet0 = np.load(getPath(
            #     row("predDir"), flyCFGs["C0"]["Name"], ext="npy"))
            # mask = (np.argmax(BNet1, axis=0) == 0).astype(int)
            # MultipageTiff.saveImage(mask, row("bkgTif"), normalize=True)
            # mask = (mask == 0) * CNet0[cls.ccdx]
            #
            # # PREDICTION WATERSHED
            # watershedMask(mask, row("maskDir"), "probability watershed mask")
            #
            # # CORRELATIONAL WATERSHED
            # keys = [key for key in ids if "moving" in key]
            # if len(keys) == 0:
            #     continue
            #
            # id = keys[0]
            # channels = self.varValue(sample, id, mode="C")
            # channel = (channels["ch01"] if "ch01" in channels else
            #            channels["ch00"])
            # self.mtpTiff = MultipageTiff(getPath(
            #     sample, id, "".join((self.bindDir, channel)),
            #     ext="tif"))
            # self.mtpTiff.blockReduce(self.kSize)
            #
            # corrMask = np.zeros(mask.shape)
            # corrMask[tuple(skf.peak_local_max(mask).T)] = 1
            # corrMask, features = scn.label(
            #     corrMask, structure=np.ones((3, 3)))
            # mask = corrMask + (mask > 0).astype(int)
            # corrRefs = []
            # for feature in range(features):
            #     YX = np.argwhere(mask == feature + 2)
            #     corrRefs += [np.sum(np.array(
            #         [getPixelSeries(YX[j, 0], YX[j, 1])
            #          for j in range(YX.shape[0])]), axis=0)]
            #
            # for feature in range(features):
            #     YX = np.argwhere(mask == feature + 2)
            #     for j in range(YX.shape[0]):
            #         mask, corrRefs = assignLabel(
            #             mask=mask, corrRefs=corrRefs, seed=(feature + 2),
            #             Y=YX[j, 0], X=YX[j, 1])
            #
            # MultipageTiff.saveImage(
            #     mask, getPath(
            #         row("maskDir"), "correlation watershed mask", ext="tif"))

        # sys.exit(("Please take a moment to review the masks generated by "
        #           "machine learning and update them as necessary"))
