#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Original code by Erin Barnhart, Vicky Mak
Python port by Ike Ogbonna
Barnhart Lab, Columbia University 2021
"""


import numpy as np
import pandas as pd
import os.path as op
import matplotlib.pyplot as plt

from ...main import tqdm
from ...utils.base import BasePipeline
from ...utils.pathutils import getPath, getPathName, changeExt
from ...utils.pipeutils import boundInt, smooth
from ...utils.visualization import wait
from ...utils.multipagetiff import MultipageTiff


class ResponseMeasurer(BasePipeline):
    cStr = "{}-mean_PI".format

    def __init__(self):
        super(ResponseMeasurer, self).__init__()

    def __call__(self):
        wait(" \nMeasuring raw responses")
        self.measureAndPlotRawResponses()

        wait(" \nMeasuring individual responses")
        self.measureIndividualResponses()

        wait(" \nMeasuring and plotting average responses")
        self.measureAndPlotAverageResponses()

        wait(" \nMeasuring and plotting responses from binned data")
        self.measureAndPlotBinnedResponses()

    @classmethod
    def measureAndPlotRawResponses(cls):
        for row in tqdm(cls.iterRows(), total=cls.len()):
            dfm = None
            for c in row("channels"):
                rawFile = row("rawFile", c)
                if op.isfile(rawFile) and op.isfile(row("mesFile")):
                    continue

                # load image stack and apply background correction
                mpt = MultipageTiff(getPath(row("cPath", c), ext="tif"))
                dfr = None
                col = [str(x + 1) for x in range(row("frames"))]
                for mskFile in row("mskDirs"):
                    res, ROI, sizes = mpt.getMaskedROIs(mskFile)
                    dfs = pd.DataFrame(data=res, columns=col)
                    dfs.insert(0, cls.cStr(c), np.mean(res, axis=-1))
                    dfs.insert(0, cls.header[0], getPathName(row("sample")))
                    lay = getPathName(row("sample")).split("-") + [
                        changeExt(getPathName(mskFile))]
                    dfs.insert(1, cls.header[1], "-".join(lay[5:]))
                    dfs.insert(2, cls.header[2], ROI)
                    dfs.insert(3, cls.exHead[0], sizes)
                    dfr = (dfs if dfr is None else pd.concat(
                        (dfr, dfs), axis=0, ignore_index=True))

                dfr.to_csv(rawFile, encoding="utf-8", index=False)
                if dfm is None:
                    dfm = dfr[cls.header + [cls.exHead[0], cls.cStr(c)]].copy()
                else:
                    dfm[cls.cStr(c)] = dfr[cls.cStr(c)]

                dfr[cls.header[1]] = (
                    dfr[cls.header[1]].astype(str) + " ROI " +
                    dfr[cls.header[2]].astype(str))
                bkg = mpt.correct(row("bkgTif"))
                for name in dfr[cls.header[1]].unique().tolist():
                    dfs = dfr[dfr[cls.header[1]] == name].copy()[col]
                    MultipageTiff.addFigure(
                        Y=smooth(dfs.to_numpy()[0]), name=name,
                        yLabel="raw", axLabel=("Frame", "F"), bkg=bkg)

                MultipageTiff.saveFigures(row("rawrFig", c))

            if dfm is not None:
                dfm.to_csv(row("mesFile"), encoding="utf-8", index=False)

    @classmethod
    def measureIndividualResponses(cls):
        for row in tqdm(cls.iterRows(), total=cls.len()):
            for c in row("channels"):
                if op.isfile(row("indFile", c)):
                    continue

                # calc median global/relative stimulus time per frame
                dff = pd.read_csv(row("frmStim"), usecols=(
                    cls.rrHead[:-1] if row("sCheck") else cls.rrHead))
                dff = dff.drop_duplicates(
                    subset=cls.rrHead[2], keep="first", ignore_index=True)
                dff = dff.sort_values(cls.rrHead[-2], ascending=True)
                rel_time = dff[cls.rrHead[1]].to_numpy()

                # epoch first index, first frame of each epoch
                try:
                    efi = (rel_time[:-1] > (rel_time[1:] + row("offset")))
                    efi = list(set([0] + (np.nonzero(efi)[0] + 1).tolist()))
                    eln = (None if row("sCheck") else dff[
                        cls.rrHead[-1]].to_numpy()[efi])
                    efi[0] = efi[1] - row("eFrames")
                except IndexError:
                    continue

                # Find, resample, normalize, save responses to single epochs
                dfr = pd.read_csv(row("rawFile", c))
                dfi = None
                for idx, edx in enumerate(efi):
                    dft = dfr[cls.header].copy()
                    dft[cls.exHead[1]] = idx + 1
                    if not row("sCheck"):
                        dft[cls.exHead[-1]] = eln[idx]

                    col = [
                        str(boundInt((edx + x), 1, row("frames")))
                        for x in range(row("eFrames"))]
                    res = dfr[col].to_numpy()
                    bln = np.median(
                        res[:, :row("frameOn")], axis=-1)[:, np.newaxis]
                    res = (res - bln) / bln
                    dft = pd.concat((dft, pd.DataFrame(
                        res, columns=row("strTime"))), axis=1)
                    dfi = (dft if dfi is None else pd.concat(
                        (dfi, dft), axis=0, ignore_index=True))

                cols = cls.header + (
                    [cls.exHead[1]] if row("sCheck") else cls.exHead[1:])
                dfi = dfi.sort_values(cols, ascending=True, ignore_index=True)
                dfi.to_csv(
                    row("indFile", c), encoding="utf-8", index=False)

    @classmethod
    def measureAndPlotAverageResponses(cls):
        for row in tqdm(cls.iterRows(), total=cls.len()):
            for c in row("channels"):
                if op.isfile(row("avgFile", c)) or not op.isfile(
                        row("indFile", c)):
                    continue

                col = cls.header + ([] if row("sCheck") else [cls.exHead[2]])
                dfi = pd.read_csv(row("indFile", c))
                dfa = dfi.drop(columns=[cls.exHead[1]], axis=1).groupby(
                    col).mean().reset_index()
                dfa[cls.imDims[3]] = dfi[cls.exHead[1]].max()
                dfa.to_csv(row("avgFile", c), encoding="utf-8", index=False)

                # generate and save plots of average per-epoch responses
                dfa[cls.header[1]] = (
                    dfa[cls.header[1]].astype(str) + " ROI " +
                    dfa[cls.header[2]].astype(str) + " Response N: " +
                    dfa[cls.imDims[3]].astype(str))
                for name in dfa[cls.header[1]].unique().tolist():
                    dfr = dfa[dfa[cls.header[1]] == name].copy()
                    avg = dfr[row("strTime")].copy().to_numpy()
                    if row("sCheck"):
                        avg = smooth(avg[0])
                        lab = "average"
                        col = "Gray"
                    else:
                        avg = np.apply_along_axis(smooth, -1, avg)
                        lab = dfr[cls.exHead[-1]].to_numpy().astype(int)
                        col = [cls.colors[x - 1] for x in lab]

                    MultipageTiff.addFigure(
                        Y=avg, name=name, yLabel=lab, color=col,
                        axLabel=("Time (s)", "DF/F"), X=row("timing"))

                MultipageTiff.saveFigures(row("measFig", c))

    @classmethod
    def measureAndPlotBinnedResponses(cls):
        for row in tqdm(cls.iterRows(), total=cls.len()):
            for c in row("channels"):
                if not op.isfile(row("binTif", c)) or op.isfile(
                        row("bnaFile", c)):
                    continue

                # load image stack and apply background correction
                mpt = MultipageTiff(row("binTif", c))
                dfr = None
                col = [str(x + 1) for x in range(len(mpt))]
                for mskFile in row("mskDirs"):
                    res, ROI, sizes = mpt.getMaskedROIs(mskFile)
                    dfs = pd.DataFrame(data=res, columns=col)
                    dfs.insert(0, cls.cStr(c), np.mean(res, axis=-1))
                    dfs.insert(0, cls.header[0], getPathName(row("sample")))
                    lay = getPathName(row("sample")).split("-") + [
                        changeExt(getPathName(mskFile))]
                    dfs.insert(1, cls.header[1], "-".join(lay[5:]))
                    dfs.insert(2, cls.header[2], ROI)
                    dfs.insert(3, cls.exHead[0], sizes)
                    dfr = (dfs if dfr is None else pd.concat(
                        (dfr, dfs), axis=0, ignore_index=True))

                dfr.to_csv(row("bnaFile", c), encoding="utf-8", index=False)
                dfr[cls.header[1]] = (
                        dfr[cls.header[1]].astype(str) + " ROI " +
                        dfr[cls.header[2]].astype(str))
                bkg = mpt.correct(row("bkgTif"))
                for name in dfr[cls.header[1]].unique().tolist():
                    dfs = dfr[dfr[cls.header[1]] == name].copy()[col]
                    MultipageTiff.addFigure(
                        Y=smooth(dfs.to_numpy()[0]), name=name,
                        yLabel="binned", axLabel=("Frame", "F"), bkg=bkg)

                MultipageTiff.saveFigures(row("bnavFig", c))
