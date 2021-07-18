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
import scipy.stats as ss
import matplotlib.pyplot as plt

from ...main import tqdm
from ...utils.base import BasePipeline
from ...utils.pipeutils import smooth
from ...utils.pathutils import getPath
from ...utils.visualization import wait
from ...utils.multipagetiff import MultipageTiff


class ReceptiveFieldMapper(BasePipeline):
    threshold = 2
    stop = False

    def __init__(self):
        super(ReceptiveFieldMapper, self).__init__()

    def __call__(self):
        wait(" \nMapping receptive field centers")
        self.mapReceptiveFieldCenters()

        wait(" \nFiltering mapped responses")
        self.filterMappedResponses()

        wait(" \nPlotting mapped responses")
        self.plotMappedResponses()

    @classmethod
    def mapReceptiveFieldCenters(cls):
        for row in tqdm(cls.iterRows(), total=cls.len()):
            if row("sCheck") or op.isfile(row("rfcFile")) or not row("rCheck"):
                continue

            cls.stop = True
            dfi = pd.read_csv(row("avgFile", channel=row("channels")[-1]))
            dfi = dfi.sort_values(
                cls.header, ascending=True, ignore_index=True)
            dfc = dfi[row("strTime")].copy()
            dfi = dfi.drop(columns=row("strTime"))
            dfi[cls.rfHead[0]] = dfc.max(axis=1)
            dfi[cls.rfHead[1]] = dfc.idxmax(axis=1).astype(float)
            dfi[cls.rfHead[1]] = np.where(
                ((dfi[cls.exHead[-1]] % 2) != 0),
                ((dfi[cls.rfHead[1]] * 4) - 10),
                ((dfi[cls.rfHead[1]] * -4) + 11))
            dfi[cls.exHead[-1]] = np.where(
                (dfi[cls.exHead[-1]] < 3), 0, 1)
            dfc = dfi[cls.header].copy().drop_duplicates(
                keep="first", ignore_index=True)
            for idx, item in enumerate(cls.rfHead[2:4]):
                dfc[item[0]] = dfi[dfi[cls.exHead[-1]] == idx].groupby(
                    cls.header).mean().reset_index()[cls.rfHead[1]]
                dfc[item[1]] = dfi[dfi[cls.exHead[-1]] == idx].groupby(
                    cls.header).agg(np.std, ddof=0).reset_index()[
                    cls.rfHead[1]]

            dfi = dfi.filter(cls.header + [cls.rfHead[0]])
            dfc[cls.rfHead[4][0]] = dfi.groupby(
                cls.header).mean().reset_index()[cls.rfHead[0]]
            dfc[cls.rfHead[4][1]] = dfi.groupby(cls.header).agg(
                np.std, ddof=0).reset_index()[cls.rfHead[0]]
            stds = [item[1] for item in cls.rfHead[2:4]]
            dfc[cls.rfHead[-1]] = np.where(
                dfc[stds].max(axis=1) < cls.threshold, 1, 0)
            dfc.to_csv(row("rfcFile"), encoding="utf-8", index=False)

        if cls.stop:
            sys.exit(
                ("Stop and conduct quality control on all of your RF"
                 " centers.csv files"))

    @classmethod
    def filterMappedResponses(cls):
        df_s = dict()
        for row in tqdm(cls.iterRows(), total=cls.len()):
            if not op.isfile(row("rfcFile")) or not row("sCheck"):
                continue

            dfi = pd.read_csv(
                row("rfcFile"), usecols=(cls.header + [cls.rfHead[-1]]))
            dfi = dfi[dfi[cls.rfHead[-1]] > 0]
            if dfi.empty:
                continue

            dfn = pd.merge(pd.read_csv(
                row("mesFile")), dfi, how="inner", on=cls.header)
            if not dfn.empty:
                df_s[row("nmsFile")] = (pd.concat(
                    (df_s[row("nmsFile")], dfn), axis=0, ignore_index=True)
                                        if row("nmsFile") in df_s else dfn)

            for c in row("channels"):
                if op.isfile(row("mapFile", c)):
                    continue

                dfa = pd.merge(pd.read_csv(
                    row("avgFile", c)), dfi, how="inner", on=cls.header)
                if not dfa.empty:
                    df_s[row("mapFile", c)] = (
                        pd.concat((df_s[row("mapFile", c)], dfa), axis=0,
                                  ignore_index=True)
                        if row("mapFile", c) in df_s else dfa)

        for file, df_ in df_s.items():
            if not op.isfile(file):
                df_.to_csv(file, encoding="utf-8", index=False)

    @classmethod
    def plotMappedResponses(cls):
        def makePlot(row, dfs, id):
            fdx = len(dfs[cls.header[0]].unique().tolist())
            rdx = dfs.shape[0]
            adx = dfs[cls.imDims[3]].sum()
            id = "{} Fly N: {}, ROI N: {}, Response N: {}".format(
                id, fdx, rdx, adx)
            dfs = np.squeeze(dfs.filter(items=row("strTime")).to_numpy())
            cen = smooth(dfs if dfs.ndim == 1 else np.mean(dfs, axis=0))
            spr = (ss.sem(dfs, axis=0) if dfs.ndim > 1 else None)
            MultipageTiff.addFigure(
                Y=cen, name=id, yLabel="average", axLabel=("Time (s)", "DF/F"),
                X=row("timing"), dY=spr, lightOn=row("lightOn"))

        for row in tqdm(cls.iterRows(), total=cls.len()):
            for c in row("channels"):
                if not op.isfile(row("mapFile", c)):
                    continue

                dfm = pd.read_csv(row("mapFile", c))
                fig = row("maprFig", c)
                for lay in dfm[cls.header[1]].unique().tolist():
                    save = getPath("-".join((fig, lay)), ext="tif")
                    if op.isfile(save):
                        continue

                    dfs = dfm[dfm[cls.header[1]] == lay].copy()
                    makePlot(row, dfs, "{} Total".format(lay))
                    for name in dfs[cls.header[0]].unique().tolist():
                        dff = dfs[dfs[cls.header[0]] == name].copy()
                        makePlot(row, dff, name)

                    MultipageTiff.saveFigures(save)
