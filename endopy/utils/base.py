#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 25 11:41:02 2021

@author: ike
"""


import os
import sys
import numpy as np
import pandas as pd
import os.path as op

from torch.utils.data import Dataset

from .pipeutils import nextFactor
from .pathutils import (
    getPath, cleanPath, recursiveGlob, firstGlob, getParent, changeExt,
    getPathName, makeParentDirectory, glob)


SCALE = 16
      

class BaseDataset(Dataset):  
    def __init__(self, cfg, masks=True):            
        self.cfg = cfg
        self.samples = None; self.masks = masks
        self.num = (cfg["OutputType"] == "classification")
    
    def __len__(self):
        if self.samples is None:
            return 0

        return self.samples.shape[0]
    
    def adjustKeys(self, keys):
        temp = -2
        while ((len(keys) % self.cfg["SampleSize"] != 0) or
               (len(keys) % self.cfg["BatchSize"] != 0)):
            temp = (-2 if (temp * -1 > len(keys)) else temp)
            keys += [keys[temp]]; temp -= 2
            
        return keys

    # def getImage(self, *args, color=None):
    #     def getPillow(f):
    #         image = (
    #             PIM.fromarray(np.load(f)) if ".np" in f else PIM.open(f))
    #         return image
    #
    #     given = ("L" if self.cfg["ImageType"] == "grayscale" else "RGB")
    #     color = (given if color is None else color)
    #     images = [getPillow(file).convert(color) for file in args]
    #     width, height = images[0].size
    #     dy = self.cfg["Hin"] - height; dx = self.cfg["Hin"] - width
    #     images = (
    #         [pad(image, padding=(0, 0, dx, dy), padding_mode="reflect")
    #          for image in images] if (dy != 0 or dx != 0) else images)
    #     images = [np.asarray(image) for image in images]
    #     image = (
    #         images[0] if len(images) == 1 else np.mean(
    #             np.array(images), axis=0))
    #     image = (np.moveaxis(image, -1, 0) if color == "RGB" else image)
    #     return image
        
    def __getitem__(self, idx):        
        pass

    
class BasePipeline(object):
    class Row(object):
        dataDir = None
        lHead1 = ["Leica .lif File Name", "Sample ID"]
        lHead2 = ["Cell type", "Imaging Date", "Fly #", "Z-plane"]
        lHead3 = [
            "SampleFolder", "Stimulus", "Frame rate",
            "Stimulus file timestamp", "Stimulus type", "Epoch length",
            "ch00 indicator", "ch01 indicator"]
        subDir = [
            "stim_files", "masks", "measurements", "plots", "predictions",
            "projections"]
        imDims = ["Hin", "Win", "frames", "N"]
        bnStep = 2  # old 0.1, new 2 * (1 / frame rate)

        def __init__(self, dfr):
            self.dfr = dfr

        def __call__(self, mode, channel=None):
            value = None
            try:
                if mode == "c0":
                    value = self[3, -2]
                elif mode == "c1":
                    value = self[3, -1]
                elif mode == "sample":
                    value = self[3, 0]
                elif mode == "stimulus":
                    value = self[3, 1]
                elif mode == "oldStim":
                    value = self[3, 3]
                elif mode == "frames":
                    value = self[5, 2]
                elif mode == "frameT":
                    value = self[3, 2]
                elif mode == "eLength":
                    value = self[3, 5]

                elif mode == "stimNum":
                    value = self[3, 1][-4:]
                elif mode == "stimName":
                    value = self[3, 1][:-17]
                elif mode == "iPath":
                    value = getPath(self[3, 0], self[3, 1])
                elif mode == "cPath":
                    value = getPath(self[3, 0], self[3, 1], channel)
                elif mode == "lightOn":
                    value = self("eLength") // 4
                    value = (value, value * 3)
                elif mode == "frameOn":
                    value = self("eFrames") // 4

                elif mode == "channels":
                    value = [self("c0"), self("c1")]
                    value = [c for c in value if c is not None]
                elif mode == "subDirs":
                    value = [getPath(self[3, 0], s) for s in self.subDir]
                elif mode == "mskDirs":
                    value = sorted(
                        glob(getPath(self("maskDir"), "*", ext="tif")))
                    value = [m for m in value if m != self("bkgTif")]

                elif mode == "eFrames":
                    value = int(self("eLength") // self[3, 2])
                elif mode == "binT":
                    value = self("frameT") * self.bnStep
                elif mode == "timing":
                    value = np.arange(0, self("eLength"), self("frameT"))[
                        :int(self("eFrames"))]
                elif mode == "binTime":
                    value = self("timing")[::self.bnStep]
                elif mode == "strTime":
                    func = "{:." + str(len(str(self("frameT")))) + "f}"
                    value = [
                        func.format(t).rstrip("0").rstrip(".") for t in
                        self("timing")]
                elif mode == "sCheck":
                    value = self[3, 4] == "on/off"
                elif mode == "rCheck":
                    value = "light" in self[3, 1]
                elif mode == "offset":
                    value = self("eLength") / 2
                elif mode == "shape":
                    value = self[5, 0], self[5, 1]
                elif mode == "bins":
                    return 4

                elif mode == "predDir":
                    value = getPath(self[3, 0], self[4, 4])
                elif mode == "maskDir":
                    value = getPath(self[3, 0], self[4, 1])
                elif mode == "plotDir":
                    value = getPath(self[3, 0], self[4, 3])

                elif mode == "newStim":
                    value = getPath(
                        self[3, 0], self[4, 0], getPathName(self[3, 3]))
                elif mode == "frmStim":
                    value = changeExt(
                        "-".join(
                            (changeExt(self("newStim")), "frames")), ext="csv")

                elif mode == "mesFile":
                    value = getPath(
                        self[3, 0], self[4, 2], "-".join(
                            (self[3, 1], "measurements")), ext="csv")
                elif mode == "rawFile":
                    value = getPath(
                        self[3, 0], self[4, 2], "-".join(
                            (self[3, 1], "raw responses", channel)),
                        ext="csv")
                elif mode == "indFile":
                    value = getPath(
                        self[3, 0], self[4, 2], "-".join(
                            (self[3, 1], "individual responses", channel)),
                        ext="csv")
                elif mode == "rfcFile":
                    value = getPath(
                        self[3, 0], self[4, 2], "RF centers", ext="csv")
                elif mode == "avgFile":
                    value = getPath(
                        self[3, 0], self[4, 2], "-".join(
                            (self[3, 1], "average responses", channel)),
                        ext="csv")
                elif mode == "bnaFile":
                    value = getPath(
                        self[3, 0], self[4, 2], "-".join(
                            (self[3, 1], "binned responses", channel)),
                        ext="csv")
                elif mode == "mapFile":
                    path = getPath(
                        self.dataDir, self[4, 2], "-".join(
                            (self("stimName"), "mapped", channel)),
                        ext="csv")
                    makeParentDirectory(path)
                    value = path
                elif mode == "nmsFile":
                    path = getPath(
                        self.dataDir, self[4, 2], "-".join(
                            (self("stimName"), "measurements")), ext="csv")
                    makeParentDirectory(path)
                    value = path

                elif mode == "binTif":
                    value = changeExt("-".join(
                        (changeExt(self("cPath", channel)), "binned")),
                        ext="tif")
                elif mode == "avpTif":
                    value = getPath(
                        self[3, 0], self[4, 5], "average", ext="tif")
                elif mode == "bkgTif":
                    value = getPath(self("maskDir"), "background", ext="tif")

                elif mode == "rawrFig":
                    value = getPath(
                        self("plotDir"), "-".join(
                            (self("stimulus"), "ROI raw", channel)),
                        ext="tif")
                elif mode == "measFig":
                    value = getPath(
                        self("plotDir"), "-".join(
                            (self("stimulus"), "ROI average", channel)),
                        ext="tif")
                elif mode == "bnavFig":
                    value = getPath(
                        self("plotDir"), "-".join(
                            (self("stimulus"), "binned ROI average", channel)),
                        ext="tif")
                elif mode == "cntfFig":
                    value = getPath(self("plotDir"), "AIN4", ext="tif")
                elif mode == "maprFig":
                    value = getPath(
                        self.dataDir, self[4, 3], "-".join(
                            (self("stimName"), "mapped responses", channel)))
                    makeParentDirectory(value)
            except KeyError:
                pass

            return value

        def __getitem__(self, item):
            if item[0] == 1:
                return self.dfr[self.lHead1[item[1]]]
            elif item[0] == 2:
                return self.dfr[self.lHead2[item[1]]]
            elif item[0] == 3:
                return self.dfr[self.lHead3[item[1]]]
            elif item[0] == 4:
                return self.subDir[item[1]]
            elif item[0] == 5:
                return self.dfr[self.imDims[item[1]]]
            else:
                raise KeyError

    cfg = None
    dfs = None
    bCSV = "*Barnhart Lab Imaging Settings*"
    sCSV = "Sorted directories"
    lHead1 = Row.lHead1
    lHead2 = Row.lHead2
    lHead3 = Row.lHead3
    imDims = Row.imDims
    unpack = False
    sTypes = {"on/off": ["flash", "PD"], "multiple": ["moving"]}
    sNames = ["2s", "10s", "moving", "PD"]
    header = ["sample", "layer", "ROI"]
    rrHead = ["global_time", "rel_time", imDims[2], "epoch_number"]
    exHead = ["size", "response_number", "epoch_label"]
    rfHead = [
        "rmax", "pmax", ("x", "x_std"), ("y", "y_STD"),
        ("amplitude", "amplitude_std"), "mappable?"]
    colors = ("#F1160C", "#0AC523", "#0F37FF", "#8F02C5")
    months = (
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")

    def __init__(self, cfg=None):
        if BasePipeline.cfg is None:
            BasePipeline.cfg = cfg
            BasePipeline.Row.dataDir = cfg["DataDir"]
            BasePipeline.findTree()

    @classmethod
    def len(cls):
        return cls.dfs.shape[0]

    @classmethod
    def getRow(cls, path, id):
        if ".lif" in path:
            columns = (cls.lHead1[0], cls.lHead1[1])
        else:
            columns = (cls.lHead3[0], cls.lHead3[1])
        row = cls.dfs[
            (cls.dfs[columns[0]] == path) &
            (cls.dfs[columns[1]] == id)]
        if row.empty:
            return None
        else:
            row = row.to_dict(orient="records")[0]
            return cls.Row(row)

    @classmethod
    def iterRows(cls):
        for path, ids in cls.getUnique("idList"):
            for id in ids:
                yield cls.getRow(path, id)

    @classmethod
    def getUnique(cls, mode):
        if mode == "leica":
            return cls.dfs[cls.lHead1[0]].unique().tolist()
        elif mode == "sampleFolders":
            return cls.dfs[cls.lHead3[0]].unique().tolist()
        elif mode == "idList":
            dfi = list()
            for s in cls.dfs[cls.lHead3[0]].unique().tolist():
                i = cls.dfs[
                    cls.dfs[cls.lHead3[0]] == s][cls.lHead3[1]].tolist()
                dfi += [(s, i)]

            return dfi
        else:
            return None

    @classmethod
    def save(cls, dfs):
        sorPath = getPath(cls.cfg["DataDir"], cls.sCSV, ext="csv")
        BasePipeline.dfs = dfs
        cls.dfs.to_csv(sorPath, encoding="utf-8", index=False)

    @classmethod
    def findTree(cls):
        def getDate(date):
            date = [d.zfill(2) for d in date.split("/")]
            date[-1] = (
                "20{}".format(date[-1]) if len(date[-1]) == 2 else date[-1])
            return "/".join(date)

        def getStimFile(date, tStamp, stimType):
            if tStamp in (None, ""):
                return None

            tStamp = str(int(tStamp)).zfill(4)
            date = date.split("/")
            date = [date[-1], cls.months[int(date[0]) - 1], date[1], tStamp]
            date[2] = str(date[2]).zfill(2)
            date = "_".join([str(d) for d in date])
            stimFile = recursiveGlob(
                cls.cfg["DataDir"], "**", "*{}".format(date), ext="csv")
            if type(stimFile) == list:
                if len(stimFile) > 1:
                    exp = cls.sTypes[stimType]
                    stimFile = [s for e in exp for s in stimFile if e in s]

                stimFile = stimFile[0]

            return stimFile

        # def getIdentifier(stimFile):
        #     for id in os.listdir(getParent(stimFile, num=2)):
        #         for sName in cls.sNames:
        #             if all(((sName in stimFile), (sName in id),
        #                     op.isdir(id))):
        #                 return id
        #
        #     return None

        # def getChannels(sampFolder, id):
        #     if id is not None:
        #         channels = sorted(os.listdir(getPath(sampFolder, id)))
        #         channels = [changeExt(c) for c in channels if "bin" not in c]
        #         channels = [c for c in channels if len(c) == len(cleanPath(c))]
        #         channels = (channels + [None, None])[:2]
        #         return channels

        logs = recursiveGlob(cls.cfg["DataDir"], "**", "*", ext="log")
        if logs is not None:
            [os.remove(log) for log in logs]

        csvPath = firstGlob(cls.cfg["DataDir"], "**", cls.bCSV, ext="csv")
        sorPath = firstGlob(cls.cfg["DataDir"], "**", cls.sCSV, ext="csv")
        if (csvPath is not None) and (sorPath is not None):
            cls.dfs = pd.read_csv(
                sorPath, sep=r'\s*,\s*', header=0, encoding='ascii',
                engine='python', skipinitialspace=True)

        elif csvPath is not None:
            dfs = pd.read_csv(
                csvPath, usecols=(cls.lHead1 + cls.lHead2 + cls.lHead3[2:]),
                sep=r'\s*,\s*', header=0, encoding='ascii', engine='python',
                skipinitialspace=True, skip_blank_lines=True)
            dfs[cls.lHead3[3:6] + [cls.lHead3[-1]]] = dfs[
                cls.lHead3[3:6] + [cls.lHead3[-1]]].fillna("")
            if dfs.isna().sum().sum() > 0:
                sys.exit("Complete all relevant columns in:\n    {}".format(
                    csvPath))

            dfs[cls.lHead1[0]] = dfs[cls.lHead1[0]].apply(lambda x: firstGlob(
                cls.cfg["DataDir"], "**", changeExt(x), ext="lif"))
            dfs = dfs.dropna(subset=[cls.lHead1[0]])
            dfs[cls.lHead1[1]] = dfs[cls.lHead1[1]].astype(str).astype(int)
            dfs[cls.lHead3[2]] = (
                dfs[cls.lHead3[2]].astype(str).astype(float).rdiv(1))
            dfs[cls.lHead3[5]] = dfs[cls.lHead3[5]].replace("", 0).astype(int)
            dfs[cls.lHead2[1]] = dfs[cls.lHead2[1]].apply(lambda x: getDate(x))
            dfs[cls.lHead3[0]] = [
                getPath(cls.cfg["DataDir"], cleanPath(
                    "{}-Fly{}-z{}-{}".format(d, f, b, c)))
                for c, d, f, b in zip(
                    dfs[cls.lHead2[0]], dfs[cls.lHead2[1]], dfs[cls.lHead2[2]],
                    dfs[cls.lHead2[3]])]
            dfs[cls.lHead3[3]] = [getStimFile(d, s, t) for d, s, t in zip(
                dfs[cls.lHead2[1]], dfs[cls.lHead3[3]], dfs[cls.lHead3[4]])]
            dfs = dfs.drop(columns=cls.lHead2)
            dfs[cls.lHead3[1]] = [
                changeExt(getPathName(s)) if s is not None else
                "-".join((changeExt(getPathName(l)), str(i))) for s, l, i in
                zip(dfs[cls.lHead3[3]], dfs[cls.lHead1[0]], dfs[cls.lHead1[1]])]
            dfs = dfs.replace({"": None})
            dfs = dfs.drop_duplicates(
                subset=cls.lHead1, keep="first", ignore_index=True)
            cls.dfs = dfs.drop_duplicates(
                subset=cls.lHead3[:2], keep="first", ignore_index=True)
            cls.unpack = True

        elif sorPath is not None:
            dfs = pd.read_csv(
                sorPath, sep=r'\s*,\s*', header=0, encoding='ascii',
                engine='python', skipinitialspace=True, skip_blank_lines=True)
            dfs[cls.lHead3[3:6] + [cls.lHead3[-1]]] = dfs[
                cls.lHead3[3:6] + [cls.lHead3[-1]]].fillna("")
            if dfs.isna().sum().sum() > 0:
                sys.exit("Complete all relevant columns in:\n    {}".format(
                    sorPath))

            srt = dfs[cls.lHead3[0]].apply(lambda x: firstGlob(x))
            if srt.isna().sum() > 0:
                sys.exit("Ensure all paths in 'SampleFolder' column are valid")

            dfs = dfs.drop(columns=dfs.columns[-1])
            dfs[cls.lHead3[2]] = (
                dfs[cls.lHead3[2]].astype(str).astype(float).rdiv(1))
            dfs[cls.lHead3[5]] = dfs[cls.lHead3[5]].replace("", 0).astype(int)
            dfs = dfs.replace({"": None})
            cls.dfs = dfs.drop_duplicates(
                subset=cls.lHead3[:2], keep="first", ignore_index=True)
            cls.unpack = False

        else:
            dfs = pd.DataFrame(columns=cls.lHead3)
            stimFiles = recursiveGlob(
                cls.cfg["DataDir"], "**", "*[0-9][0-9][0-9][0-9]", ext="csv")
            if stimFiles is None:
                sys.exit("Sort your stimulus.csv files")

            dfs[cls.lHead3[3]] = stimFiles
            dfs[cls.lHead3[0]] = dfs[cls.lHead3[3]].apply(
                lambda x: getParent(x, num=2))
            dfs = dfs.fillna("")
            sorPath = getPath(cls.cfg["DataDir"], cls.sCSV, ext="csv")
            dfs.to_csv(sorPath, encoding="utf-8", index=False)
            sys.exit("Complete all columns in:\n    {}".format(sorPath))

    @classmethod
    def inVivo(cls):
        return not cls.dfs.empty


class BaseMachine(object):
    class Parameters(object):
        header = [
            "Name", "Date", "Subject description", "Version", "Downsampling",
            "Image type", "Input", "Output", "Sample size", "Depth",
            "Batch size", "Classes", "Epochs"]
        custom = ["Din", "Dout", "Cin", "dataset", "weights"]
        cPaths = ["DataDir"]
        preblt = dict(LR=0.001, Gamma=1.0)

        def __init__(self, dfm):
            self.dfm = dfm

        def __call__(self, mode):
            value = None
            try:
                if mode == "name":
                    value = self[0, 0]
                elif mode == "model":
                    value = ("R" if self[1, 0] == 2 else "E")
                elif mode == "reduce":
                    value = self[0, 4]
                elif mode == "vers":
                    value = self[0, 3]
                elif mode == "Din":
                    value = self[1, 0]
                elif mode == "Dout":
                    value = self[1, 1]
                elif mode == "Cin":
                    value = self[1, 2]
                elif mode == "Cout":
                    value = self[0, 11]
                elif mode == "depth":
                    value = self[0, 9]
                elif mode == "sampleSize":
                    value = self[0, 8]
                elif mode == "dataset":
                    value = ("M" if self("Din") == self("Dout") else "F")
                elif mode == "LR":
                    value = self[3, 0]
                elif mode == "Gamma":
                    value = self[3, 1]

                elif mode == "mSave":
                    value = getPath(
                        self[2, 0], self("name"), "checkpoint", ext="pt")
                    makeParentDirectory(value)
                elif mode == "sSave":
                    value = getPath(
                        self[2, 0], self("name"), "statistics", ext="csv")
                    makeParentDirectory(value)
                elif mode == "fSave":
                    value = getPath(
                        self[2, 0], self("name"), "figures", ext="tif")
                    makeParentDirectory(value)
            except KeyError:
                pass

            return value

        def __getitem__(self, item):
            if item[0] == 0:
                return self.dfm[self.header[item[1]]]
            elif item[0] == 1:
                return self.dfm[self.custom[item[1]]]
            elif item[0] == 2:
                return self.dfm[self.cPaths[item[1]]]
            elif item[0] == 3:
                return self.dfm[self.preblt[item[1]]]
            else:
                raise KeyError

    cfg = None
    dfm = None
    dfd = None
    bCSV = "*Barnhart Lab Machine Learning Query*"
    dCSV = "*Barnhart Lab Machine Learning Data Directories*"
    scale = 16
    header = Parameters.header
    custom = Parameters.custom
    direct = ["Images", "Masks"]

    def __init__(self, cfg=None):
        if BaseMachine.cfg is None:
            BaseMachine.cfg = cfg
            BaseMachine.findModel()

    @classmethod
    def getParameters(cls):
        return cls.Parameters(cls.dfm)

    @classmethod
    def getUnique(cls, mode):
        if mode == "train":
            return [p for p in cls.dfd if None not in p]
        elif mode == "predict":
            return [p[0] for p in cls.dfd]
        else:
            return None

    @classmethod
    def findModel(cls):
        csvPath = firstGlob(cls.cfg["DataDir"], "**", cls.bCSV, ext="csv")
        datPath = firstGlob(cls.cfg["DataDir"], "**", cls.dCSV, ext="csv")
        if (csvPath is not None) and (datPath is not None):
            dfm = pd.read_csv(
                csvPath, sep=r'\s*,\s*', header=0, encoding='ascii',
                engine='python', skipinitialspace=True).fillna("")
            dfm = dfm.to_dict(orient="records")[0]
            dfm[cls.header[0]] = cleanPath("-".join(
                (dfm[cls.header[0]], dfm[cls.header[1]], dfm[cls.header[2]])))
            [dfm.pop(key) for key in cls.header[1:3]]
            dfm[cls.custom[0]] = (3 if dfm[cls.header[6]] == "ZHW" else 2)
            dfm[cls.custom[1]] = (3 if dfm[cls.header[7]] == "ZHW" else 2)
            dfm[cls.header[7]] = min(dfm[cls.header[6]], dfm[cls.header[7]])
            if dfm[cls.custom[0]] == 3:
                dfm[cls.header[9]] = nextFactor(dfm[cls.header[9]], cls.scale)
                dfm[cls.header[8]] = nextFactor(
                    dfm[cls.header[8]], dfm[cls.header[9]])
                dfm[cls.custom[2]] = 1
            else:
                dfm[cls.header[9]] = 1
                dfm[cls.custom[2]] = dfm[cls.header[8]]

            if dfm[cls.header[5]] == "RGB":
                dfm[cls.header[9]] = 1
                dfm[cls.header[8]] = 1
                dfm[cls.custom[2]] = 3

            cls.dfm = dfm.update(cls.cfg.Paths)
            dfd = pd.read_csv(
                datPath, sep=r'\s*,\s*', header=0, encoding='ascii',
                engine='python', skipinitialspace=True).fillna("")
            dfd[cls.direct[0]] = dfd[cls.direct[0]].apply(
                lambda x: firstGlob(x) if x is not None else None)
            dfd[cls.direct[1]] = dfd[cls.direct[1]].apply(
                lambda x: firstGlob(x) if x is not None else None)
            dfd = dfd.dropna(subset=[cls.direct[0]])
            dfd.replace({"": None})
            dfd = [(i, m) for i, m in zip(
                dfd[cls.direct[0]], dfd[cls.direct[1]])]
            cls.dfd = dfd

        else:
            sys.exit("Create a 'Barnhart Lab Machine Learning Query.csv' file")
