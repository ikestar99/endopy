#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 10:29:19 2021

@author: ike
"""
import sys
import shutil
import os.path as op

import readlif.reader as rr

from ..main import tqdm
from ..utils.base import BasePipeline
from ..utils.pathutils import (
    getPath, getPathName, getParent, movePath, glob, recursiveGlob,
    makeDirIfEmpty, changeExt, removeParentIfEmpty)
from ..utils.visualization import wait
from ..utils.multipagetiff import MultipageTiff


class TreeSorter(BasePipeline):
    def __init__(self, pCFG):
        super(TreeSorter, self).__init__(pCFG)

    def __call__(self):
        if self.unpack:
            wait(" \nUnpacking Lifs")
            self.sortLifs()

    @classmethod
    def sortLifs(cls):
        lifDir = getPath(cls.cfg["DataDir"], "Leica Files")
        with tqdm(total=0) as pbar:
            for lifFile in cls.getUnique("leica"):
                lifFileReader = rr.LifFile(lifFile)
                pbar.total += lifFileReader.num_images
                for idx, lifImage in enumerate(lifFileReader.get_iter_image()):
                    pbar.set_description(str(idx), refresh=True)
                    row = cls.getRow(lifFile, (idx + 1))
                    if row is None:
                        pbar.update(1)
                        continue

                    for c in range(lifImage.channels):
                        channel = row("c{}".format(c))
                        if channel is None:
                            channel = "ch{}".format(str(c).zfill(2))

                        savePath = row("cPath", channel=channel)
                        if ((glob("{}*".format(savePath), ext="tif")
                             is not None) or op.isdir(savePath)):
                            continue
                        if lifImage.dims.z == 1 and lifImage.dims.t > 1:
                            pilArray = [p for p in lifImage.get_iter_t(c=c)]
                            MultipageTiff.savePillowArray(pilArray, savePath)
                        elif lifImage.dims.z > 1 and lifImage.dims.t == 1:
                            pilArray = [p for p in lifImage.get_iter_z(c=c)]
                            MultipageTiff.savePillowArray(
                                pilArray, "{} z stack".format(savePath))
                        elif lifImage.dims.z > 1 and lifImage.dims.t > 1:
                            for t in range(lifImage.dims.t):
                                pilArray = [
                                    p for p in lifImage.get_iter_z(t=t, c=c)]
                                MultipageTiff.savePillowArray(
                                    pilArray, getPath(
                                        savePath, "t{} stack".format(t)))

                    for subDir in row("subDirs"):
                        makeDirIfEmpty(subDir)

                    if row("oldStim") is not None:
                        movePath(src=row("oldStim"), dst=row("newStim"))

                    pbar.update(1)

                newFile = getPath(lifDir, getPathName(lifFile))
                movePath(src=lifFile, dst=newFile)

        pbar.close()
        del pbar

        for directory in glob(cls.cfg["DataDir"], "*"):
            if all((
                    (recursiveGlob(directory, "**", "*", ext="tif") is None),
                    (recursiveGlob(directory, "**", "*", ext="lif") is None),
                    (recursiveGlob(directory, "**", "*", ext="csv") is None),
                    (recursiveGlob(directory, "**", "*", ext="pt") is None),
                    ("." not in directory))):
                shutil.rmtree(directory)

        dfs = cls.dfs.drop(columns=cls.lHead1)
        dfs = dfs.dropna(subset=cls.lHead3[:-1], how="any")
        BasePipeline.dfs = dfs
        cls.save(dfs)

    # @classmethod
    # def sortExisting(cls):
    #     dfs = cls.dfs
    #     dfs[cls.lHead2[0]] = [
    #         getPath(getParent(s), c, getPathName(f)) for s, c, f in zip(
    #             dfs[cls.lHead3[0]], dfs[cls.lHead2[0]], dfs[cls.lHead3[3]])]
    #     [movePath(src=o, dst=n) for o, n in zip(
    #         dfs[cls.lHead3[0]], dfs[cls.lHead2[0]])]
    #     dfs[cls.lHead3[0]] = dfs[cls.lHead2[0]]
    #     dfs = dfs.drop(columns=cls.lHead2[0])
    #     dfs = dfs.drop_duplicates(
    #         subset=cls.lHead3[:2], keep="first", ignore_index=True)
    #     dfs.dropna(subset=[cls.lHead3[:-1]])
    #     for row in cls.iterRows():
    #         for channel in row("channels"):
    #             cPath = row("cPath", channel=channel)
    #             if op.isfile(changeExt(cPath, ext="tif")):
    #                 continue
    #
    #             imFiles = recursiveGlob(cPath, "**", ext="tif")
    #             myFiles = glob("{}*".format(cPath), ext="tif")
    #             imFiles = (myFiles if imFiles is None else (
    #                 imFiles if myFiles is None else imFiles + myFiles))
    #             if len(imFiles) == 1:
    #                 movePath(src=imFiles[0], dst=changeExt(cPath, ext="tif"))
    #                 removeParentIfEmpty(imFiles[0])
    #             elif len(imFiles) > 1:
    #                 imStack = MultipageTiff(imFiles)
    #                 imFiles = list(set([getParent(p) for p in imFiles]))
    #                 [shutil.rmtree(p) for p in imFiles]
    #                 imStack.save(changeExt(cPath, ext="tif"))
    #             else:
    #                 sys.exit(
    #                     "No images for {} channel in:\n    {}".format(
    #                         channel, row("iPath")))
    #
    #     cls.save(dfs)
