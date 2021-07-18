#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 03:04:49 2021

@author: ike
"""


import os
import csv
import shutil
import string
import os.path as op
from glob import glob as gg


ALLOWED = "".join((string.ascii_letters, string.digits, "_", "-"))


def getPath(*args, ext=None):
    path = op.normpath(op.join(*[
        op.join(*op.join(*arg.split("\\")).split("/")) for arg in args]))    
    path = (".".join([path, ext]) if ext is not None else path)
    return "".join(("/", path))


def getParent(path, num=1):
    for x in range(num):
        if "/" in path:
            path = path[:path.rindex("/")]

    return path


def csvDict(file, classMap):
    with open(file, newline='') as csvFile:
        reader = csv.DictReader(csvFile)
        csvDict = {
            row["File_Name"]: classMap[row["cristae_condition"]]
            for row in reader}
    
    return csvDict


def getPathName(path):
    if "/" in path:
        name = path[path.rindex("/") + 1:]

    return name


def cleanPath(path):
    path = list(str(path))
    for idx in range(len(path)):
        if path[idx] not in ALLOWED:
            path[idx] = "-"

    path = "".join(path)
    return path


def namePathDict(pathList, subString=None):
    if subString is None:
        pathDict = {getPathName(path): path for path in pathList}
    else:
        pathDict = {path.replace(subString, ""): path for path in pathList}
    return pathDict


def dictOverlap(*args):
    keys = None
    for arg in args:
        keys = ([key for key in arg] if keys is None else 
                [key for key in keys if key in arg])
    
    return keys


def changeExt(path, ext=None):
    path = (path[:path.rindex(".")] if "." in path else path)
    if ext is not None:
        path = ".".join([path, ext])

    return path


def csvSave(file, kwargs):
    head = ["File name", "Prediction", "Class Index"]
    with open(file, 'w') as csvFile: 
        writer = csv.DictWriter(csvFile, fieldnames=head)    
        writer.writeheader()
        for i in range(len(kwargs["Name"])):
            writer.writerow(
                {head[0]: kwargs["Name"][i], head[1]: kwargs["Prediction"][i],
                 head[2]: kwargs["Class"][i]})


def makeDirIfEmpty(path):
    if not op.isdir(path):
        os.makedirs(path, exist_ok=False)


def removeParentIfEmpty(path):
    parent = getParent(path)
    if op.isdir(parent) and not os.listdir(parent):
        shutil.rmtree(parent)


def makeParentDirectory(path):
    makeDirIfEmpty(getParent(path))


def movePath(src, dst):
    if all(((type(src) == str), (type(dst) == str),
            (not any(((op.isfile(dst)), (op.isdir(dst))))),
            (op.isfile(src) or op.isdir(src)))):
        makeParentDirectory(dst)
        shutil.move(src=src, dst=dst)


def glob(*args, ext=None):
    path = getPath(*args, ext=ext)
    pathList = sorted(gg(path))
    pathList = (pathList if len(pathList) > 0 else None)
    return pathList


def recursiveGlob(*args, ext=None):
    path = getPath(*args, ext=ext)
    pathList = sorted(gg(path, recursive=True))
    pathList = (pathList if len(pathList) > 0 else None)
    return pathList


def firstGlob(*args, ext=None):
    path = getPath(*args, ext=ext)
    path = (gg(path, recursive=True) if "**" in args else gg(path))
    path = (path[0] if len(path) > 0 else None)
    return path

