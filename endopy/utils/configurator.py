#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 21:28:00 2020


@author: ike
"""

import os
import sys
import os.path as op

from IPython.display import clear_output

from .menu import Menu
from .visualization import wait, getDirectoryTree


MACHINETREE = dict(
    Data_Directory=dict(
        first_recording=dict(
            image_directory=dict(images_tif=""),
            mask_directory=dict(mask_tif="")),
        second_Recording=dict(etc="")))
PIPELINETREE = dict(
    Unsorted_Data_Directory={
        "06_10_2021_fly2.lif": "",
        "06_10_2021_fly3.lif": "",
        "full_field_flashes-2s_2021_Jun_10_1656.csv": "",
        "Barnhart Lab Imaging Settings.csv": ""})


class Configurator(object):
    def __init__(self, mode):
        self.Paths = dict(DataDir=None)
        self.DirStructure = (MACHINETREE if mode == "M" else PIPELINETREE)
        self.getPaths()

    def __str__(self):
        def dictAsString(string, dictionary, divider):
            for key, item in dictionary.items():
                if type(item) is dict and key not in (
                        "DirStructure", "Notification"):
                    string = "\n".join((string, divider, key))
                    for k, i in item.items():
                        if i is not None:
                            if type(i) == str:
                                i = (i[i.rindex("/") + 1:]
                                     if (i.count("/") > 2) else i)
                            string = "\n".join(
                                (string, "    {}: {}".format(k, i)))

            return string

        divider1, divider2 = Menu.divider1, Menu.divider2
        string = "\n".join((divider1, "Current configuration"))
        string = dictAsString(string, self.__dict__, divider2)

        if self.DirStructure is not None:
            string = "\n".join(
                (string, divider2,
                 "Your data should follow this directory structure",
                 getDirectoryTree(self.DirStructure)))

        string = "\n".join((string, divider1))
        return string

    def __call__(self, multiplier=1.0):
        wait(self.__str__(), multiplier)

    def __getitem__(self, key):
        for attr, item in self.__dict__.items():
            if attr != "DirStructure":
                if key in item:
                    return self.__dict__[attr][key]

        raise KeyError()

    def __setitem__(self, key, value):
        for attr, item in self.__dict__.items():
            if attr != "DirStructure":
                if key in item:
                    self.__dict__[attr][key] = value
                    return

    def __delattr__(self, name):
        values = {key: None for key in getattr(self, name)}
        setattr(self, name, values)

    def __contains__(self, key):
        try:
            return self[key] == self[key]
        except KeyError:
            return False

    def choice(self, key, message, args):
        self[key] = Menu.choiceFunctional(
            message=message, args=args, extraMessage=self.__str__())

    def entry(self, key, message, casting, path=False, norm=None):
        os.system('cls' if os.name == 'nt' else 'clear')
        clear_output()
        wait(self.__str__())
        self[key] = Menu.entryFunctional(
            message=message, casting=casting, path=path)
        if key == "DataDir" and not op.isdir(self["DataDir"]):
            self.entry(key, message, casting, path=True)
        elif norm is not None:
            self[key] = self[key][(self[key].find("/", (len(norm) + 1)) + 1):]

    def getPaths(self):
        self.entry(
            "DataDir", "Input path to your data folder", str, path=True)
        assert op.isdir(self["DataDir"]), "Endopy could not find {}".format(
            self["DataDir"])

    def proceed(self, stage):
        response = Menu.choiceFunctional(
            "Are you ready to :    {}?".format(stage), ("Yes", "Hold on"),
            extraMessage=self.__str__())
        if response != "Yes":
            sys.exit("endopy closed successfully, return when you are ready")