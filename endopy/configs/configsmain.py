#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 26 18:42:55 2021

@author: ike
"""


from pkg_resources import resource_filename as RFN

from ..utils.configurator import Configurator


CONFIGS = dict(
    # B0 = RFN("endopy.configs", "BoundaryNet0.yaml"),
    B1 = RFN("endopy.configs", "BoundaryNet1.yaml"),
    C0 = RFN("endopy.configs", "CompartmentNet0.yaml"))

STATEDICTS = dict(
    # B0 = RFN("endopy.configs", "BoundaryNet0.pt"),
    B1 = RFN("endopy.configs", "BoundaryNet1.pt"),
    C0 = RFN("endopy.configs", "CompartmentNet0.pt"))


def getFlyCFGs():
    pass
    # flyCFGs = dict()
    # for key, cfgPath in CONFIGS.items():
    #     flyCFGs[key] = Configurator(load=True, cfgPath=cfgPath)
    #     flyCFGs[key]["Pause"] = STATEDICTS[key]
    #
    # return flyCFGs