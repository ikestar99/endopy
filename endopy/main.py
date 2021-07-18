#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 16:02:54 2021

@author: ike
"""


import sys
if 'google.colab' in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm
    
from .pipeline.pipelinemain import (
    processLeicaData, trainMachineLearning)  # , useMachineLearning)
from .utils.menu import Menu


MAINMENU = Menu("Root Menu -- I want to...")
MAINMENU["Access leica raw data pipeline"] = [processLeicaData]
MAINMENU["Train a machine learning model"] = [trainMachineLearning]
# MAINMENU["Use a trained machine learning model"] = [useMachineLearning]


def userInput():
    while True:
        MAINMENU()