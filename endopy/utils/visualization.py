#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 03:38:05 2021

@author: ike
"""


import time
import numpy as np
import matplotlib.pyplot as plt


COLORS = ("#FF0000", "#FF8000", "#FFFF00", "00FF00", "00FFFF", "0000FF",
          "7f00ff", "FF00FF", "FF00F7", "808080")
SPACE = "    "
TRUNK = "│   "
SPLIT = "├── "
FINAL = "└── "


def plotIterations(bSize, save, **kwargs):
    iters = (kwargs["Train"].shape[0] if "Train" in kwargs else
             kwargs["Test"].shape[0])
    alpha = 1.0
    for key, array in kwargs.items():
        labels = (
            ["{} Class {}".format(key[:2], x) for x in range(
                array.shape[1] - 2)] + ["{} Total".format(key[:2])])
        for i in range(array.shape[1] - 1):
            # color = i % len(COLORS)
            xIters = np.linspace(0, iters, array.shape[0])      
            plt.plot(xIters, array[:,i], linewidth=2, #color=COLORS[color],
                     alpha=alpha, label=labels[i])
        
        alpha -= 0.2
            
    plt.legend()
    plt.title("Classwise and/or Total Accuracies")
    plt.legend(loc="upper left")
    plt.xlabel("Iterations x {}".format(bSize)) 
    plt.ylabel("Accuracy")
    plt.xlim(left=0)
    plt.ylim(bottom=0, top=1)
    plt.savefig(save, dpi=300)
    plt.show()
    

def wait(message, multiplier=0.25):
    print(message)
    time.sleep((1.0 * multiplier))
    
    
def getInput(message=""):
    wait(message)
    value = input(">> ")
    return value


def getDirectoryTree(dirDict):
    def listDirs(dictionary, array=None, depth=1):
        array = (
            np.array([[".    "]]).astype('U100') if array is None else array)
        for i, folder in enumerate(dictionary):
            array = np.concatenate((array, array[-1][np.newaxis]), axis=0)
            array[-1] = SPACE
            while depth + 1 > array.shape[-1]:
                array = np.concatenate(
                    (array, array[:,-1][:,np.newaxis]), axis=-1)
                array[:,-1] = SPACE
                
            array[-1,depth] = folder
            if i + 1 == len(dictionary):
                array[-1,depth - 1] = FINAL
                
            if type(dictionary[folder]) is dict:
                array = listDirs(
                    dictionary[folder], array=array, depth=depth + 1)
        
        return array
    
    def rowToString(rowArray):
        string = "".join([rowArray[x] for x in range(rowArray.size)])
        while string[-1] == " ":
            string = string[:-1]
        
        return string
    
    array = listDirs(dictionary=dirDict)
    count = 0
    while True:
        reference = (
            (array == FINAL).astype(int) + (array == TRUNK).astype(int) +
            (array == SPLIT).astype(int))
        check = np.argwhere(reference)
        for x in range(check.shape[0]):
            if array[check[x,0] - 1, check[x,1]] == SPACE:
                count += 1
                array[check[x,0] - 1, check[x,1]] = (
                    TRUNK if array[check[x,0] - 1, check[x,1] + 1] in
                    (SPACE, FINAL, TRUNK, SPLIT) else SPLIT)
        
        if count == 0:
            break
        else:
            count = 0
    
    string = "\n".join([rowToString(array[x]) for x in range(array.shape[0])])
    return string
