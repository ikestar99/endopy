#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  7 13:06:21 2021
Core code obtained from Vicky Mak, Barnhart Lab
Class structure and relative imports from Ike Ogbonna, Barnhart Lab
@author: ike
"""

import shutil
import numpy as np
from glob import glob

from .pathutils import getPath, changeExt, makeParentDirectory


def clearEmptyDirectories(base):
    for directory in glob(getPath(base, "*")):
        if "." in directory:
            continue
        else:
            test = glob(getPath(directory, "**", "*.*"), recursive=True)
            if len(test) == 0:
                shutil.rmtree(directory)


def isInt(string):
    try:
        int(string)
        return True
    except ValueError:
        return False


def boundInt(num, low, up):
    num = min((max(num, low)), up)
    return num


def nextFactor(num, div):
    rem = div - (num % div)
    num = (0 if rem == div else rem) + num
    return num
                

def smooth(x, window_len=5, window="hanning"):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: input signal 
        window_len: dimension of the smoothing window; should be an odd integer
        window: type of window from "flat", "hanning", "hamming", "bartlett",
        "blackman"
            flat window will produce a moving average smoothing.

    output:
        smoothed signal
        
    example:
    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    np.hanning, np.hamming, np.bartlett, np.blackman, np.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead
        of a string
    NOTE: length(output) != length(input)
        to correct: return y[(window_len/2-1):-(window_len/2)] instead of y.
    """

    if x.ndim != 1:
        raise (ValueError, "smooth only accepts 1 dimension arrays.")
    elif x.size < window_len:
        raise (ValueError, "Input vector needs to be bigger than window size.")
    elif window not in ("flat", "hanning", "hamming", "bartlett", "blackman"):
        raise (ValueError,
               "Window is 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
    if window_len < 3:
        return x

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-1 - window_len:-1]]  # print(len(s))
    if window == "flat":  # moving average
        w = np.ones(window_len, "d")
    else:
        w = eval("np.{}(window_len)".format(window))

    y = np.convolve(w / w.sum(), s, mode="valid")[2:-2]
    return y


def upsample(x, t, up):
    x_up = x
    t_up = t
    iters = 0
    while iters < up:
        x_new = list()
        x_new.append(x_up[:-1])
        x_new.append(x_up[1:])
        x_up.extend(list(np.mean(np.asarray(x_new), axis=0)))

        t_new = list()
        t_new.append(t_up[:-1])
        t_new.append(t_up[1:])
        t_up.extend(list(np.mean(np.asarray(t_new), axis=0)))

        A = np.zeros((len(t_up), 2))
        A[:, 0] = t_up
        A[:, 1] = x_up

        AS = A[A[:, 0].argsort()]
        t_up = list(AS[:, 0])
        x_up = list(AS[:, 1])
        iters += 1
    return t_up, x_up


def downsample(x, t, bin_size, first_bin, last_bin):
    bin_centers = np.arange(first_bin, last_bin + bin_size, bin_size)
    A = np.zeros((len(t), 2))
    A[:, 0] = t
    A[:, 1] = x
    AS = A[A[:, 0].argsort()]
    ds_t = []
    ds_x = []
    for b in bin_centers:
        bi1 = np.searchsorted(AS[:, 0], b - bin_size / 2.)
        bi2 = np.searchsorted(AS[:, 0], b + bin_size / 2.)
        ds_t.extend([np.mean(AS[bi1:bi2, 0])])
        ds_x.extend([np.mean(AS[bi1:bi2, 1])])
    return ds_t, ds_x


def resample(x, t, up, bin_size, first_bin, last_bin):
    t_up, x_up = upsample(x, t, up)
    ds_t, ds_x = downsample(x_up, t_up, bin_size, first_bin, last_bin)
    return ds_t, ds_x
