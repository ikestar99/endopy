#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 01:32:55 2020

@author: ike
"""


import os
import sys

from IPython.display import clear_output

from .pathutils import getPath
from .visualization import wait, getInput


class Menu(object):
    class Item(object):
        def __init__(self, parent, option, function=None, argList=[]):
            self.parent = parent
            self.option = option
            self.function = function
            self.argList = argList
        
        def __call__(self):
            if (self.option == "-1"):
                self.parent.parent()
            elif self.option == "-2":
                os.system('cls' if os.name == 'nt' else 'clear')
                clear_output()
                sys.exit("Endopy closed successfully")
            elif self.function is not None:
                temp = None
                if type(self.function) is type:
                    while True:
                        try:
                            temp = self.function(getInput("Value"))
                            return temp
                        except (TypeError, ValueError):
                            wait("Please try gain")
                    
                self.function(*self.argList)
        
        def update(self, function, argList=[]):
            self.function = function
            self.argList = argList

    divider1 = "=" * 100
    divider2 = "-" * 100

    def __init__(self, name, parent=None):
        self.name = name
        self.parent = parent
        self.options = {}
        self.extra = {"Back": Menu.Item(self, "-1"),
                      "Exit": Menu.Item(self, "-2")}
        self.mapping = []

    @staticmethod
    def choiceFunctional(message, args, extraMessage="\n  "):
        optMenu = Menu(message)
        for option in args:
            optMenu[option] = [None]

        choice, _ = optMenu(getName=True, message=extraMessage)
        return choice

    @staticmethod
    def entryFunctional(message, casting, path=False):
        wait("Enter an integer or a path from a file browser \n")
        while True:
            try:
                value = casting(getInput(message))
                return (getPath(value) if path else value)
            except (ValueError, TypeError):
                print("Enter a valid integer, string, or path")

    def __str__(self):
        def getString(itn, maxt):
            string = "".join(["    " for x in range(maxt - itn[1])])
            string += ("{}: ".format(itn[0] + 1) if itn[1] == 0 else "--")
            string = "\n{}{}".format(string, itn[2])
            return string

        maxt = self.mapping[0][1]
        string = "\n".join((self.divider1, self.name, self.divider2))
        for x in range(len(self.mapping)):
            string += getString(self.mapping[x], maxt)
            if (self.mapping[x][1] == 0 and
                (x + 1 == len(self.mapping) or self.mapping[x + 1][1] != 0)):
                if self.parent is not None:
                    string += getString((-2, 0, "Back"), maxt)
                
                string += getString((-3, 0, "Exit"), maxt)
            
        string = "\n".join(
            (string, self.divider1,
             ("\n        <<<Select an integer and press enter>>>        ")))
        return string
         
    def __call__(self, getName=False, message=""):
        os.system('cls' if os.name == 'nt' else 'clear')
        clear_output()
        self.update()
        wait(message); wait(self)
        while True:
            idx = None
            try:
                idx = int(getInput())
                idx = (idx - 1 if idx > 0 else idx)
                if 0 <= idx < len(self.options) or idx in [-1, -2]:
                    if not (self.parent is None and idx == -1):
                        break
            except ValueError:
                pass
            
            wait("Please select a valid menu option")
        
        if idx >= 0:
            keys = sorted([key for key in self.options])
            if getName is True:
                return (self.options[keys[idx]].option,
                        self.options[keys[idx]]())
            else:
                self.options[keys[idx]]()    
        elif idx == -1:
            self.extra["Back"]()      
        elif idx == -2:
            self.extra["Exit"]()   
    
    def __getitem__(self, key):
        if key not in self.options:
            raise KeyError
        return self.options[key]
    
    def __setitem__(self, key, value):
        if key in self.options:
            self.options[key].update(*value)
        else:
            self.options[key] = Menu.Item(self, key, *value)
        
        self.update()
        
    def __delitem__(self, key):
        del self.options[key]
        self.update()
        
    def __len__(self):
        return len(self.options)
    
    def update(self):
        self.mapping = [key for key in sorted(self.options)]
        self.mapping = [[i, 0, key] for i, key in enumerate(self.mapping)]
        if self.parent is not None:
            self.parent.update()
            temp, idx = self.parent.mapping, None
            if temp is None:
                return
            for x in range(len(temp)):
                idx = (x + 1 if temp[x][2] == self.name else idx)
                temp[x][1] += 1
            if idx is not None:
                self.mapping = temp[:idx] + self.mapping + temp[idx:]
