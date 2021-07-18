#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 23:28:26 2020

@author: ike
"""


import setuptools


with open("README.md", "r") as f:
    long_description = f.read()


setuptools.setup(
    name="endopy",
    version="0.0.1",
    author="Ikechukwu Ogbonna",
    author_email="ieo2105@columbia.edu",
    description="Neural networks to segment Drosophila neurons",
    long_description=long_description,
    long_description_content_type='ext/markdown',
    include_package_data=True,
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[

        "pystackreg>=0.2.5",
        "readlif>=0.6.1"
    ]
)