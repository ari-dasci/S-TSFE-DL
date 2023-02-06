#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 17:16:34 2022
Python TSFEDL Setup.
"""

from setuptools import setup, find_packages

from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    #  Project name.
    #  $ pip install pyDML
    name='TSFEDL',

    # Version
    version='1.0.7.1',

    # Description
    description='Time Series Spatio-Temporal Feature Extraction using Deep Learning',

    # Long description (README)
    long_description=long_description,

    # URL
    url='https://github.com/ari-dasci/S-TSFE-DL',

    # Author
    author='Ignacio Aguilera Martos, Ángel Miguel García Vico, Julian Luengo, Francisco Herrera',

    # Author email
    author_email='nacheteam@ugr.es',

    # Keywords
    keywords=['Time series',
              'Feature extraction',
              'Deep learning',
              'recurrent',
              'cnn'],

    # Packages
    packages=find_packages(exclude=['docker', 'docs', 'test', 'examples']),

    # Test suite
    test_suite='test',

    # Requeriments
    install_requires=['pytorch-lightning', 'scikit-learn', 'tensorflow==2.6.0', 'torchmetrics', 'wfdb', 'obspy', 'keras==2.6.0'],

    long_description_content_type='text/markdown'

)
