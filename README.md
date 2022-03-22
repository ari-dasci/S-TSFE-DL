# TSFEDL: A Python Library for Time Series Spatio-Temporal Feature Extraction and Prediction using Deep Learning.

## Description

Time series feature extraction is a classical problem in time series analysis. Classical addition and multiplication models have been used for this purpose until the appearance of Artificial Neural Networks and Deep Learning. This problem has gained attention since multiple real life problems imply the usage of time series.

In this repository we introduce a new Python module which compiles 20 backbones for time series feature extraction using Deep Learning. This module has been created to cover the necessity of a versatile and expandable piece of software for practitioners to use in their problems.

## How to run

First, install dependencies

```bash
# clone project
git clone https://github.com/ari-dasci/S-TSFE-DL.git

# install project
cd S-TSFE-DL
pip install -e .
```   

In order to run a example, navigate to any file and run it.

```bash
cd project/examples

# run example
python arrythmia_experiment.py
```

## Imports
This project is setup as a package which means you can now easily import any file into any other file like so:

```python
import tensorflow as tf
import TSFEDL.models_keras as TSFEDL

# get the OhShuLih model
model = TSFEDL.OhShuLih(input_tensor=input, include_top=True)

# compile and fit as usual
model.compile(optimizer='Adam')
model.fit(X, y, epochs=20)
```

## Citation

Please cite this work as:

Time Series Feature Extraction using Deep Learning library (https://github.com/ari-dasci/S-TSFE-DL/)

Paper citation is pending.

<!--
```
@article{YourName,
  title={Your Title},
  author={Your team},
  journal={Location},
  year={Year}
}
```
-->
