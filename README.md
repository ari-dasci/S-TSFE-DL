# TSFEDL: A Python Library for Time Series Spatio-Temporal Feature Extraction and Prediction using Deep Learning.

## Description

Time series feature extraction is a classical problem in time series analysis. Classical addition and multiplication models have been used for this purpose until the appearance of Artificial Neural Networks and Deep Learning. This problem has gained attention since multiple real-life problems imply the usage of time series.

In this repository, we introduce a new Python module which compiles 20 backbones for time series feature extraction using Deep Learning. This module has been created to cover the necessity of a versatile and expandable piece of software for practitioners to use in their problems.

## How to run

### Conda environment for GPU clusters

To easily use the library inside a conda environment the following commands are recommended to install the module. First of all install pip inside anaconda, which will install python inside the environment as well to encapsulate the whole installation.

```bash
conda install -c anaconda pip
```

After this, if a GPU is going to be used, we should install cuDNN 8.2.1 for the current tensorflow-gpu version (2.6.0). The NVIDIA CUDA toolkit will be also installed as a cuDNN dependency.

```bash
conda install -c anaconda cudnn==8.2.1
```

Finally, we can install the TSFEDL library using pip3 (which will be inside the conda environment, you can check this by running "which pip3"). This will install as dependencies pytorch-lightning, pytorch, tensorflow-gpu and all the needed packages. Use the --use-feature=2020-resolver flag if the installation runs into an error.

```bash
pip3 install --use-feature=2020-resolver tsfedl
```

Otherwise use

```bash
pip3 install tsfedl
```

After this everything is set up.

### PyPi

The module is uploaded to PyPi for an easy installation:
```bash
pip install tsfedl
```
or
```bash
pip3 install tsfedl
```

### Using the repository

First, install dependencies

```bash
# clone project
git clone https://github.com/ari-dasci/S-TSFE-DL.git

# install project
cd S-TSFE-DL
pip install -e .
```   

### Examples

To run an example, navigate to any file and run it.

```bash
cd project/examples

# run example
python arrythmia_experiment.py
```

## Imports
This project is set up as a package which means you can now easily import any file into any other file like so:

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

ArXiV reference: https://arxiv.org/abs/2206.03179

```
@article{https://doi.org/10.48550/arxiv.2206.03179,
  doi = {10.48550/ARXIV.2206.03179},
  url = {https://arxiv.org/abs/2206.03179},
  author = {Aguilera-Martos, Ignacio and García-Vico, Ángel M. and Luengo, Julián and Damas, Sergio and Melero, Francisco J. and Valle-Alonso, José Javier and Herrera, Francisco},
  keywords = {Neural and Evolutionary Computing (cs.NE), Artificial Intelligence (cs.AI), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {TSFEDL: A Python Library for Time Series Spatio-Temporal Feature Extraction and Prediction using Deep Learning (with Appendices on Detailed Network Architectures and Experimental Cases of Study)},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```
