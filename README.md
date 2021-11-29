<div align="center">    
 
# pyCNN-LSTM: A python package for Time series feature extraction and forecasting by means of convolutional and recurrent deep neural networks.

<!--
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/NeurIPS-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/ICLR-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)  

ARXIV   
[![Paper](http://img.shields.io/badge/arxiv-math.co:1480.1111-B31B1B.svg)](https://www.nature.com/articles/nature14539)
![CI testing](https://github.com/PyTorchLightning/deep-learning-project-template/workflows/CI%20testing/badge.svg?branch=master&event=push)
-->

</div>
 
## Description   
Time series feature extraction is a classical problem in time series analysis. Classical addition and multiplication models have been used for this purpose until the appearance of Artificial Neural Networks and Deep Learning. This problem has gained attention since multiple real life problems imply the usage of time series.

In this repository we introduce a new Python module which compiles 20 backbones for time series feature extraction using Deep Learning. This module has been created to cover the necessity of a versatile and expandable piece of software for practitioners to use in their problems.

## How to run   
First, install dependencies   
```bash
# clone project   
git clone https://github.com/ari-dasci/S-CNN-LSTM.git

# install project   
cd S-CNN-LSTM
pip install -e .   
pip install -r docker/requirements.txt
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
import pyCNN_LSTM.models_keras as pyCNN_LSTM

# get the OhShuLih model
model = pyCNN_LSTM.OhShuLih(input_tensor=input, include_top=True)

# compile and fit as usual
model.compile(optimizer='Adam')
model.fit(X, y, epochs=20)
```

### Citation   

Please cite this work as:

*not available yet*

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
