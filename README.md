# MachineLearningProjects
This repository contains various machine learning projects I've engaged in. Each subdirectory represents a different project.

# Malgo models

This repo is dedicated to Machine Learning models I've created in both an academic and non-academic setting. My focus is implementing models for PoC and exploring how different models perform on the same datasets. 

## Getting Started

Using a python3 environment, run the following to install required libraries:
```
pip install -r requirements.txt
```
Recommend using virtualenv to sandbox your work

### Manual Install 
Recommend using [Anaconda](https://www.anaconda.com/distribution/). Anaconda does not come with TensorFlow, Keras, or Prophet so you will need to install those seperately. 
```
pip install pyex quandl tensorflow keras fbprophet 
```
For additional information on installing [TensorFlow](https://www.tensorflow.org/install), [Keras](https://keras.io/#installation), and [Prophet](https://facebook.github.io/prophet/docs/installation.html) 

If you do not wish to use Anaconda, the following should satisfy the requirements: 
```
pip install pyex quandl pandas numpy matplotlib scipy scikit-learn h5py tensorflow keras fbprophet
```
