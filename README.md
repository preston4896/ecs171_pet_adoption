# ECS 171 - Predicting Per Adoption Speed

Group project for F19 ECS171 by: Amitpal Gill, April Vang, Brian Cirieco, Gan Qiu, Guanzi Yao, Pavlos Maltsev, Preston Ong, Stephanie Olivera, Xuan Deng, Zoe Kanavas

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. The ```src``` folder contains all source files, while ```data_management.py``` contains data analysis.

```scr/main.py``` and ```scr/main.ipynb``` are scripts that will run the whole code pipeline for you and report useful information as well as output all graphs. Each file can be run individually, or the ```main``` script will run everything in the correct order.

### Prerequisites

In order to run the source files, please install:
* pip: astetik, numpy, pandas, scipy, tensorflow, keras, sklearn, talos, lightgbm
* Homebrew: libomp

## Running

```src/``` files are meant to be run in the following order: 
1. get_data_final.py
2. pca.py
3. HyperOptimization.py
4. test_ANN_model.py
5. pr_curve.py

* ```get_data_final.py``` will remove outliers, normalize and split data into training testing sets.
* ```pca.py``` is mainly use for visualization purposes
* ```HyperOptimization.py``` runs a grid search on the ANN ro determine best params. It is recommended not to run this file since it takes a long time to complete. The optimal hyper-parameters have been saved and are already used by the ANN (rerunning this script will override the current configuration).
* ```test_ANN_model.py``` runs the ANN with best params and RFE results,  training/testing error & accuracy, and useful graphs.
* ```pr_curve.py``` will output ROC and AUC.

### Again, ```src/main.py``` or ```scr/main.ipynb``` will initialize the correct pipeline and output all the information.
