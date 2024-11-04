<!-- PROJECT LOGO -->
<br />
<p align="center">
  <h1 align="center">CTFAD: Coarse-to-Fine Vision-based <br />
Welding Spot Anomaly Detection </h1>
<br />

![Tensorflow](https://img.shields.io/badge/Tensorflow-2.16.1-orange?logo=tensorflow)
![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Linux](https://img.shields.io/badge/Linux-22.04-yellow?logo=linux)

## Overview

This repository contains a pytorch implementation for the paper: CTFAD: Coarse-to-Fine Vision-based
Welding Spot Anomaly Detection. CTFAD provides a two-stage coarse-to-fine solution for Welding Spot anomaly
detection under various challenging real industrial scenarios.

## Code Structure
```
|-- selfsupervised-ad
    |-- config
    |-- dataset
        |-- dataset_loader
            |-- ....
        |-- WeldingSpot
    |-- job
    |-- algo
        |-- Yolo
        |-- AnomalyDetector
    |-- outputs
```

## Preparation

### Installation
Clone this repository: 

```Shell
git clone https://github.com/HP-CAO/CTFAD.git
cd CTFAD
```

Install requirements in a virtual environment (Tested on anaconda/miniconda environment):

```Shell
conda create --name ctfad python==3.11
```

Install python packages:

```Shell
pip install -r requirements.txt
```

### Data Preparation for MVTec-AD
Download MVTec-AD data from [https://www.mvtec.com/](https://www.mvtec.com/company/research/datasets/mvtec-ad). Install dependencies. <mark>[To be added]</mark>


### Data Preparation for WeldingSpot
Download WeldingSpot detection data from <mark>[To be added]</mark>, and unzip it to ./dataset/weldingspot 

#### Preprocess WeldingSpot Data
Create dataset_split 
```Shell
python ./dataset/dataset_loader/utils/create_dataset_split.py
```
Create dataset_anomaly
```Shell
python dataset/dataset_loader/utils/create_dataset_anomaly.py
```

Split dataset into train and test set.
```Shell
python ./dataset/dataset_loader/utils/create_train_val.py
```

Preprocess dataset into tfrecord format. This step will take a while...
```Shell
python ./dataset/dataset_loader/utils/create_tf_record.py 
```


## Run

```
python ./job/run_train.py --config ./config/config.yaml --e
```

### Visualize Results in Tensorboard
```
tensorboard --logdir outputs
```

### Make Videos
```

```

### A More Detailed Doc is Coming Soon

### Demo



## Citation

If you find our work useful in your research, please cite:

```BiBTeX

```

## Acknowledgements
This repo is built based on the fantastic work [UniAD](https://github.com/zhiyuanyou/UniAD) by Zhiyuan You.


