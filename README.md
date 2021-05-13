# Vectorization historical map

## Pipeline
<p align="center">
<img src="./images/pipeline_v1.svg" width="800px", alt="teaser">
</p>

## Table of Content

* [1. Visual Results](#1-visual-results)
* [2. Installation](#2-installation)
* [3. Prepare training data](#4-preparetrainingdata)
* [4. Train](#4-train)
* [5. Evaluation](#5-evaluation)

## 1. Visual Results

## 2. Installation

### 2.1. Dependencies

Install Pytorch adapted to your CUDA version : 

* [Pytorch 1.8.1+cu111](https://pytorch.org/get-started/previous-versions) 
* [Torchvision 0.9.1+cu111](https://pytorch.org/get-started/previous-versions)

Install dependencies through requirement.txt: 
``` Bash
pip install -r requirement.txt
```

### 2.2. Pre-trained models 

Download pre-trained models:

Please check the [Releases](https://github.com/soduco/ICDAR-2021-Vectorization/releases) to download it.

### 2.3. Datasets

Dataset used to train, validate and test our approach:

Please check the [Releases](https://github.com/soduco/ICDAR-2021-Vectorization/releases) to download it.

## 3. Prepare training data

### 3.1. Download the dataset

To ease the usage of training data, we provide two versions of dataset:

1. Full images (~7000px * 11000px)

2. Patch version (already cutting into small patches to ease training, every patch have size 500px 500px), and file index .lst file: test_pari.lst, train_pair.lst, val_pair.lst
Please check the [Releases](https://github.com/soduco/ICDAR-2021-Vectorization/releases) to download it.

### 3.2. Reproducing the training on historical maps 

The training patch data need to be downloaded from [here](https://github.com/soduco/ICDAR-2021-Vectorization/releases), zipped and saved into `./data_generator`. 
The file index .lst file can be also download in the same Release.

The file structure is : 

```
./ICDAR-2021-Vectorization/data_generator
├── train_data_tiles
├── val_data_tiles
├── test_data_tiles
├── train_gt_tiles
├── val_gt_tiles
├── test_gt_tiles
├── train_pari.lst
├── val_pair.lst
└── test_pair.lst
```

The configuration file cfg.py, 'data_lst' should link with the path of the .lst file, mean_bgr is the average value of bgr channel for the whole train/val/test dataset, but you can also configure *cfg.py* and more options are available:

```python
config = {
    'HistoricalMap2020': {'data_root': '../../data_generator/',
                'data_lst': 'train_pair.lst',
                'mean_bgr': [168.1872554, 195.97301654, 204.64264255]},
}

config_test = {
    'HistoricalMap2020': {'data_root': '../../data_generator/',
                'data_lst': 'test_pair.lst',
                'mean_bgr': [158.0, 191.0, 210.0]},
}

config_val = {
    'HistoricalMap2020': {'data_root': '../../data_generator/',
                'data_lst': 'val_pair.lst',
                'mean_bgr': [168.1872554, 195.97301654, 204.64264255]},
}

```

## 4. Train
we can now train the models with the prepared data. Remeber the data should be patched and .lst file should be correctly generated and put into the right place. If you want to get more control for your training, you can list all the arguments:

 ``` Bash
cd ./train/<model_name>/train.py
python train.py --help
```

If you want to train HED and BDCN with pre-train model (The folder ./train/HED_pretrain and ./train/BDCN_pretrain) please download the .pth pretrain model file (vgg16-397923af.pth) from [here](https://github.com/soduco/ICDAR-2021-Vectorization/releases) and put it into ./pretrain_weight/.

## 5. Evaluation
