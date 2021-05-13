# Vectorization historical map

## Pipeline
<p align="center">
<img src="./images/pipeline_v1.svg" width="800px">
</p>

## Abstract
Maps have been a unique source of knowledge for centuries. 
Such historical documents provide invaluable information for analyzing the complex spatial transformation of landscapes over important time frames. 
This is particularly true for urban areas that encompass multiple interleaved research domains (social sciences, economy, etc.). 
The large amount and significant diversity of map sources call for automatic image processing techniques in order to extract the relevant objects under a vectorial shape. 
The complexity of maps (text, noise, digitization artifacts, etc.) has hindered the capacity of proposing a versatile and efficient raster-to-vector approaches for decades. 
We propose a learnable, reproducible, and reusable solution for the automatic transformation of raster maps into vector objects (building blocks, streets, rivers). 
It is built upon the complementary strength of mathematical morphology and convolutional neural networks through efficient edge filtering.
Evenmore, we modify ConnNet and combine with deep edge filtering architecture to make use of pixel connectivity information and built an end-to-end system without requiring any post-processing techniques.
In this paper, we focus on the comprehensive benchmark on various architectures on multiple datasets coupled with a novel vectorization step.
Our experimental results on a new public dataset using COCO Panoptic metric exhibit very encouraging results confirmed by a qualitative analysis of the success and failure cases of our approach.
Code, dataset, results and extra illustrations are freely available at this github repository.

## Table of Content

* [1. Visual Results](#1-visual-results)
* [2. Installation](#2-installation)
* [3. Prepare training data](#4-preparetrainingdata)
* [4. Train](#4-train)
* [5. Evaluation](#5-evaluation)

## 1. Visual Results

## 2. Installation

### 2.1. Get the code and install Dependencies

Install Pytorch adapted to your CUDA version : 

* [Pytorch 1.8.1+cu111](https://pytorch.org/get-started/previous-versions) 
* [Torchvision 0.9.1+cu111](https://pytorch.org/get-started/previous-versions)

```shell script
git clone https://github.com/soduco/ICDAR-2021-Vectorization.git
cd ICDAR-2021-Vectorization
```
You now have the code in directory ICDAR-2021-Vectorization.
At this point, you should probably create a virtual environment. For instance:
```shell script
python3 -m venv vec-env
source ./vec-env/bin/activate
```
Now, install the dependencies (with pip).
```shell script
pip install -r requirements.txt
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

2. Patch version (already cutting into small patches to ease training, every patch have size 500px 500px), and file index .lst file: *test_pari.lst*, *train_pair.lst*, *val_pair.lst*
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

The configuration file *cfg.py*, 'data_lst' should link with the path of the .lst file, *mean_bgr* is the average value of bgr channel for the whole *train/val/test* dataset, but you can also configure *cfg.py* and more options are available:

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

If you want to train HED and BDCN with pre-train model (The folder ./train/HED_pretrain and ./train/BDCN_pretrain) please download the .pth pretrain model file (vgg16-397923af.pth) from [here](https://github.com/soduco/ICDAR-2021-Vectorization/releases) and put it into *./pretrain_weight/*.

## 5. Evaluation
