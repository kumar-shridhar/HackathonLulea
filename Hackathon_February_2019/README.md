# Hackathon_February_2019
This repository contains the framework for the MindGarage Hackathon in February 2019


## Table of Contents
1. [Dataset Preparation](#dataset-preparation)
    1. [Tobacco-3482](#tobacco-3482)
    2. [RVL-CDIP](#rvl-cdip)
    3. [RVL-CDIP without overlap](#rvl-cdip-without-overlap)
2. [Dataset Iterators](#dataset-iterators)
    1. [Tobacco-3482 Iterator](#tobacco-3482-iterator)
    2. [RVL-CDIP Iterator](#rvl-cdip-iterator)

3. [Utilities](#utilities)

## Dataset Preparation
### Tobacco-3482
- Download and extract the Tobacco-3482 dataset from https://lampsrv02.umiacs.umd.edu/projdb/project.php?id=72



### RVL-CDIP
- Download and extract the RVL-CDIP dataset from http://www.cs.cmu.edu/~aharley/rvl-cdip/
- If you want to use OCR, also download the metadata files from https://ir.nist.gov/cdip/ and extract them to `xml/`

We expect the directory structure to be as follows:

```
.
├── images
│   ├── imagesa
│   ├── ...
│   └── imagesz
├── labels
│   ├── test.txt
│   ├── train.txt
│   └── val.txt
└── xml
    ├── iitcdip.a.a.xml
    ├── ...
    └── iitcdip.z.z.xml
```

#### OCR Generation
If you want to extract the OCR data for the image files, adjust the `base_path` in `tools/rvl_cdip-ocr_generation.py` 
and run it. The script will write a *.txt file containing the OCR data for each image file. The *.txt files will be
stored conserving the original directory structure of the image files.


### RVL-CDIP without overlap
Some of the images are contained in both of the datasets. To find these files, adjust the paths in
`tools/find_duplicates.py` and run the script.
To exclude these files from the RVL-CDIP dataset, set the `exclude_tobacco` flag in `CDIP` to `True`.

## Dataset Iterators
Refer to `example_dataset.ipynb` for how to use the dataset iterators.

### Tobacco-3482 Iterator
The `datasets.Tobacco` class takes the following options:
- `root`: the root directory of the dataset, required
- `num_train`: Number of (training+validation) samples per class. One of [10, 20, ... 100]. Default: 100
- `train_val_ratio`: Ratio of num_train images which are used for training/validation. Default: 0.8
- `num_splits`: Number of different dataset partitions. Default: 10
- `channels`: 1 or 3; return single-channel images or RGB images with R=G=B. Default: 1
- `preprocess`: List of preprocessors to do on-the-fly preprocessing. Default: None
- `random_state`: Seed for pseudo-random shuffling. Default: 1337
 
The Tobacco-3482 iterator object creates `num_splits` different partitions of the images. Each of the partitions is
itself split into three disjoint sets for training, validation and testing. To iterate over another partition, call
`Tobacco.load_split(mode, index)` where `mode` is one of `['train', 'val', 'test']` and index is the index of the
partition to load. 
 
### RVL-CDIP Iterator
The `datasets.CDIP` class takes the following options:

- `root`: the root directory of the dataset, required
- `mode`: 'train', 'val' or 'test'. Default: train
- `channels`: 1 or 3; return single-channel images or RGB images with R=G=B. Default: 1
- `include_ocr`: Whether OCR should be returned or not. Default: False
- `exclude_tobacco`: Whether images which are present in Tobacco-3482 should be excluded or not. Default: False
- `preprocess`: List of preprocessors to do on-the-fly preprocessing. Default: None
- `random_state`: Seed for pseudo-random shuffling. Default: 1337

## Utilities
`datasets.transformation` includes classes for data augmentation and conversion. These can be used for on-the-fly
preprocessing of the images and OCR data.
