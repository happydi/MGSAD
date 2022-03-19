# Multi-granularity Semantic Alignment Distillation Learning for Remote Sensing Image Semantic Segmentation
This repository contains the PyTorch implementation of graduation project.

## 环境配置

All the codes are tested in the following environment:

* Linux (tested on Ubuntu 18.04.3 LTS)
* Python 3.6.13
* PyTorch 1.3.1
* NVDIA GeForce RTX 2080 Ti

## 框架和包

* Install PyTorch: ` conda install pytorch`
* Install torchvision: `conda install torchvision`
* Install other dependences: ` pip install opencv-python scipy `
* Install tensorboad: `conda install tensorboard`
* Install tesnsorboadX: `conda install tensorboardX`
* Install imageio:`pip install imageio`
* Install tqdm: `pip install tqdm`
* Install InPlace-ABN:
```bash
cd libs
sh build.sh
python build.py
```

## 数据集和模型

* Dataset: WHDLD, ISPRS Postam
* Teacher: PSPNet (ResNet-101)
* Student: PSPNet (ResNet-18)、PSPNet (ResNet-50)、PSPNet (MobileNetv2)
