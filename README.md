# Underwater Video Compression Based on Spatial-Frequency Dual-Domain Scaling

Official repository for the paper:

**Underwater Video Compression Based on Spatial-Frequency Dual-Domain Scaling**

This paper has been submitted to **IEEE Transactions on Circuits and Systems for Video Technology (TCSVT)**.

## Overview

This repository provides the official implementation of **RUVC**, a spatial-frequency dual-domain scaling based framework for underwater video compression.

## Environment Setup

We recommend using a standard Python virtual environment with **Python 3.10**.

```bash
conda create -n RUVC python=3.10
conda activate RUVC
cd RUVC

# Then initialize the dependencies:
bash _DependenciesInit.sh
# In most cases, the default configuration is sufficient. 
# All Enter is fine.
```

## Dataset Preparation

We provide the UVC46k dataset.

Full Dataset: https://pan.baidu.com/s/1yEKXJL1fMpc53Wk3Pxiftg (Extraction code: 4600)

Demo Dataset (only the testset and valset): https://www.kaggle.com/datasets/shichengque/uvc46k-demo

## Directory Structure

Please download and extract the dataset into a newly created dataset folder under the root directory of this repository.

The training and testing scripts use this directory as the default dataset path.

Expected directory structure:

```bash
/
└ dataset/
  ├── UVC46k_trainingset/
  │   └── 00001/
  │       └── 000.png
  ├── UVC46k_testset/
  │   └── class1_ClickerAndTarget/
  │       └── 000.png
  └── UVC46k_valset/
      └── 00000/
          └── 000.png
```

If you use a non-default dataset path, please modify the corresponding path variables in the training and testing scripts.

## Pretrain

We provide the official training log files in the `log` folder, and official weights in the `model` folder.

Please note that, for faster training and evaluation, we use a very simple validation subset for eval during training instead of the complete `UVC46k_valset`.

Therefore, the validation metrics reported in the training logs may have slight deviations and should be used only as a reference.

For the final model performance, please refer to the testing results on the official `UVC46k_testset`.

## Testing
```bash
# Full RUVC:
bash test_RUVC.sh

# Lite RUVC:
bash test_LiteRUVC.sh
```

## Training
```bash
# Full RUVC:
bash train_RUVC.sh

# Lite RUVC:
bash train_LiteRUVC.sh
```

## Acknowledgement
Thank you for your interest in our work.
