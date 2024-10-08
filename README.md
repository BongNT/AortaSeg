<div align=center>
    <h1>MedAI team's baselines</h1>

This document outlines the steps for running our baselines.

---

</div>

## Directory Structure

Before we begin, let's familiarize ourselves with the directory structure required:


```
AortaSeg/
├─ segresnet/                         # Directory for segresnet baseline
|  ├── run                            # Source code for training segresnet
|  ├── gen_skeleton_from_maks         # Script to create skeletons from masks folder
|  ├── inference/                     # Scripts to infer segmentation masks using segresnet
|  └── requirements.txt               # List of Python packages required for execution        
|
├─ nnUnet/                            # Directory for segresnet baseline
   ├── inference/                     # Scripts to infer segmentation masks using nnunet
   └── requirements.txt               # List of Python packages required for execution        

```
## Segresnet Train tutorial
Follow this [segresnet train tutorial](./segresnet/README.md)

## nnUnet train
We follow this tutorial [nnUnet tutorial](https://github.com/MIC-DKFZ/nnUNet?tab=readme-ov-file#how-to-get-started) to install enviroment and train Residual Encoder nnUNet.

These are commands we used to train the model:
```
nnUNetv2_plan_and_preprocess -d 1 -pl nnUNetPlannerResEncL --verify_dataset_integrity

nnUNetv2_train 1 3d_fullres all -p nnUNetResEncUNetLPlans
```

## Segresnet inference
```
inference/
├─ resources/                         
|  └── model.pt                       # Pretrained model   
├─ test/                            
|  ├── input/images/ct-angiography/    # input of inferenence.py
|  └── ouput/                          # output of inferenence.py
├─ Dockerfile                         
├─ requirements                         
├─ save.sh                            # script to export Docker to tar.gz
├─ test_run.sh                        # script to build and test Docker in local computer
├─ inferenence.py                     # script to segment a file in "test/input" using Segresnet
└─ multi_infer.py                     # script to segment multiple files using Segresnet

```
1. Copy your pretrained model to resources, and change the pretrained model structure in inferenence.py or multi_infer.py
2. Change GLOBAL_VARIABLES in multi_infer.py to segment all images in any folder that you choose
3. If you want to build a docker and infer single image, please copy image to "ct-angiography/", then run "sh test.sh". It will build a docker and use inference.py to segment the image.

## nnUnet inference
```
inference/
├─ resources/   
|  ├── dataset.json                   # dataset config that used to train the pretrained model
|  ├── plans.json                     # model nnUnet structures that is suitable with the pretrained model
|  └── fold_all                        
|       └── checkpoint_final.pth                 # Pretrained model   
├─ test/                            
|  ├── input/images/ct-angiography    # input of inferenence.py
|  └── ouput                          # output of inferenence.py
├─ nnunetv2                           # please clone this folder from https://github.com/MIC-DKFZ/nnUNet.git                    
├─ Dockerfile                         
├─ requirements                         
├─ save.sh                            # script to export Docker to tar.gz
├─ test_run.sh                        # script to build and test Docker in local computer
├─ inferenence.py                     # script to segment a file in "test/input" using nnUnet
└─ multi_infer.py                     # script to segment multiple files using nnUnet

```
1. Clone nnunetv2 from https://github.com/MIC-DKFZ/nnUNet.git
2. Copy your pretrained_model, dataset.json, plans.json to resources
3. Change GLOBAL VARIABLES in multi_infer.py to segment all images in any folder that you choose
4. If you want to build a docker and infer single image, please copy image to "ct-angiography/", then run "sh test.sh". It will build a docker and use inference.py to segment the image.