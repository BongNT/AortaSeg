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
|  ├── inference                      # Script to infer segmentation masks using segresnet
|  └── requirements.txt               # List of Python packages required for execution        
|
├─ nnUnet/                            # Directory for segresnet baseline
   ├── inference                      # Script to infer segmentation masks using nnunet
   └── requirements.txt               # List of Python packages required for execution        

```
## Segresnet Train tutorial
Follow this [segresnet tutorial](./segresnet/README.md)

## nnUnet train
We follow this tutorial [nnUnet tutorial](https://github.com/MIC-DKFZ/nnUNet?tab=readme-ov-file#how-to-get-started) to install enviroment and train Residual Encoder nnUNet.

These are commands we used to train the model:
```
nnUNetv2_plan_and_preprocess -d 1 -pl nnUNetPlannerResEncL --verify_dataset_integrity

nnUNetv2_train 1 3d_fullres all -p nnUNetResEncUNetLPlans
```