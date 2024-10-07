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
