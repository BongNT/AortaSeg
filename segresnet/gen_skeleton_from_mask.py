
import os
import sys
from skimage.morphology import skeletonize, dilation
from pathlib import Path
import SimpleITK as sitk


import numpy as np
# import json
from tqdm import tqdm
# import pandas as pd
# import matplotlib.pyplot as plt
base = Path("/home/user/AortaSeg24/datasets/masks")
files = [x for x in base.glob('*.mha') if x.is_file()]
skeleton_folder_ouput = "/home/user/AortaSeg24/datasets/skeletons"
def gen_skl_save(path):
    ske_file = str(Path(path.replace("label", "skeleton")).name)
    output_path = str(Path(skeleton_folder_ouput)/ske_file)
    print(output_path)
    file = sitk.ReadImage(path)
    mask = sitk.GetArrayFromImage(file)
    spacing = file.GetSpacing()
    direction = file.GetDirection()
    origin = file.GetOrigin()
    print(mask.shape)
    mask2 = np.copy(mask)
    mask[mask>0] ==1
    skeleton = skeletonize(mask, method='lee').astype(np.uint8)
    skeleton = dilation(dilation(skeleton))
    skeleton = skeleton * mask2
    print(skeleton.shape)
    # print(np.unique(skeleton))
    # print(np.sum(skeleton))
    image = sitk.GetImageFromArray(skeleton)
    image.SetDirection(direction) # My line
    image.SetOrigin(origin)
    image.SetSpacing(spacing)
    sitk.WriteImage(image,output_path)
    
for f in tqdm(sorted(files)):
    print(f)
    gen_skl_save(str(f))
    # break