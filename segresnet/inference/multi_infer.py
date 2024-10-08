
from pathlib import Path

from glob import glob
import SimpleITK
import numpy
from torch.cuda.amp import GradScaler, autocast

import torch
from monai.apps.auto3dseg.transforms import EnsureSameShaped
from monai.inferers import SlidingWindowInfererAdapt
from monai.transforms import (
    Compose,EnsureType, ScaleIntensityRange, Lambda, CastToType
)
from monai.networks.nets import SegResNetDS
from monai.inferers import sliding_window_inference
import numpy as np
import gc, os

import time

# GLOBAL VARIABLES
IMG_INPUT_FOLDER = Path("/home/user/AortaSeg24/datasets")
OUTPUT_PATH_FOLDER = Path("/home/user/AortaSeg24/docker_nnuet/infer2")
RESOURCE_PATH_FOLDER = Path("./resources")
CHECKPOINT_NAME = "model.pt"# {RESOURCE_PATH_FOLDER}/{CHECKPOINT_NAME}
DEVICE = 0


print('All required modules are loaded!!!')

def image_and_masks_paths(root_dir):
    print(root_dir)
    # Initialize empty lists to store image and segmentation file paths
    image_paths = []
    segmentation_paths = []

    # Iterate through all subdirectories within the root directory
    for subdir, dirs, files in os.walk(root_dir):
        # Check for the "images" subdirectory
        images_subdir = os.path.join(subdir, "images")
        if os.path.isdir(images_subdir):
            # Find the image file within the "images" subdirectory
            for filename in os.listdir(images_subdir):
                if filename.lower().endswith((".mha")):  # Adjust extensions as needed
                    image_path = os.path.join(images_subdir, filename)
                    image_paths.append(image_path)

        # Check for the "masks" subdirectory
        masks_subdir = os.path.join(subdir, "masks")
        if os.path.isdir(masks_subdir):
            # Find the segmentation file within the "masks" subdirectory
            for filename in os.listdir(masks_subdir):
                segmentation_path = os.path.join(masks_subdir, filename)
                segmentation_paths.append(segmentation_path)
    files = [{"image": image_name, "label":label_name} for image_name, label_name in zip(sorted(image_paths), sorted(segmentation_paths))]
    return files

def run():
    # Read the input
    _show_torch_cuda_info()
    # Set the environment variable to handle memory fragmentation
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    ############# Lines You can change ###########
    # Requirement: 
    # GPU: 16GB,
    # CPU: 32GB, 
    # Maximum processing duration per case: 5 minutes ,
    # Input_path_format: /input/images/ct-angiography/image_name.mha
    # Output_path_format: /output/images/aortic-branches/image_name.mha
    # Orientation of input and output: RAS
    # Spacing of input and output: 1,1,1
    
    
    
    device = torch.device(f"cuda:{DEVICE}" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    
    transform = Compose(
        [
            EnsureType(data_type="tensor", dtype=torch.float),
            # range
            # ScaleIntensityRange(
            #     a_min=-25.025, a_max=470.259375, b_min=-1, b_max=1, clip=False
            # ),
            # Lambda( func=lambda x: torch.sigmoid(x)), 
            
            #zscore
            Lambda(func=lambda x: (x - x.mean()) /(max(x.std(), 1e-8)))
            # CastToType(dtype=label_dtype)
        ]
    )
    model = SegResNetDS(spatial_dims=3, 
                    init_filters=32,
                    in_channels=1,
                    out_channels=24,
                    blocks_down=[1, 2, 4, 5, 6],
                    norm="INSTANCE",
                    act="leakyrelu",
                    dsdepth=4)
    state_dict = torch.load(str(RESOURCE_PATH_FOLDER/CHECKPOINT_NAME), map_location='cpu')["state_dict"]
    model.load_state_dict(state_dict=state_dict)
    del state_dict  # Free memory used by the state dictionary
    gc.collect()
    model = model.to(device)
    sliding_inferrer = SlidingWindowInfererAdapt(
                roi_size=[160, 160, 256],
                sw_batch_size=1,
                overlap=0.625,
                mode="gaussian",
                cache_roi_weight_map=True,
                progress=True,
            )
    
    files = image_and_masks_paths(IMG_INPUT_FOLDER)
    # print(files)
    for path in files:
        image, spacing, direction, origin = load_image_file_as_array(path["image"])
        image = transform(image)
        image = image.permute(2, 1, 0).unsqueeze(0).unsqueeze(0)
        print(f"image shape {image.shape}")
        image = image.to(device)
        print("Defined the model and loaded both image and model to the appropriate device...")
        print(image.shape)
        model.eval()
        
        with torch.no_grad():
            with autocast(enabled=True):
                val_outputs = sliding_inferrer(inputs=image, network=model)
        val_outputs = val_outputs.detach().cpu()
        
        
        print('Done with prediction! Now saving!!!')
        pred_label = torch.argmax(val_outputs, dim = 1).to(torch.uint8)
        
        
        del image # to save some memory
        del val_outputs # to save some memory
        torch.cuda.empty_cache()
        gc.collect()    
        aortic_branches = pred_label.squeeze().permute(2, 1, 0).numpy()
        print(f"Aortic Branches: Min={np.min(aortic_branches)}, Max={np.max(aortic_branches)}, Type={aortic_branches.dtype}")
        
        
        ########## Don't Change Anything below this 
        # For some reason if you want to change the lines, make sure the output segmentation has the same properties (spacing, dimension, origin, etc) as the 
        # input volume
        # Save your output
        write_array_as_image_file(
            location=OUTPUT_PATH_FOLDER,
            array=aortic_branches,
            spacing=spacing, 
            direction=direction, 
            origin=origin,
            file_name=str(Path(path["image"]).name)
        )
        print('Saved!!!')
    return 0


def load_image_file_as_array(path):
    print(path)
    # Use SimpleITK to read a file
    result = SimpleITK.ReadImage(path)
    spacing = result.GetSpacing()
    direction = result.GetDirection()
    origin = result.GetOrigin()
    # Convert it to a Numpy array
    return SimpleITK.GetArrayFromImage(result), spacing, direction, origin


def write_array_as_image_file(*, location, array, spacing, origin, direction, file_name = f"output.mha"):
    location.mkdir(parents=True, exist_ok=True)

    # You may need to change the suffix to .tiff to match the expected output

    image = SimpleITK.GetImageFromArray(array)
    image.SetDirection(direction) # My line
    image.SetOrigin(origin)
    SimpleITK.WriteImage(
        image,
        location / file_name,
        useCompression=True,
    )


def _show_torch_cuda_info():
    import torch

    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)


if __name__ == "__main__":
    raise SystemExit(run())



