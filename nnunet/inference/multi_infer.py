
from pathlib import Path

from glob import glob

import torch
import os
import numpy as np
import SimpleITK
from glob import glob
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
import numpy as np
import gc, os

import time

# GLOBAL VARIABLES
IMG_INPUT_FOLDER = Path("/home/user/AortaSeg24/datasets")
OUTPUT_PATH_FOLDER = Path("/home/user/AortaSeg24/docker_nnuet/infer2")
RESOURCE_PATH_FOLDER = Path("/home/user/AortaSeg24/docker_nnuet/resources")
CHECKPOINT_NAME = "checkpoint_final.pth"
FOLD = "all" # {RESOURCE_PATH_FOLDER}/folder_{FOLD}/{CHECKPOINT_NAME}
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
    # Read the input
    
    
    # Process the inputs: any way you'd like
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
    
    
    
    predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=False,
            perform_everything_on_device=True,
            device=torch.device('cuda', DEVICE),
            verbose=True,
            verbose_preprocessing=True,
            allow_tqdm=True
        )
    predictor.initialize_from_trained_model_folder(
        "./resources",
        use_folds=(FOLD),
        checkpoint_name=CHECKPOINT_NAME,
    )
    files = image_and_masks_paths(str(IMG_INPUT_FOLDER))
    # print(files)
    for path in files:
            # nnUNet_results = "/opt/app/nnUNet/nnUNet_results"`
        
        

        output_file_name = OUTPUT_PATH_FOLDER/str(Path(path["image"]).name)

        # predict a single numpy array
        img, props = SimpleITKIO().read_images([str(path["image"])])
        print("img.shape: ", img.shape)
        print("props: ", props)
        pred_array = predictor.predict_single_npy_array(img, props, None, None, False)
        # post process in here
        
        pred_array = pred_array.astype(np.uint8)
        print("pred_array.shape: ", pred_array.shape)
        if not os.path.exists(str(OUTPUT_PATH_FOLDER.absolute())):
            os.makedirs(str(OUTPUT_PATH_FOLDER.absolute()), mode=777, exist_ok=True)
        image = SimpleITK.GetImageFromArray(pred_array)
        image.SetDirection(props['sitk_stuff']['direction'])
        image.SetOrigin(props['sitk_stuff']['origin'])
        image.SetSpacing(props['sitk_stuff']['spacing'])
        SimpleITK.WriteImage(
            image,
            str(output_file_name),
            useCompression=True,
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



