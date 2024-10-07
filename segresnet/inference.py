"""
The following is a simple example algorithm.

It is meant to run within a container.

To run it locally, you can call the following bash script:

  ./test_run.sh

This will start the inference and reads from ./test/input and outputs to ./test/output

To save the container and prep it for upload to Grand-Challenge.org you can call:

  ./save.sh

Any container that shows the same behavior will do, this is purely an example of how one COULD do it.

Happy programming!
"""
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


# print("Before the sleep statement")
# time.sleep(60)

print('All required modules are loaded!!!')

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("resources")


def run():
    # Read the input
    # Read the input
    image, spacing, direction, origin = load_image_file_as_array(
        location=INPUT_PATH / "images/ct-angiography",
    )
    
    
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
    
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    image = transform(image)
    image = image.permute(2, 1, 0).unsqueeze(0).unsqueeze(0)
    print(f"image shape {image.shape}")
    model = SegResNetDS(spatial_dims=3, 
                    init_filters=32,
                    in_channels=1,
                    out_channels=24,
                    blocks_down=[1, 2, 4, 5, 6],
                    norm="INSTANCE",
                    act="leakyrelu",
                    dsdepth=4)
    state_dict = torch.load(str(RESOURCE_PATH/"model_final.pt"), map_location='cpu')["state_dict"]
    model.load_state_dict(state_dict=state_dict)
    del state_dict  # Free memory used by the state dictionary
    gc.collect()
    model = model.to(device)
    image = image.to(device)
    print("Defined the model and loaded both image and model to the appropriate device...")
    print(image.shape)
    model.eval()
    
    sliding_inferrer = SlidingWindowInfererAdapt(
                roi_size=[160, 160, 256],
                sw_batch_size=1,
                overlap=0.625,
                mode="gaussian",
                cache_roi_weight_map=True,
                progress=True,
            )
    with torch.no_grad():
        with autocast(enabled=True):
            val_outputs = sliding_inferrer(inputs=image, network=model)
    val_outputs = val_outputs.detach().cpu()
    
    
    print('Done with prediction! Now saving!!!')
    pred_label = torch.argmax(val_outputs, dim = 1).to(torch.uint8)
    
    
    del model # to save some memory
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
        location=OUTPUT_PATH / "images/aortic-branches",
        array=aortic_branches,
        spacing=spacing, 
        direction=direction, 
        origin=origin,
    )
    print('Saved!!!')
    return 0


def load_image_file_as_array(*, location):
    # Use SimpleITK to read a file
    input_files = glob(str(location / "*.tiff")) + glob(str(location / "*.mha"))
    print(location)
    print(input_files)
    result = SimpleITK.ReadImage(input_files[0])
    spacing = result.GetSpacing()
    direction = result.GetDirection()
    origin = result.GetOrigin()
    # Convert it to a Numpy array
    return SimpleITK.GetArrayFromImage(result), spacing, direction, origin


def write_array_as_image_file(*, location, array, spacing, origin, direction):
    location.mkdir(parents=True, exist_ok=True)

    # You may need to change the suffix to .tiff to match the expected output
    suffix = ".mha"

    image = SimpleITK.GetImageFromArray(array)
    image.SetDirection(direction) # My line
    image.SetOrigin(origin)
    SimpleITK.WriteImage(
        image,
        location / f"output{suffix}",
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



