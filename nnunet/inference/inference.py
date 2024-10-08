"""
nnU-Net template
"""

import torch
import os
import numpy as np
import SimpleITK
from glob import glob
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO

def run():
    _show_torch_cuda_info()
    
    # nnUNet_results = "/opt/app/nnUNet/nnUNet_results"
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=True,
        device=torch.device('cuda', 0),
        verbose=True,
        verbose_preprocessing=True,
        allow_tqdm=True
    )
    predictor.initialize_from_trained_model_folder(
        "./resources",
        use_folds=("all"),
        checkpoint_name='checkpoint_best.pth',
    )
    
    input_folder = "/input/images/ct-angiography"
    output_folder = "/output/images/aortic-branches"
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, mode=777, exist_ok=True)
    
    input_files = glob(str(input_folder + "/*.tiff")) + glob(str(input_folder + "/*.mha"))
    input_file_name = input_files[0]
    output_file_name = output_folder + "/output.mha"

    # predict a single numpy array
    img, props = SimpleITKIO().read_images([input_file_name])
    print("img.shape: ", img.shape)
    print("props: ", props)
    pred_array = predictor.predict_single_npy_array(img, props, None, None, False)
    # post process in here
    
    pred_array = pred_array.astype(np.uint8)
    print("pred_array.shape: ", pred_array.shape)

    image = SimpleITK.GetImageFromArray(pred_array)
    image.SetDirection(props['sitk_stuff']['direction'])
    image.SetOrigin(props['sitk_stuff']['origin'])
    image.SetSpacing(props['sitk_stuff']['spacing'])
    SimpleITK.WriteImage(
        image,
        output_file_name,
        useCompression=True,
    )
                                 
    print('Saved!!!')
    return 0

def _show_torch_cuda_info():

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