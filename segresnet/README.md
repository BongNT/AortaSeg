# Training
## Move to "segresnet/run" folder, install environment using requirement.txt
```
pip install -r requirements.txt
```
## Generate skeleton using script in dataset/gen_skeleton_from_mask.py
change variables "skeleton_folder_ouput" and "base" to be suitable with your environment, dataset folder.

## Prepare data.json 
[data1.json](run/work_dir/data.json) and [data2.json](run/work_dir/data.json) must be same and follow this structure:
```
{
    "testing": [
        {
            "image": "/home/user/AortaSeg24/datasets/images/subject01_CTA.mha",
            "label": "/home/user/AortaSeg24/datasets/masks/subject01_label.mha",
            "fold": 0
        },....
    ],
    "training": [
        {
            "image": "/home/user/AortaSeg24/datasets/images/subject044_CTA.mha",
            "label": "/home/user/AortaSeg24/datasets/masks/subject044_label.mha",
            "fold": 0
        },....
    ]
}
```
## Change 2 config in [config file](run/work_dir/segresnet_0/configs/hyper_parameters.yaml)
data_file_base_dir = your current segresnet/run folder. E.g.: data_file_base_dir: /home/user/AortaSeg24/models/package/segresnet/run

skel_folder = folder that contain skeleton masks. E.g: skel_folder: "/home/user/AortaSeg24/datasets/skeletons"
## Run training with "python run.py" command


