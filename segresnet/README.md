
1. install environment using requirement.txt
2. generate skeleton using script in dataset/gen_skeleton_from_mask.py
3. move to run folder, prepare data.json and task.yaml
data.json follow this structure:
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


4. run training with "python run.py" command

