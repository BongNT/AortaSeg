


from pathlib import Path


def add_skeleton_file(list_data, skeleton_folder):
    # list_data= [{"image": path1, "label":path2, ...} ,...]
    new_data = []
    for data in list_data:
        label_path = data["label"]
        ske_file = str(Path(label_path.replace("label", "skeleton")).name)
        full_path = str(Path(skeleton_folder)/ske_file)
        data["skeleton"] = full_path
        new_data.append(data)
    return new_data