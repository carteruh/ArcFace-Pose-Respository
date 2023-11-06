import os
import shutil

def merge_identity_subfolders(dataset_path):
    # for dataset in os.listdir(root_dir):
    #     dataset_path = os.path.join(root_dir, dataset)
    #     print(dataset_path)
    #     if not os.path.isdir(dataset_path):
    #         continue
        
    # Step 1: Walk through the dataset directory and identify subfolders
    for root, dirs, files in os.walk(dataset_path):
        # We don't want to recurse further since we're manipulating the directory structure
        break

    identity_map = {}
    for dir in dirs:
        # Split folder name to extract identity
        parts = dir.split('_')
        if len(parts) < 3: 
            # Not in the expected format, skip
            continue
        identity = f"{parts[0]}_{parts[1]}"
        
        # Collect all subfolders belonging to the same identity
        if identity not in identity_map:
            identity_map[identity] = []
        identity_map[identity].append(dir)

    # Step 2: Merge subfolders belonging to the same identity
    for identity, subfolders in identity_map.items():
        new_dir_path = os.path.join(dataset_path, identity)
        os.makedirs(new_dir_path, exist_ok=True)

        for subfolder in subfolders:
            subfolder_path = os.path.join(dataset_path, subfolder)
            for file in os.listdir(subfolder_path):
                file_path = os.path.join(subfolder_path, file)
                if os.path.isfile(file_path):
                    shutil.copy2(file_path, new_dir_path)

            # Optionally: Remove the old subfolder after copying
            shutil.rmtree(subfolder_path)

if __name__ == '__main__':
    root_directory = './data/300WLPA_2d/HELEN_detected'
    merge_identity_subfolders(root_directory)
