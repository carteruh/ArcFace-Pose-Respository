import random
import os
import shutil

# Set your dataset directory path here
def MakeFoldersTrainTest(dataset_dir, dist_dir):
    # Set your train, val, and test ratios here
    train_ratio = 0.8
    test_ratio = 0.2

    # Get the list of subdirectories in the dataset directory
    subdirs = [name for name in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, name))]

    # Shuffle the list of subdirectories
    random.shuffle(subdirs)

    # Split the subdirectories into train, val, and test sets
    train_subdirs = subdirs[:int(len(subdirs) * train_ratio)]
    test_subdirs = subdirs[int(len(subdirs) * (train_ratio)):]

    # Create the train, val, and test directories
    train_dir = os.path.join(dist_dir, "HELEN_train")
    os.makedirs(train_dir, exist_ok=True)
    test_dir = os.path.join(dist_dir, "HELEN_test")
    os.makedirs(test_dir, exist_ok=True)

    # Move the images from the subdirectories to the train, val, and test directories
    for subdir in train_subdirs:
        src_dir = os.path.join(dataset_dir, subdir)
        dst_dir = os.path.join(train_dir, subdir)
        os.makedirs(dst_dir, exist_ok=True)
        for img_file in os.listdir(src_dir):
            src_path = os.path.join(src_dir, img_file)
            dst_path = os.path.join(dst_dir, img_file)
            shutil.copy(src_path, dst_path)

    for subdir in test_subdirs:
        src_dir = os.path.join(dataset_dir, subdir)
        dst_dir = os.path.join(test_dir, subdir)
        os.makedirs(dst_dir, exist_ok=True)
        for img_file in os.listdir(src_dir):
            src_path = os.path.join(src_dir, img_file)
            dst_path = os.path.join(dst_dir, img_file)
            shutil.copy(src_path, dst_path)
            
if __name__ == '__main__':
    dataset_dir = "./data/300WLPA_2d/HELEN_detected/"
    MakeFoldersTrainTest(dataset_dir=dataset_dir, dist_dir= './data/300WLPA_2d/')