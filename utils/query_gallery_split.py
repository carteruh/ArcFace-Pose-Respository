import os
import pickle

def create_pose_bin_structure(base_path):
    # Get a list of all pose bin subdirectories
    pose_bins = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    # print("Available Pose Bins:", pose_bins)


    # First, let's build an ID-to-images mapping for each pose bin
    pose_bin_images = {}
    for pose_bin in pose_bins:
        pose_bin_path = os.path.join(base_path, pose_bin)
        pose_bin_images[pose_bin] = {}
        for root, _, files in os.walk(pose_bin_path):
            for file in files:
                if file.endswith('.jpg'):
                    # Obtain the id name 
                    id_name = root.split('/')[-1].split('_')[-1]
                   
                    if id_name not in pose_bin_images[pose_bin]:
                        pose_bin_images[pose_bin][id_name] = []
                    pose_bin_images[pose_bin][id_name].append(os.path.join(root, file))
        # print("ID to Image Mapping for each Pose Bin:", pose_bin_images[pose_bin])

    
    # Filter out IDs with less than 2 images and that are not present in all pose bins
    valid_ids = set()
    for id_name in pose_bin_images[pose_bins[0]]:
        if all(id_name in pose_bin_images[pose_bin] for pose_bin in pose_bins):
            if all(len(pose_bin_images[pose_bin][id_name]) >= 2 for pose_bin in pose_bins):
                valid_ids.add(id_name)
    print("Valid IDs:", valid_ids)

                
    # Define the query and gallery sets based on the above criteria
    pose_bin_structure = {}
    for pose_bin in pose_bins:
        pose_bin_structure[pose_bin] = {"query": [], "gallery": []}
        for id_name in valid_ids:
            # First image as query, rest as gallery
            pose_bin_structure[pose_bin]["query"].append(pose_bin_images[pose_bin][id_name][0])
            pose_bin_structure[pose_bin]["gallery"].extend(pose_bin_images[pose_bin][id_name][1:])
    
    # Save the structure as a .pkl file
    with open('./test_sets/query_galleries_HELEN.pkl', 'wb') as f:
        pickle.dump(pose_bin_structure, f)

    return pose_bin_structure

base_path = "./data/300WLPA_2d/HELEN_test_bins_rad2degrees/"
pose_bin_structure = create_pose_bin_structure(base_path)
