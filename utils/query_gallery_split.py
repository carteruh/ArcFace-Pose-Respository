import os
import pickle
from collections import Counter

'''
This script makes the pkl file that structures are query and gallery sets. For each pose groups, we take a single image as the query images and 
add 15 images from the same id as enrolled faces within the gallery set
'''

def addPosesToBins(pose_bin_images, pose_bin, id_name, pose_bin_structure, track_pose= True):
    if track_pose:
        # track the poses needed to obtain in each id
        poses_tracked = {
 '-90.0': 0, '-90.0_flipped': 0, '-75.0': 0, '-75.0_flipped': 0, '-67.5': 0, '-67.5_flipped': 0,
 '-60.0': 0, '-60.0_flipped': 0, '-45.0': 0, '-45.0_flipped': 0, '-30.0': 0, '-30.0_flipped': 0,
 '-22.5': 0, '-22.5_flipped': 0, '-15.0': 0, '-15.0_flipped': 0, '-0.0': 0, '-0.0_flipped': 0,
 '15.0': 0, '15.0_flipped': 0, '22.5': 0, '22.5_flipped': 0, '30.0': 0, '30.0_flipped': 0,
 '45.0': 0, '45.0_flipped': 0, '60.0': 0, '60.0_flipped': 0, '67.5': 0, '67.5_flipped': 0,
 '75.0': 0, '75.0_flipped': 0, '90.0': 0, '90.0_flipped': 0
}

        # For each ID take an images from each pose
        for i in range(1, len(pose_bin_images[pose_bin][id_name])):
            file_name = pose_bin_images[pose_bin][id_name][i]                
            yaw = '_'.join(file_name.split('/')[-1].strip('.jpeg').split('_')[5:7])
            pitch =  file_name.split('/')[-1].strip('.jpeg').split('_')[4]
            print(f'yaw: {yaw}, pitch: {pitch}')

            if poses_tracked[yaw] < 1: # We add N poses for each id_name 
                print(pose_bin_images[pose_bin][id_name][i])
                pose_bin_structure[pose_bin]["gallery"].append(pose_bin_images[pose_bin][id_name][i])
                poses_tracked[yaw] += 1
    else:
        pose_bin_structure[pose_bin]["gallery"].extend(pose_bin_images[pose_bin][id_name][1:16])
        
    return pose_bin_structure 

'''This will create the query and gallery sets for probe-gallery testing'''
def create_pose_bin_structure(base_path, pkl_file_name):
    # Get a list of all pose bin subdirectories
    pose_bins = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

    # First, let's build an ID-to-images mapping for each pose bin
    pose_bin_images = {}
    for pose_bin in pose_bins:
        pose_bin_path = os.path.join(base_path, pose_bin)
        pose_bin_images[pose_bin] = {}
        for root, _, files in os.walk(pose_bin_path):
            for file in files:
                if file.endswith('.jpeg'):
                    # Obtain the id name 
                    id_name = root.split('/')[-1]
                    if id_name not in pose_bin_images[pose_bin]:
                        pose_bin_images[pose_bin][id_name] = []
                    pose_bin_images[pose_bin][id_name].append(os.path.join(root, file))
        # print("ID to Image Mapping for each Pose Bin:", pose_bin_images[pose_bin])

    # Filter out IDs with less than 2 images and that are not present in all pose bins
    valid_ids = set()
    for pose_bin in pose_bins:
        for id_name in pose_bin_images[pose_bin]:
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
            
            # if '-90_90' in pose_bin:
            pose_bin_structure = addPosesToBins(pose_bin_images= pose_bin_images, pose_bin= pose_bin, id_name= id_name, pose_bin_structure= pose_bin_structure, track_pose=True)
            # else:
                # pose_bin_structure = addPosesToBins(pose_bin_images= pose_bin_images, pose_bin= pose_bin, id_name= id_name, pose_bin_structure= pose_bin_structure, track_pose=False)
          
            # pose_bin_structure[pose_bin]["gallery"].extend(pose_bin_images[pose_bin][id_name][1:16])
    
    # Save the structure as a .pkl file
    with open(pkl_file_name, 'wb') as f:
        pickle.dump(pose_bin_structure, f)

    return pose_bin_structure
    
if __name__ == '__main__':
    base_path = "./data/M2FPA/Test_Bins_yaw_degradation"
    pkl_file = './test_sets/query_galleries_M2FPA_Bins_yaw_degradation.pkl'
    pose_bin_structure = create_pose_bin_structure(base_path, pkl_file)
