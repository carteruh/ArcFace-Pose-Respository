from retinaface import RetinaFace
import os
from PIL import Image
import shutil

# Clean images by deploying RetinaFace
def DetectAndAlign(root_directory_path, save_root, not_detected_root):
    
    # Make sure destination folder path exists
    if not os.path.exists(save_root):
        os.makedirs(save_root)
        
    if not os.path.exists(not_detected_root):
        os.makedirs(not_detected_root)

    # Walk from the root directory and replace every image in each subdirectory with their image in the subdirectory.
    for root, dirs, files in os.walk(root_directory_path, topdown= True):
        # Go through each file and conduct facial detection and alignment
        for file in files:
            if "jpg" in file:
                file_path = os.path.join(root, file)
                
                # Check if sub-folder path exists
                new_dir_path = os.path.join(save_root, os.path.relpath(root, root_directory_path))
                if not os.path.exists(new_dir_path):
                    os.makedirs(new_dir_path)
                
                face = RetinaFace.extract_faces(img_path= file_path, align= True, allow_upscaling= False)
                if len(face) > 0:
                    face = Image.fromarray(face[0])
                    face = face.convert("RGB")
                    print("Detected " + file)
                    # plt.show(face)
                    new_file_path = os.path.join(new_dir_path, file)
                    face.save(new_file_path)
                else:
                    print("Not Detected " + file )
                    
                    # Check if not detected sub-directory exists
                    not_detected_new_dir_path = os.path.join(not_detected_root, os.path.relpath(root, root_directory_path))
                    if not os.path.exists(not_detected_new_dir_path):
                        os.makedirs(not_detected_new_dir_path)
                    
                    not_detected_file_path = os.path.join(not_detected_new_dir_path, file)
                    shutil.copy(file_path, not_detected_file_path)
                    

if __name__ == '__main__':
    root_dir = './data/300WLPA_2d/LFPW'
    save_root=  './data/300WLPA_2d/LFPW_detected'
    not_detected_root= './data/300WLPA_2d/LFPW_not_detected'
    DetectAndAlign(root_directory_path= root_dir, save_root= save_root, not_detected_root= not_detected_root)
    