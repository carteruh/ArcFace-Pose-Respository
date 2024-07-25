import cv2 
import shutil 
import os

def create_flip_images(source_folder, destination_folder):
    
    # Copy the folder of images to the destination
    if not os.path.exists(destination_folder):
        shutil.copytree(source_folder, destination_folder)
    
    for root, dirs, files in os.walk(destination_folder):
        for file in files: 
            if '-45' in file.strip('.jpeg').split('_')[-1]:
                file_path = os.path.join(root, file)
                image = cv2.imread(file_path)
                flip_image = cv2.flip(image, flipCode=1)
                
                # Construct the new file and full path for the flipped image
                # new_filename = os.path.splitext(file)[0] + '_flipped' + os.path.splitext(file)[1]
                parts = file.split('_')
                new_filename = '_'.join(parts[:-2]) + '_flipped_' + parts[-2] + '_45.jpeg'
                print(new_filename)
                flipped_image_path = os.path.join(root, new_filename)
                
                # Save the flipped image
                cv2.imwrite(flipped_image_path, flip_image)
            
    
if __name__ == '__main__':
    create_flip_images(source_folder= './data/M2FPA/Train_Bins_Raw_Augment_Experiments/-30_0_-45_0', destination_folder='./data/M2FPA/-30_0_-45_0_45synth')