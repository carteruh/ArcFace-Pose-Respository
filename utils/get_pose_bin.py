import torch
import numpy as np
import torch
import numpy as np
import shutil
import os
from PIL import Image
import math
from typing import Any
# import models.deepheadpose.code.hopenet
# import models.deepheadpose.code.utils
# from ..models.deepheadpose.code.utils import softmax_temperature
# from ..models.deepheadpose.code.hopenet import Hopenet 


# Get a set of images
def Get_Pose_Bin(folder_path, dest_path, pose_bins):
    
    # Define the ranges for the pitch and yaw pose bins
    # yaw_min, yaw_max = yaw_range
    # pitch_min, pitch_max = pitch_range
    root_dest = dest_path
    print(root_dest)
    
    for pitch_min, pitch_max, yaw_min, yaw_max in pose_bins:
        
        # Make directory if path exists
        dest_path = root_dest + f'{pitch_min}_{pitch_max}_{yaw_min}_{yaw_max}/'
        
        print(dest_path)
        os.makedirs(dest_path, exist_ok= True)
        
        device = torch.device("cuda")
        
        
        for root, dirs, files in os.walk(folder_path, topdown= False):
            if len(dirs) == 0: 
                ID = root.split('/')[-1]
                
                for file in files:
                    if "jpeg" in file:
                        
                        # Take path and extract pose
                        img_path = os.path.join(root, file)
                        filename = file.strip('.jpeg').split('_')
                        # print(filename)
                        pitch, yaw = filename[-2:]
                        pitch, yaw = float(pitch), float(yaw)                      
                                                                
                        # Check if image is within desired bin range 
                        # if (yaw >= yaw_min) and (yaw <= yaw_max) and (pitch >= pitch_min) and (pitch <= pitch_max):
                        if yaw in  [0, 45, -45] and (pitch >= pitch_min) and (pitch <= pitch_max):
                            file_dest_path = os.path.join(dest_path, ID)
                            file_source_path = os.path.join(root, file)
                            #print(f'Pitch: {pitch}, Yaw: {yaw}')
                            
                            os.makedirs(file_dest_path, exist_ok= True)
                            shutil.copy(file_source_path, file_dest_path)

if __name__ == '__main__':        
    pose_bins = [(-30, 30, -15, 15), (-30, 30, -15, 15), (-30, 30, -45, -15), 
                 (-30, 30, -45, -15), (-30, 30, 15, 45), (-30, 30, 15, 45)
                 ,(-30, 30, -70, -45), (-30, 30, -70, -45),(-30, 30, 45, 70), (-30, 30, 45, 70)
                 ,(-30, 30, -90,-70), (-30, 30, -90,-70),(-30, 30, 70, 90), (-30, 30, 70, 90)]
    # pose_bins = [(30, 30, -45, 45), (-30, -30, -45, 45)]
    
    Get_Pose_Bin(folder_path= "./data/M2FPA/Test", dest_path= "./data/M2FPA/Test_Bins_all_pitch/", pose_bins= pose_bins)