import torch
import numpy as np
import math
import sklearn
import torchvision
import sklearn
import torch
import torchvision
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
                        pitch, yaw = filename[2:]
                        pitch, yaw = float(pitch), float(yaw)                      
                                                                
                        # Check if image is within desired bin range 
                        if (yaw >= yaw_min) and (yaw <= yaw_max) and (pitch >= pitch_min) and (pitch <= pitch_max):
                            file_dest_path = os.path.join(dest_path, ID)
                            file_source_path = os.path.join(root, file)
                            #print(f'Pitch: {pitch}, Yaw: {yaw}')
                            
                            os.makedirs(file_dest_path, exist_ok= True)
                            shutil.copy(file_source_path, file_dest_path)

if __name__ == '__main__':
    # pose_bins = [(-45, 15, -30, 30), (-100, -45, -30, 30), (-45, 15, -60, -30), 
    #              (-100, -45, -60, -30), (-45, 15, 30, 60), (-100, -45, 30, 60), 
    #              (-45, 15, 60, 90), (-100, -45, 60, 90), (-45, 15, -90, -60), 
    #              (-100, -45, -90, -60)]
    
    # pose_bins = [(-20, 50, -30, 30), (-50, -20, -30, 30), (-20, 50, -60, -30), 
    #              (-50, -20, -60, -30), (-20, 50, 30, 60), (-50, -20, 30, 60)
    #              ,(-20, 50, -90, -60), (-50, -20, -90, -60),(-20, 50, 60, 90), (-50, -20, 60, 90)]
    
        
    pose_bins = [(0, 50, -15, 15), (-50, 0, -15, 15), (0, 50, -45, -15), 
                 (-50, 0, -45, -15), (0, 50, 15, 45), (-50, 0, 15, 45)
                 ,(0, 50, -70, -45), (-50, 0, -70, -45),(0, 50, 45, 70), (-50, 0, 45, 70)
                 ,(0, 50, -90,-70), (-50, 0, -90,-70),(0, 50, 70, 90), (-50, 0, 70, 90)]
    
    Get_Pose_Bin(folder_path= "./data/M2FPA/Train_detected", dest_path= "./data/M2FPA/Train_Bins/", pose_bins= pose_bins)