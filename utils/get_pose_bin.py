import torch
import numpy as np
import math
import sklearn
import torchvision
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import torch
import torchvision
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torchvision.models as model
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import tarfile
import shutil
from torch.nn import DataParallel
from pathlib import Path
from PIL import Image
import cv2
import os
from retinaface import RetinaFace
from deepface import DeepFace
import torch.nn as nn
from PIL import Image
import math
import time
import pytorch_lightning as L
from typing import Any
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from pytorch_lightning.utilities.types import STEP_OUTPUT
from pytorch_lightning.tuner import Tuner
from torch import Tensor


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
                    if "jpg" in file:
                        # Open image path and convert to tensor 
                        img_path = os.path.join(root, file)
                        filename = file.strip('.jpg').split('/')[-1].split('_')
                        # print(filename)
                        pitch, yaw, roll = filename[4:len(filename)]
                        
                        pitch = math.degrees(float(pitch) )
                        yaw = math.degrees(float(yaw))
                        roll = math.degrees(float(roll))
                        
                        # Check if image is within desired bin range 
                        if (yaw > yaw_min) and (yaw < yaw_max) and (pitch > pitch_min) and (pitch < pitch_max):
                            file_dest_path = os.path.join(dest_path, ID)
                            file_source_path = os.path.join(root, file)
                            print("Yaw: {}".format(yaw))
                            print("Pitch: {}".format(pitch))
                            
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
    
        
    pose_bins = [(-20, 50, -15, 15), (-50, -20, -15, 15), (-20, 50, -45, -15), 
                 (-50, -20, -45, -15), (-20, 50, 15, 45), (-50, -20, 15, 45)
                 ,(-20, 50, -70, -45), (-50, -20, -70, -45),(-20, 50, 45, 70), (-50, -20, 45, 70)
                 ,(-20, 50, -90,-70), (-50, -20, -90,-70),(-20, 50, 70, 90), (-50, -20, 70, 90)]
    
    Get_Pose_Bin(folder_path= "./data/300WLPA_2d/HELEN_test/", dest_path= "./data/300WLPA_2d/HELEN_test_bins_rad2degrees/", pose_bins= pose_bins)