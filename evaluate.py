import torch
import numpy
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
from models.deepheadpose.code.hopenet import Hopenet
from models.deepheadpose.code.utils import softmax_temperature
from typing import Any
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from pytorch_lightning.utilities.types import STEP_OUTPUT
from pytorch_lightning.tuner import Tuner
from torch import Tensor
from config.config import get_config
from models.Focal_Loss import FocalLoss
from models.ArcFace import ArcFaceLoss
from models.resnet50 import iresnet50
from torchsummary import summary

if __name__ == '__main__':
    device = torch.device("cuda")
    
    transform_pose = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    LPA_D = torchvision.datasets.ImageFolder(root= "./data/M2FPA/Test", transform= transform_pose)
    num_classes = len(set(LPA_D.classes))
    
    # Estimate the Pitch and Yaw of every image in the MAPIR Dataset
    pitch_list = []
    yaw_list = []

    for idx, (filename, label) in enumerate(LPA_D.imgs):
        filename = filename.strip('.jpeg').split('/')[-1].split('_')
        print(filename)
        pitch, yaw = filename[4:]

        pitch_list.append(float(pitch))
        yaw_list.append(float(yaw))

        print("Pitch: {}, Yaw: {}".format(pitch, yaw))
        
    # Create a figure and axes for the plot
    fig, ax = plt.subplots()

    # Create histograms
    ax.hist(pitch_list, bins=50, alpha=0.5, label='Pitch')
    ax.hist(yaw_list, bins=50, alpha=0.5, label='yaw')

    # Set labels and title
    ax.set_xlabel('Angles')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Pitch and Yaw Angles')

    # Add a legend
    ax.legend()

    # Display the plot
    plt.show()
    plt.savefig('./data/plot_images/plot_figure.jpg', bbox_inches='tight')
    