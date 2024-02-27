import torch
import torchvision
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import torch
import torchvision
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torchvision.models as model
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torch.nn as nn
from PIL import Image
import time
from models.deepheadpose.code.hopenet import Hopenet
from models.deepheadpose.code.utils import softmax_temperature
from typing import Any
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from pytorch_lightning.utilities.types import STEP_OUTPUT
from config.config import get_config
from models.Focal_Loss import FocalLoss
from models.ArcFace import ArcFaceLoss
from models.resnet50 import iresnet50
from torchsummary import summary
from visualize_loss import visualize_loss
from datasets.dataset import Image_Dataset
import os

def train_bins(file_path, target_bin= None):
    # Get personal configuration
    cfg = get_config()
    # data_path = 'data/pickles/helen_train.pkl'
    # file_path = './data/300WLPA_2d/HELEN_train_bins_merged'
    bin_name = file_path.split('/')[-1]
    
    if target_bin:
        pose_bin_list = target_bin
    else:
        pose_bin_list = [pose_bin for pose_bin in os.listdir(file_path)]
    
    # for pose_bin in os.listdir(file_path):
    for pose_bin in pose_bin_list:
        
        # Initialize Device
        device = torch.device("cuda")
        
        # transform into tensors
        # transform_train = transforms.Compose([
        #     transforms.RandomHorizontalFlip(),
        #     transforms.Resize((112, 112)),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # ])
        
        transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # ])
        
        # Initialize Image Dataset Class
        train_bin = torchvision.datasets.ImageFolder(root= f'./data/M2FPA/{bin_name}/{pose_bin}', transform= transform)
        # train_bin = Image_Dataset(data_path, transform)
        dataloader = DataLoader(dataset=train_bin, batch_size=256, shuffle=True, num_workers=4)
        num_classes = len(set(train_bin))
        print(num_classes)
        
        # dataloader = DataLoader(train_bin, batch_size= 128, shuffle= True, num_workers=2, collate_fn= None, pin_memory= True)

        # Initialize resnet backbone
        if cfg.backbone == "r50":
            models = torchvision.models.resnet50(pretrained=True)
            for param in models.parameters():
                param.requires_grad_(False)
            models.fc= nn.Linear(2048, 512)
            models.fc.requires_grad_(True)
            models.to(device)
        elif cfg.backbone == "ir50":
            models = iresnet50(pretrained= True)
            for param in models.parameters():
                param.requires_grad_(False)
            models.fc.requires_grad_(True)
            models.to(device)
            # models.load_state_dict(torch.load('./models/weights/resnet50_weights_AFW_250_epochs_all_poses.pth', map_location= device))

        # Display backbone model 
        summary(models, (3, 112, 112))
        
        # Focal loss is an improved method of Cross Entropy that can focus on scaling the harder cases then the easier cases
        Loss = FocalLoss(gamma= 2).to(device)
        Loss.requires_grad = True

        metric_fc = ArcFaceLoss(512, num_classes, 64.0, 0.5)
        metric_fc.requires_grad = True
        metric_fc.to(device)
        # metric_fc.load_state_dict(torch.load('./models/weights/arcface_weights_AFW_250_epochs_all_poses.pth'))

        # Setup Optimizer and Scheduler
        optimizer = torch.optim.SGD([{'params': models.parameters()},
                                    {'params': metric_fc.parameters()}],
                                    lr = 1e-1, weight_decay= cfg.weight_decay, momentum= cfg.momentum)  # Can use Adam or SGD
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones= [10, 40, 120, 170], gamma= 0.1)

        # Train Model with Provided ArcFace 
        start = time.time()
        history = []
        best_acc = 0.0


        for epoch in range(10):
            epoch_start = time.time()
            print("Epoch: {}/{}".format(epoch + 1, 10))

            # Set model to train mode
            models.train()
            metric_fc.train()
            
            # Step through scheduler
            scheduler.step()

            train_loss = 0.0
            train_acc = 0.0 

            valid_loss = 0.0
            valid_acc = 0.0
            
            if epoch % 50 == 0 and epoch != 0:
                # Save Weights from metric_fc and resnet50
                torch.save(models.state_dict(), f'./models/weights/weights_M2FPA_pose_bin/resnet50_weights_M2FPA_Raw_10_epochs_{pose_bin}.pth')

            for i, (images, labels) in enumerate(dataloader):
                
                # Put tensors into device to run CUDA
                images, labels = images.to(device), labels.to(device)
                
                # Run Resnet50 on images to compute the embedding of size 512
                embeddings = models(images)

                # Push this embedding to our Arcface fc output layer
                output = metric_fc(embeddings, labels)
                
                # Calculate loss
                loss = Loss(output, labels)

                # Clean existing gradients
                optimizer.zero_grad()
                
                # Back propogate the gradients
                loss.backward()

                # Update Parameters after back propogation
                optimizer.step()
                
                # Accumulate loss
                train_loss += loss.item()

                # Accuracy prediction
                _, predictions = torch.max(output.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))

                # Convert correct_counts to float then compute the mean
                acc = torch.mean(correct_counts.type(torch.FloatTensor))
        
                # Compute the total accuracy in the whole batch and add to train_acc
                train_acc += acc.item()
                if (i % 100 == 0):
                    print('Batch #: [{:03d}/{}], Train Loss: {}, Train Accuracy: {}'
                        .format(i+1, len(dataloader), train_loss / (i+1), train_acc / (i + 1)))
            

            history.append({'loss': train_loss / len(dataloader), 'acc': train_acc / len(dataloader)})

        # Save Weights from metric_fc and resnet50
        torch.save(models.state_dict(), f'./models/weights/weights_M2FPA_pose_bin/resnet50_weights_M2FPA_Raw_10_epochs_{pose_bin}.pth')
        print("Saved")
        
        visualize_loss(history= history, bin= pose_bin)
    
if __name__ == '__main__':
    file_path = './data/M2FPA/Train_Bins_Raw'
    train_bins(file_path= file_path, target_bin= ['-30_-30_-45_0_45', '30_30_-45_0_45'])
