import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import pickle
import os
import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline
from models.resnet50 import iresnet50 
from config.config import get_config

"""
Conduct model performance evaluation of probe images from probe sets against gallery sets. We initialize the 
appropriate pose sets collected in data and test distinct pose sets from the gallery against a query set 
composed of all poses
"""

def get_rank_k(query_list, gallery_list, model_weight_path, query_set_bin, gallery_set_bin):
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Initialize model with weights
    model = iresnet50(pretrained=True)
    model.to("cuda")  # Move model to the appropriate device before loading state dict
    model.load_state_dict(torch.load(model_weight_path, map_location='cuda'))
    model.eval()  # Set the model to evaluation mode

    query_set = {"embedding": [], "label": []}
    gallery_set = {"embedding": [], "label": []}

    # Generate embeddings and obtain labels for the query and gallery sets
    with torch.inference_mode():
        for image_path in gallery_list:
            img = transform(Image.open(image_path)).unsqueeze(0).to("cuda")
            embedding = model(img)
            gallery_set['embedding'].append(embedding)
            label = image_path.split('/')[-2]
            gallery_set['label'].append(int(label))  # Cast label to int assuming it's an integer ID

        for image_path in query_list:
            img = transform(Image.open(image_path)).unsqueeze(0).to("cuda")
            embedding = model(img)
            query_set['embedding'].append(embedding)
            label = image_path.split('/')[-2]
            query_set['label'].append(int(label))  # Cast label to int assuming it's an integer ID

        gallery_set["embedding"] = torch.stack(gallery_set["embedding"]).squeeze()
        gallery_set["label"] = torch.tensor(gallery_set["label"], device="cuda")  # Labels as tensor

        correct_counts = [0.0] * len(gallery_set["label"])
        total_counts = [0.0] * len(gallery_set["label"])
        top_k_accuracy = [0.0] * len(gallery_set["label"])

        for probe_idx in range(len(query_set['embedding'])):
            probe_emb = query_set['embedding'][probe_idx]
            probe_label = query_set['label'][probe_idx]
            cos_similarity = F.cosine_similarity(probe_emb, gallery_set['embedding'], dim=-1)

            # Iterate over all values of k
            for k in range(1, len(gallery_set["label"]) + 1):
                _, pred = torch.topk(cos_similarity, k)
                correct_counts[k-1] += int(any(probe_label == gallery_set['label'][i] for i in pred.to("cpu").numpy().tolist()[:k]))
                if k == 1:
                    is_correct = int(any(probe_label == gallery_set['label'][i] for i in pred.to("cpu").numpy().tolist()[:k]))
                    gallery_pred = gallery_set['label'][pred[0]]
                    # print(f'Top K: {k}, Probe Label: {probe_label}, Prediction: {gallery_pred}, is correct: {is_correct}')
                    # if probe_label != gallery_pred:
                    probe_file = query_list[probe_idx].split('/')[-1]
                    gallery_file = gallery_list[pred[0]].split('/')[-1]
                    print(f'Probe Path: {probe_file}, Gallery Path: {gallery_file}, correct: {is_correct}')
                total_counts[k-1] += 1
                top_k_accuracy[k-1] = correct_counts[k-1] / total_counts[k-1]


    # Plot the accuracy vs top-k rank
    print(f"Query Set: {query_set_bin}, Gallery Set: {gallery_set_bin}, Rank 1 Accuracy: {top_k_accuracy[0]}, Rank 5 Accuracy: {top_k_accuracy[4]}")
    
    return top_k_accuracy


def visualize_pose_groups(query_bin_list: list,  # The pose bins that are evaluated in the query set 
    gallery_bin_list: list,  # The pose bins that are evaluated in the gallery
    query_gallery_set_file_name: str= "query_galleries_M2FPA_Bins_Raw", # The pickle file name for the query and gallery file lists
    fig_name: str= 'top_k_accuracies_M2FPA_Raw', # The figure name for the Top-1 accuracies across multiple yaw ranges
    table_name: str= 'accuracies_table', # The table name for all the top 1 and top 5 accuracies of the pose group
    image_type: str = "Raw",  #  Describe the image type that is being trained on
    labels: list = ['-90_-70', '-70_-45','-45_-15','-15_15', '15_45', '45_70', '70_90'],
    model_weights: list = ['./models/weights/weights_M2FPA_pose_bin/resnet50_weights_M2FPA_Raw_all_pitch_10_epochs_-30_30_-45_0_45.pth'] 
    ):
    
    # Open and load the .pkl file
    with open(f'./test_sets/{query_gallery_set_file_name}.pkl', 'rb') as f:
        data = pickle.load(f)
        
    results = {}

    # Initialize figure and figure configurations
    plt.figure()
    plt.xticks(range(len(gallery_bin_list)),labels= labels)
    plt.ylim([0, 0.7])
    plt.xlabel("Yaw Pose in Gallery")
    plt.ylabel("Rank-1 Accuracy")
    plt.tight_layout(rect=[0, 0, 0.75, 1])
    plt.subplots_adjust(right=1.3)
    
    # iterate through query list and gallery list
    for query_bin in query_bin_list:
        accuracies = []
        top_k_list = []
        for gallery_bin in gallery_bin_list:   
            query_set_bin = f'{query_bin}'
            gallery_set_bin = f'{gallery_bin}'

            query_set_path = data[f'{query_bin}']['query']
            gallery_set_path = data[f'{gallery_bin}']['gallery']
            
            # model_weight_path = f'./models/weights/weights_M2FPA_pose_bin/resnet50_weights_M2FPA_{image_type}_10_epochs_{query_set_bin}.pth'
            # model_weight_path = f'./models/weights/weights_M2FPA_pose_bin/resnet50_weights_M2FPA_Raw_all_pitch_10_epochs_-30_30_-45_0_45.pth'
            model_weight_path = model_weights[gallery_bin]
            print(f'Loading model: {model_weight_path}')
            
            top_k_accuracies = get_rank_k(query_list=query_set_path, gallery_list=gallery_set_path, model_weight_path=model_weight_path, query_set_bin=query_set_bin, gallery_set_bin=gallery_set_bin)
            top_k_list.append(top_k_accuracies[0])
            accuracies.append(f"{top_k_accuracies[0]:.4f}")

        # Smoothen out the data using spline interpolation
        # y_spline_vals = make_interp_spline(range(len(gallery_bin_list)), top_k_list)
        # x_vals = np.linspace(0, len(gallery_bin_list) - 1, 300)
        # y_vals = y_spline_vals(x_vals)
        # plt.plot(x_vals, y_vals)
        
        plt.plot(range(len(gallery_bin_list)), top_k_list)
        
        results[f'Query {query_bin}'] = accuracies
        
        # Compute the average top-1 accuracy for the query set and append to end of the list
        # average_top_1 = np.mean([float(acc.split(",")[0].split(": ")[1]) for acc in accuracies]) 
        average_top_1 = np.mean([float(acc) for acc in accuracies]) 
        results[f'Query {query_bin}'].append(f'{average_top_1:.5f}')

        
    plt.legend(["Probe Set: -90° to 90°"], bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=3, title= 'Yaw Groups')
    plt.savefig(f'./data/plot_images/Pose_Bin_Visualizations/{fig_name}.jpg', bbox_inches= 'tight')
    plt.close()
    
    results_df = pd.DataFrame(results, index= [f'Gallery {bin}' for bin in gallery_bin_list] + ['Average'])

    results_df.to_csv(f'./data/table_metrics/{table_name}.csv')
    print(results_df)


if __name__ == '__main__':
    os.makedirs('./data/plot_images/CMC_Curves/M2FPA', exist_ok=True)  # Ensure the directory exists
    
    # Get configurations for testing
    cfg = get_config()
    
    # Define the bin lists for both pitch groups  
    query_bin_list = ['-30_30_-90_90']  # Query bin represents probe images of random poses across the whole pose range

    gallery_bin_list_low = ['-30_0_-90_-70', '-30_0_-70_-45','-30_0_-45_-15',
                          '-30_0_-15_15', '-30_0_15_45', '-30_0_45_70', '-30_0_70_90']
    gallery_bin_list_high = ['0_30_-90_-70', '0_30_-70_-45','0_30_-45_-15',
                          '0_30_-15_15', '0_30_15_45', '0_30_45_70', '0_30_70_90']
    
    gallery_bin_list = ['0_0_0_0', '0_0_15_15', '0_0_30_30', '0_0_45_45', '0_0_75_75', '0_0_90_90']
    
    # labels= ['-90_-70', '-70_-45','-45_-15',
            #  '-15_15', '15_45', '45_70', '70_90']
            
    model_weights = {'0_0_0_0':'./models/weights/weights_M2FPA_pose_bin/resnet50_weights_M2FPA_yaw_degradation_10_epochs_0_0_0_0.pth',
                     '0_0_15_15': './models/weights/weights_M2FPA_pose_bin/resnet50_weights_M2FPA_yaw_degradation_10_epochs_0_0_15_15.pth',
                     '0_0_30_30': './models/weights/weights_M2FPA_pose_bin/resnet50_weights_M2FPA_yaw_degradation_10_epochs_0_0_30_30.pth',
                     '0_0_45_45': './models/weights/weights_M2FPA_pose_bin/resnet50_weights_M2FPA_yaw_degradation_10_epochs_0_0_45_45.pth',
                     '0_0_75_75': './models/weights/weights_M2FPA_pose_bin/resnet50_weights_M2FPA_yaw_degradation_10_epochs_0_0_75_75.pth',
                     '0_0_90_90': './models/weights/weights_M2FPA_pose_bin/resnet50_weights_M2FPA_yaw_degradation_10_epochs_0_0_90_90.pth'
                     }
    labels = ['0°', '15°', '30°', '45°', '75°', '90°']

    
    # Evaluate the probe and galleries
    visualize_pose_groups(query_bin_list= query_bin_list, 
                          gallery_bin_list= gallery_bin_list, 
                          query_gallery_set_file_name= 'query_galleries_M2FPA_Bins_yaw_degradation', 
                          fig_name= 'top_k_accuracies_M2FPA_Raw_query_pitch_-30_30_yaw_all_gallery_pitch_-30_0_yaw_-45_0_45_image_reduction_experiment',
                          table_name= "accuracies_table_align_query_pitch_-30_30_yaw_all_gallery_pitch_-30_0_yaw_-45_0_45_image_reduction_experiment",
                          image_type= "cropped",
                          labels= labels,
                          model_weights= model_weights
                          )
