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
                    print(f'Top K: {k}, Probe Label: {probe_label}, Prediction: {gallery_pred}, is correct: {is_correct}')
                total_counts[k-1] += 1
                top_k_accuracy[k-1] = correct_counts[k-1] / total_counts[k-1]


    # Plot the accuracy vs top-k rank
    print(f"Query Set: {query_set_bin}, Gallery Set: {gallery_set_bin}, Rank 1 Accuracy: {top_k_accuracy[0]}, Rank 5 Accuracy: {top_k_accuracy[4]}")
    
    return top_k_accuracy

'''
def visualize_pose_groups(query_bin_list, gallery_bin_list, fig_name, table_name):
    
     # Open and load the .pkl file
    with open('./test_sets/query_galleries_M2FPA_Bins_Raw.pkl', 'rb') as f:
        data = pickle.load(f)
        
    results = {}

    # Initialize figure and figure configurations
    plt.figure()
    plt.xticks(range(len(gallery_bin_list)),labels= ['-90_-70', '-70_-45','-45_-15',
                          '-15_15', '15_45', '45_70', '70_90'])
    plt.xlabel("Yaw Pose Groups in Gallery")
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
            
            model_weight_path = f'./models/weights/weights_M2FPA_pose_bin/resnet50_weights_M2FPA_Raw_10_epochs_{query_set_bin}.pth'
            
            top_k_accuracies = get_rank_k(query_list=query_set_path, gallery_list=gallery_set_path, model_weight_path=model_weight_path, query_set_bin=query_set_bin, gallery_set_bin=gallery_set_bin)
            top_k_list.append(top_k_accuracies[0])
            accuracies.append(f"Rank 1: {top_k_accuracies[0]:.4f}, Rank 5: {top_k_accuracies[4]:.4f}")

        y_spline_vals = make_interp_spline(range(len(gallery_bin_list)), top_k_list)
        x_vals = np.linspace(0, len(gallery_bin_list) - 1, 300)
        y_vals = y_spline_vals(x_vals)
        plt.plot(x_vals, y_vals)
        results[f'Query {query_bin}'] = accuracies
        
    plt.legend(query_bin_list_high, bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=3, title= 'Yaw Groups')
    plt.savefig(fig_name, bbox_inches= 'tight')
    plt.close()

    results_df = pd.DataFrame(results, index=[f'Gallery {bin}' for bin in gallery_bin_list_high])

    results_df.to_csv(table_name)
    print(results_df)
'''

if __name__ == '__main__':
    os.makedirs('./data/plot_images/CMC_Curves/M2FPA', exist_ok=True)  # Ensure the directory exists
    
    # Get configurations for testing
    cfg = get_config()

    # Open and load the .pkl file
    with open('./test_sets/query_galleries_M2FPA_Bins_Raw.pkl', 'rb') as f:
        data = pickle.load(f)
        
    # Define the bin lists for both pitch groups
    query_bin_list_low = ['-30_0_-90_-70', '-30_0_-70_-45','-30_0_-45_-15',
                          '-30_0_-15_15', '-30_0_15_45', '-30_0_45_70', '-30_0_70_90']
    
    # query_bin_list_high = ['0_30_-90_-70', '0_30_-70_-45','0_30_-45_-15',
    #                       '0_30_-15_15', '0_30_15_45', '0_30_45_70', '0_30_70_90']
    
    query_bin_list_low = ['-30_0_+-15_+-45']
    
    query_bin_list_high = ['0_30_+-15_+-45']
    

    gallery_bin_list_low = ['-30_0_-90_-70', '-30_0_-70_-45','-30_0_-45_-15',
                          '-30_0_-15_15', '-30_0_15_45', '-30_0_45_70', '-30_0_70_90']
    
    # gallery_bin_list_low = ['-50_0_-90_-70', '-50_0_-70_-45','-50_0_-45_-15',
    #                       '-50_0_-15_15', '-50_0_15_45', '-50_0_45_70', '-50_0_70_90']
    
    gallery_bin_list_high = ['0_30_-90_-70', '0_30_-70_-45','0_30_-45_-15',
                          '0_30_-15_15', '0_30_15_45', '0_30_45_70', '0_30_70_90']
 
    results = {}

    # Initialize figure and figure configurations
    plt.figure()
    plt.xticks(range(len(gallery_bin_list_low)),labels= ['-90_-70', '-70_-45','-45_-15',
                          '-15_15', '15_45', '45_70', '70_90'])
    plt.ylim([0, 1.1])
    plt.xlabel("Yaw Pose Groups in Gallery")
    plt.ylabel("Rank-1 Accuracy")
    plt.tight_layout(rect=[0, 0, 0.75, 1])
    plt.subplots_adjust(right=1.3)
    
    # iterate through query list and gallery list
    for query_bin in query_bin_list_low:
        accuracies = []
        top_k_list = []
        for gallery_bin in gallery_bin_list_low:   
            query_set_bin = f'{query_bin}'
            gallery_set_bin = f'{gallery_bin}'

            query_set_path = data[f'{query_bin}']['query']
            gallery_set_path = data[f'{gallery_bin}']['gallery']
            
            model_weight_path = f'./models/weights/weights_M2FPA_pose_bin/resnet50_weights_M2FPA_Raw_10_epochs_{query_set_bin}.pth'
            
            top_k_accuracies = get_rank_k(query_list=query_set_path, gallery_list=gallery_set_path, model_weight_path=model_weight_path, query_set_bin=query_set_bin, gallery_set_bin=gallery_set_bin)
            top_k_list.append(top_k_accuracies[0])
            # print(f'Rank 1 Accuracy for Pose {gallery_bin}: {top_k_accuracies[0]}')
            accuracies.append(f"Rank 1: {top_k_accuracies[0]:.4f}, Rank 5: {top_k_accuracies[4]:.4f}")

        y_spline_vals = make_interp_spline(range(len(gallery_bin_list_low)), top_k_list)
        x_vals = np.linspace(0, len(gallery_bin_list_low) - 1, 300)
        y_vals = y_spline_vals(x_vals)
        plt.plot(x_vals, y_vals)
        results[f'Query {query_bin}'] = accuracies
        
    plt.legend(query_bin_list_low, bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=3, title= 'Yaw Groups')
    plt.savefig(f'./data/plot_images/Pose_Bin_Visualizations/top_k_accuracies_M2FPA_Raw_pitch_-30_0_Merged_+-15_+-45.jpg', bbox_inches= 'tight')
    plt.close()

    results_df = pd.DataFrame(results, index=[f'Gallery {bin}' for bin in gallery_bin_list_low])

    results_df.to_csv('accuracies_table_pitch_-30_0_M2FPA_Raw_Merged_+-15_+-45.csv')
    print(results_df)
