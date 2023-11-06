import pickle
import os
import os.path as osp

dataset_path = 'data/300WLPA_2d/HELEN_train'
dataset_list = []
for id in os.listdir(dataset_path):
    id_path = osp.join(dataset_path, id)
    for img in os.listdir(id_path):
        img_path = osp.join(id_path, img)
        dataset_list.append({
            'id': int(id.split('_')[-1]),
            'img_path': img_path
        })
        
with open('data/pickles/helen_train.pkl', 'wb') as f:
    pickle.dump(dataset_list, f)

        
