from torch.utils.data import Dataset, DataLoader
import pickle
import torch 
from PIL import Image

class Image_Dataset(Dataset):
    def __init__(self, data_path, transform) -> None:
        super().__init__()
        self.data_path = data_path
        self.transform = transform
        self.data_list = self._get_data(self.data_path)
        
    def _get_data(self, data_path):
        with open(data_path, 'rb') as f:
            data_list = pickle.load(f)
        return data_list
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        sample = self.data_list[index]
        # sample is now a dict
        label = sample['id']
        img_path = sample['img_path']
        
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        return (img, label)
    

    
        