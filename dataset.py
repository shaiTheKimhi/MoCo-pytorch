from PIL import Image
import pandas as pd
import torch
import os

classes = {'n01440764': 0,
           'n02102040': 1,
           'n02979186': 2,
           'n03000684': 3,
           'n03028079': 4,
           'n03394916': 5,
           'n03417042': 6,
           'n03425413': 7,
           'n03445777': 8,
           'n03888257': 9}
#TODO: add augmentation classes


class ImagenetteDataset(torch.utils.data.Dataset):
    def __init__(self, path, crop_size=224, train=True, augment=True, label_index=1):#TODO: check crop_size, train and augment
        self.labels = label_index
        csv_path = os.path.join(path, 'noisy_imagenette.csv')
        csv_file = pd.read_csv(csv_path)
        self.path = path
        self.images_path = df_csv['path'].values.tolist()
        
    def __getitem__(self, index):
        image_path = os.path.join(self.path, self.images_path[index])
        label = classes[self.images_path[index].split('/')[1]]
        #TODO: return image and label
        
    def __len__(self):
        return len(self.images_path)