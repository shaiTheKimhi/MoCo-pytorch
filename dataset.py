from PIL import Image
import pandas as pd
import torch
import os
import torchvision
from torchvision import transforms

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
NUM_CHANNELS = 3


def get_statistics(dataset):
    bs = 1 #batch size
    marr = torch.zeros(5, NUM_CHANNELS) #3 is channels number
    stdarr = torch.zeros(5, NUM_CHANNELS)
    for idx in range(5): #len(dataset)/ bs 
        image = dataset[idx*bs: (idx+1)*bs][0].to(device) #could also calculate throught cuda
        m = torch.tensor([torch.mean(image.permute(1 ,0 , 2, 3)[i]) for i in range(NUM_CHANNELS)])
        std = torch.tensor([torch.std(image.permute(1 ,0 , 2, 3)[i]) for i in range(NUM_CHANNELS)])
        for i in range(NUM_CHANNELS):
            marr[i] += m[i]
            stdarr[i] += std[i]
        #m = torch(
        #std = torch
    return marr
        
           
           
#TODO: add augmentation classes


class ImagenetteDataset(torch.utils.data.Dataset):
    def __init__(self, path, crop_size=224, train=True, augment=True, label_index=1):#TODO: check crop_size, train and augment
        self.crop_size = crop_size
        self.augment = augment
        
        self.labels = label_index
        csv_path = os.path.join(path, 'noisy_imagenette.csv')
        csv_file = pd.read_csv(csv_path)
        self.path = path
        self.images_path = csv_file['path'].values.tolist()
        
    def __getitem__(self, index):
        if type(index) is int:
            image_path = os.path.join(self.path, self.images_path[index])
            label = classes[self.images_path[index].split('/')[1]]
            image = Image.open(image_path).convert('RGB')
            return self.transform(image), label 
        elif type(index) is slice:
            step = index.step if index.step is not None else 1
            stop = index.stop if index.stop is not None else len(self)
            start = index.start if index.start is not None else 0 
            if step == 0:
                raise ZeroDivisionError()
            bs = int((stop - start)/step)
            if bs == 1:
                image_path = os.path.join(self.path, self.images_path[start])
                label = classes[self.images_path[start].split('/')[1]]
                image = Image.open(image_path).convert('RGB')
                return self.transform(image), label 
            else:
                batch = torch.zeros(bs, 3, self.crop_size, self.crop_size) #3 is rgb- num of channels, crop_size could be changed to a tuple right now integer
            labels = torch.zeros(bs)
            for i in range(start, stop, step):
                image_path = os.path.join(self.path, self.images_path[i])
                label = classes[self.images_path[i].split('/')[1]]
                image = Image.open(image_path).convert('RGB')
                batch[i] += self.transform(image)
                labels[i] += label
            return batch, labels       
        
        
        
    def __len__(self):
        return len(self.images_path)
        
    def transform(self, image):
        tran = torchvision.transforms.Compose([transforms.Resize(self.crop_size),
                                                  transforms.CenterCrop(self.crop_size),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                       std=[0.229, 0.224, 0.225])])
        
        return tran(image)
        pass #TODO: implement transformations based on augmentations, normalizations and cropping