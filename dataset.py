from PIL import Image
from albumentations.augmentations.functional import channel_shuffle
import pandas as pd
import torch
import os
import torchvision
from torchvision import transforms
import numpy as np
import torch

import albumentations

from albumentations.augmentations import transforms as augments

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
DATABASE_MEAN =  [0.4661, 0.4581, 0.4292] #Imagenette statistics slightly differ from ImageNet's
DATABASE_STD =  [0.2382, 0.2315, 0.2394]



def get_statistics(dataset):
    bs = 1 #batch size
    n =  len(dataset)
    marr = torch.zeros(n, NUM_CHANNELS) #3 is channels number
    stdarr = torch.zeros(n, NUM_CHANNELS)
    for idx in range(int(n / bs)):  
        image = dataset[idx*bs: (idx+1)*bs][0].to(device) #could also calculate throught cuda
        if bs > 1:
            m = torch.tensor([torch.mean(image.permute(1 ,0 , 2, 3)[i]) for i in range(NUM_CHANNELS)])
            std = torch.tensor([torch.std(image.permute(1 ,0 , 2, 3)[i]) for i in range(NUM_CHANNELS)])
        else:
            m = torch.tensor([torch.mean(image[i]) for i in range(NUM_CHANNELS)])
            std = torch.tensor([torch.std(image[i]) for i in range(NUM_CHANNELS)])
        for i in range(NUM_CHANNELS):
            marr[idx][i] += m[i]
            stdarr[idx][i] += std[i]
        #m = torch(
        #std = torch
    return [torch.mean(marr.T[i]) for i in range(NUM_CHANNELS)], [torch.mean(stdarr.T[i]) for i in range(NUM_CHANNELS)]

   

functions = [augments.Blur, augments.GaussianBlur, augments.ChannelShuffle, augments.ColorJitter, augments.CoarseDropout\
    , augments.Cutout, augments.GaussNoise, augments.Flip, augments.HueSaturationValue, augments.RandomBrightnessContrast,
     augments.RandomFog, augments.RandomShadow, augments.RandomRain, augments.VerticalFlip, augments.Sharpen] #, augments.ChannelDropout, 
     # augments.CLAHE, augments.Equalize, augments.FancyPCA (requires integer type),

#TODO: add augmentation classes
class Augmentations:
    def __init__(self, types = None, normalization = None):
        self.types = types
        self.normalization = normalization
    def augment(self, image, probs=None, k=1):
        '''
        this function returns the image after applying random augmentations
        image - the image to augment
        probs - list of probability factors determining which augmentation to pick (parameters are picked randomly independantly)
        k - number of augmentations to pick and implement
        '''
        transformed = np.ascontiguousarray(image.permute(1,2,0).numpy())
        n = len(functions)
        probs = torch.tensor([1/n for i in range(n)]) if probs is None else probs
        c = torch.distributions.Categorical(probs)
        for i in range(k):
            f = functions[c.sample().item()](always_apply=True) #this uses default parameters for random augmentations
            print(f)
            transformed = f(image = transformed)['image']

            #self.apply(image, c.sample().item())
        return transforms.ToTensor()(transformed)


    def apply(self, image, augmentation_num):
        '''
        applies the given augmentation
        '''
        if augmentation_num == 1: #Blur
            blur_limit = torch.radnint(3,10, (1,)).item()
            f = functions[augmentation_num](blur_limt=blur_limit)
            return f(image)
        elif augmentation_num == 2: #GaussianBlur
            #will use default std (CAN CHANGE THIS)
            blur_limit = (torch.radnint(3,10, (1,)).item(), torch.radnint(3,10, (1,)).item())
            f = functions[augmentation_num](blur_limt=blur_limit) #sigma_limit = torch.rand(1).item() * 5
            return f(image)
        elif augmentation_num == 3: #Channel Dropout
            pass
        pass



            



class ImagenetteDataset(torch.utils.data.Dataset):
    def __init__(self, path, crop_size=224, train=True, augment=2, normalize = True, label_index=1, num_augmentations = 1):#TODO: check crop_size, train and augment
        self.crop_size = crop_size
        self.augment = augment
        self.mean = DATABASE_MEAN if normalize else [0,0,0]
        self.std = DATABASE_STD if normalize else [1,1,1] 
        self.k = num_augmentations

        if augment: #not zero
            self.augmentor = Augmentations() #CAN CHECK: if to crop image before augmentation, right now it will be after
        
        #TODO: split the dataset to train and validation sets, store them in self.train_set, self.valid_set

        self.labels = label_index
        csv_path = os.path.join(path, 'noisy_imagenette.csv')
        csv_file = pd.read_csv(csv_path)
        self.path = path
        self.images_path = csv_file['path'].values.tolist()
        
    def __getitem__(self, index):
        if type(index) is int:
            image_path = os.path.join(self.path, self.images_path[index])
            label = torch.tensor(classes[self.images_path[index].split('/')[1]])
            image = Image.open(image_path).convert('RGB')
            transformed = self.transform(image)
            return transformed[0], transformed[1], label 
        elif type(index) is slice:
            step = index.step if index.step is not None else 1
            stop = index.stop if index.stop is not None else len(self)
            start = index.start if index.start is not None else 0 
            if step == 0:
                raise ZeroDivisionError()
            bs = int((stop - start)/step)
            if bs == 1:
                image_path = os.path.join(self.path, self.images_path[start])
                label = torch.tensor(classes[self.images_path[start].split('/')[1]])
                image = Image.open(image_path).convert('RGB')
                transformed = self.transform(image)
                return transformed[0], transformed[1], label 
            else:
                batch = torch.zeros(2, bs, NUM_CHANNELS, self.crop_size, self.crop_size) #2- two images for a case of augmentation
            labels = torch.zeros(bs)
            for i in range(start, stop, step):
                image_path = os.path.join(self.path, self.images_path[i])
                label = classes[self.images_path[i].split('/')[1]]
                image = Image.open(image_path).convert('RGB')
                transformed = self.transform(image)
                batch[i - start][0] += transformed[0]
                batch[i - start][1] += transformed[1]
                labels[i - start] += label
            return batch[0], batch[1], labels       
        
        
        
    def __len__(self):
        return len(self.images_path)
        
    def transform(self, image): 
        tran = torchvision.transforms.Compose([transforms.Resize(self.crop_size),
                transforms.CenterCrop(self.crop_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std)])
        if self.augment == 0:
            return tran(image), torch.zeros(1)
        img = torchvision.transforms.Compose([transforms.ToTensor()])(image)
        if self.augment == 2:
            q = self.augmentor.augment(img, k=self.k)
            k = self.augmentor.augment(img, k=self.k)
            return tran(q), tran(k)
        elif self.augment == 1:
            return tran(image), tran(transforms.ToPILImage()(self.augmentor.augment(img, k=self.k)))
        

        #transforms.Normalize(mean=[0.485, 0.456, 0.406], ImageNet statistics
        #std=[0.229, 0.224, 0.225])
        #([tensor(0.4661), tensor(0.4581), tensor(0.4292)], [tensor(0.2382), tensor(0.2315), tensor(0.2394)]) imagenette statistics
        
        return tran(image), torch.zeros(1)
