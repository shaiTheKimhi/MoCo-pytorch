from PIL import Image
import pandas as pd
import torch
import os
import torchvision
from torchvision import transforms
import numpy as np
import torch
import random

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

#Imagenette statistics slightly differ from ImageNet's
DATABASE_MEAN = [0.4661, 0.4581, 0.4292]
DATABASE_STD = [0.2382, 0.2315, 0.2394]


def get_statistics(dataset):
    """
    normelize the DS by the statistics (sslighly diffrent then imagenet)
    :param dataset:
    :return:
    """
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




#todo: delete
class create_GaussianBlur():
    def __init__(self, k_size, sigma_blur, channels):
        self.k_size = k_size
        self.sigma_blur = sigma_blur
        self.channels = channels

        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_cord = torch.arange(k_size)
        x_grid = x_cord.repeat(k_size).view(k_size, k_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1)

        mean = (k_size - 1) / 2.
        variance = sigma_blur ** 2.

        pi = 3.14
        gaussian_kernel = (1. / (2. * pi * variance)) * torch.exp(
            -torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance))
        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # Reshape to 2d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.view(1, 1, k_size, k_size)
        gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

        gaussian_filter = torch.nn.Conv2d(in_channels=channels, out_channels=channels,
                                          kernel_size=k_size, groups=channels, bias=False)

        gaussian_filter.weight.data = gaussian_kernel
        gaussian_filter.weight.requires_grad = False
        self.gaussian_filter = gaussian_filter

    def __call__(self, img):
        to_PIL = transforms.ToPILImage()
        to_tensor = transforms.ToTensor()

        img_tensor = to_tensor(img)
        if len(img_tensor.shape) == 2:  # if gray turn to rgb
            img_tensor = img_tensor.unsqueeze_(0).repeat(1, 1, 3)
        img_tensor = img_tensor.unsqueeze(0)
        img_tensor_filtered = self.gaussian_filter(img_tensor)
        img_tensor_filtered = img_tensor_filtered.squeeze()
        img_PIL = to_PIL(img_tensor_filtered)
        return img_PIL


#todo: delete
def augment_images(img, crop_width):
    sigma_boundries = [0.1, 2]
    sigma = random.uniform(sigma_boundries[0], sigma_boundries[1])
    kernel_size = 5

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    augment_transformation = transforms.Compose([
        transforms.RandomResizedCrop(crop_width, scale=(0.2, 1.)),
        transforms.RandomGrayscale(p=0.2),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
        transforms.RandomHorizontalFlip(),
        create_GaussianBlur(kernel_size, sigma, 3),
        transforms.ToTensor(),
        normalize, ])

    augmented_image = augment_transformation(img)
    return augmented_image



class ImagenetteDataset(torch.utils.data.Dataset):
    def __init__(self, path, crop_size=299, train=True, augment=2, normalize=True, num_augmentations=1):
        """
        create dataset
        :param path: dataset path
        :param crop_size: like optimal for imagenet, can also be 224 for other models
        :param train: True for train DS
        :param augment:0/1/2
        :param normalize: norm with ds statistics
        :param num_augmentations:
        """
        self.train=train
        self.crop_size = crop_size
        self.augment = augment
        self.mean = DATABASE_MEAN if normalize else [0,0,0]
        self.std = DATABASE_STD if normalize else [1,1,1] 
        self.k = num_augmentations
        if augment: #not zero
            self.augmentor = Augmentations()
        self.classes = [i for i in range(10)]

        #load the dataset from dir
        csv_path = os.path.join(path, 'noisy_imagenette.csv')
        csv_file = pd.read_csv(csv_path)
        self.path = path
        all_paths = csv_file['path'].values.tolist()

        if self.train:
            im_path = [cur_path for cur_path in all_paths if cur_path.split('/')[0] == 'train']
        else:
            im_path = [cur_path for cur_path in all_paths if cur_path.split('/')[0] == 'val']
        self.im_path = im_path
        self.labels = [classes[self.path_list[i].split('/')[1]] for i in range(len(self.im_path))]

    def __getitem__(self, index):

        image_path = os.path.join(self.path, self.im_path[index])
        label = torch.tensor(classes[self.im_path[index].split('/')[1]])
        #label = self.labels[i]
        image = Image.open(image_path).convert('RGB')

        #todo: use shai self.transform()
        if self.augment:
            q_batch = augment_images(image, self.crop_size)
            k_batch = augment_images(image, self.crop_size)
            return q_batch, k_batch, label
        else:
            image_transform = transforms.Compose([transforms.Resize(self.crop_size),
                                                  transforms.CenterCrop(self.crop_size),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(mean=DATABASE_MEAN,
                                                                       std=DATABASE_STD)])
            return image_transform(image), np.zeros((1)), label
        #transformed = self.transform(image)
        #return transformed[0], transformed[1], label

    def __len__(self):
        return len(self.im_path)
        
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