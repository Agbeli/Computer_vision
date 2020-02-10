# Coding: utf-8
import os
import glob 
import numpy as np 
import pandas as pd 
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.utils.data.dataloader 
from Folder import ImageFolder 
from torchvision import datasets, transforms


#train_path = "Data/train"
#test_path = "Data/test"


### The function to obtain the mean and standard deviation of the images 

###### The entire mean and standard deviation of the images.
#[0.45615122 0.47518504 0.3208348 ]
#[0.21813191 0.20548095 0.20645605]

Transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

data = datasets.ImageFolder("Data/train",transform=Transform)
dataloader = torch.utils.data.DataLoader(data,batch_size=64, shuffle=False, num_workers=4)

data_mean = [] # Mean of the dataset
data_std0 = [] # std of dataset
for images,labels in dataloader:

    # shape (batch_size, 3, height, width)
    numpy_image = np.array(images)

    # shape (3,)
    batch_mean = numpy_image.mean(axis=(0,2,3))
    batch_std0 = numpy_image.std(axis=(0,2,3))

    data_mean.append(batch_mean)
    data_std0.append(batch_std0)

# shape (num_iterations, 3) -> (mean across 0th axis) -> shape (3,)
data_mean = np.array(data_mean).mean(axis=0)
data_std0 = np.array(data_std0).mean(axis=0)
print(data_mean)
print(data_std0)




class Load_data(object):
    '''
    Load the images for both train and test sets 
    '''
    def __init__(self,train_path="Data/train",test_path="Data/",train_size=32,test_size=1,mean=data_mean,std=data_std0):
        '''
        Arge:
            train_size: the batch_size for the training 
            test_size : the test_size for testing 
            mean: array of input for normalizing 
            std: array of numbers for the standard deviation.
        '''
        self.train_path = train_path
        self.test_path = test_path
        self.train_size = train_size
        self.test_size = test_size
        self.mean = mean 
        self.std = std 

    def traintset(self):

        '''
        Return:
            batch of datasets using the dataloader 
        '''
        Transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(45),
            transforms.ToTensor(),
            transforms.Normalize(self.mean,self.std)
        ])
        trainsets = datasets.ImageFolder(self.train_path,transform=Transforms)

        """
        Return:
            train datasets for a batch.
        """
        return torch.utils.data.DataLoader(trainsets,batch_size=self.train_size,shuffle=True,num_workers=0)
    
    def testset(self):
        
        '''
        Return:
            batch of the testsets from dataloader. 
        '''
        Transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(self.mean,self.std)
        ])

        testsets = datasets.ImageFolder(self.test_path,transform= Transforms)

        """
        Return:
            Batch of test set for testing 
        """

        return torch.utils.data.DataLoader(testsets,batch_size=self.train_size,shuffle=False,num_workers=0)
