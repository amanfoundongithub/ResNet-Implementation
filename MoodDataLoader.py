import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


import matplotlib.pyplot as plt
import numpy as np

class MoodDataLoader:

    def __init__(self,
                 img_size = 224,
                 batch_size = 64):

        self.__transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),  
                transforms.ToTensor(),          
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  
            ]
        )

        self.__batch_size = batch_size
        self.__img_size   = img_size

    
    def load_train_dataset(self):
        
        traindata = ImageFolder(
            root = 'facedataset/train',
            transform = self.__transform
        )

        n = len(traindata)

        train_data_loader = DataLoader(
            dataset = traindata,
            batch_size = self.__batch_size,
            shuffle = True
        )

        return (train_data_loader, n)

    def load_test_dataset(self):

        traindata = ImageFolder(
            root = 'facedataset/test',
            transform = self.__transform
        )

        n = len(traindata)

        train_data_loader = DataLoader(
            dataset = traindata,
            batch_size = self.__batch_size,
            shuffle = True
        )

        return (train_data_loader, n)
