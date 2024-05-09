from torchvision.datasets import CIFAR10
from torchvision import transforms 

from torch.utils.data import DataLoader


class CIFAR10DataLoader:

    def __init__(self,
                 img_size = 224, 
                 batch_size = 100):
        

        self.__transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), 
                (0.5, 0.5, 0.5)),
                ])

        self.__imgsize = img_size
        self.__batchsize = batch_size

    def load_train_dataset(self):

        train_data = CIFAR10(
                root = './data',
                train = True,
                download = True, 
                transform = self.__transform
        )

        

        length = len(train_data)

        return (DataLoader(train_data, 
                          batch_size = self.__batchsize,
                          shuffle = True), length)

    def load_test_dataset(self):

        test_data = CIFAR10(
            root = './data',
            train = False,
            download = True, 
            transform = self.__transform
        )

        return DataLoader(test_data, 
                          batch_size = self.__batchsize,
                          shuffle = True)
    
    def load_classes(self):

        return ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



