import torch.nn as nn
import torch.optim as optim
import torch 
from tqdm import tqdm 

from ResNet34.ResNet34 import ResNet
from CIFAR10DataLoader import CIFAR10DataLoader


class Classifier:
    def __init__(self,
                 device = 'cpu'):


        self.__classifier_model = ResNet(
            layers = [3, 4, 6, 3], num_classes = 10
        ).to(device)
        self.__device           = device

    def train(self, num_epochs = 20):
        # Training set up
        criterion = nn.CrossEntropyLoss()


        trainer = torch.optim.SGD(self.__classifier_model.parameters(), 
                                    lr = 0.1, 
                                    weight_decay = 0.001, momentum = 0.9)  


        lr_scheduler = optim.lr_scheduler.StepLR(optimizer = trainer, 
                                                 step_size = 7, 
                                                 gamma = 0.1)
        
        # Call the data loader
        CIFAR10Data = CIFAR10DataLoader(
            batch_size = 50
        )

        # Load the training dataset from the loader 
        train_loader, length = CIFAR10Data.load_train_dataset()

        test_loader = CIFAR10Data.load_test_dataset() 


        for epoch in range(num_epochs):
            print("\n-------------")
            print(f"Epoch #{epoch}")
            running_loss = 0.0 

            for i, data in tqdm(enumerate(train_loader), total = int(length/train_loader.batch_size)):
                
                inputs, labels = data 

                inputs = inputs.to(self.__device)
                labels = labels.to(self.__device)

                trainer.zero_grad()
                
                outputs = self.__classifier_model(inputs) 
                loss    = criterion(outputs, labels)

                loss.backward()
                trainer.step()

                running_loss += loss.item()

                del inputs, labels, outputs 

            print("Running loss: ", (running_loss/length) * train_loader.batch_size, end = '\r')
            print("\n-------------")

            self.test_accuracy(test_loader)
            print("\n-------------")
    

    def test_accuracy(self, test_loader):

        device = self.__device
        with torch.no_grad():
            correct = 0
            total = 0

            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = self.__classifier_model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                del images, labels, outputs
            
            print(f"Accuracy on Test data: {correct/total * 100 : 02}%")
        
    def total_parameters(self):
        return sum(p.numel() for p in self.__classifier_model.parameters())

            


            



        
