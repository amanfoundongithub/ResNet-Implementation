from ResNet34Classifiers.CIFAR10Classifier import Classifier

import torch 

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(device) 


model = Classifier(device = device)


print("Params:",model.total_parameters())


