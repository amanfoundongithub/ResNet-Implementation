from ResNet34Classifiers.MoodClassifier import MoodClassifier

import torch 

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(device) 


model = MoodClassifier(device = device)

model.train(num_epochs = 20)  

