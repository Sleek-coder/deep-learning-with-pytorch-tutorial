import torch 
import torch.nn as nn 
import torch.nn.Fuctional as F 
import torchvision 
import trochvision.transforms as transforms
import matplotlib.pyplot as plt 
import numpy as np  

#Device config 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Hyperparameters 

num_epochs = 4 
batch_size = 4 
learning_rate  = 0.001
# dataset has  PILImage  images of range [0, 1]

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', traini=False,
                                            download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size= batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, 
                                          shuffle=True)

classes= ("plane" "car","bird","cat", 'deer", dog ' ')