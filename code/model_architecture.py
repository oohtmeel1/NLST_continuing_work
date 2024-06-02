import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import nn
class Modelo(nn.Module):
    def __init__(self):
        super(Modelo, self).__init__()
        self.conv = nn.Sequential(
        nn.Conv2d(1, 20, kernel_size = (1,1)),
        nn.Conv2d(20, 50, kernel_size = (1,1)),
        nn.Conv2d(50, 20, kernel_size = (1,1)),
        nn.LeakyReLU(inplace=True),
        nn.Conv2d(20, 20, kernel_size = (1,1)),
        nn.Conv2d(20, 10, kernel_size = (1,1)),
        torch.nn.BatchNorm2d(10),
        nn.Dropout(p=0.05),
        nn.LeakyReLU(inplace=True),
        nn.Conv2d(10, 10, kernel_size = (1,1)),
        nn.Conv2d(10, 10, kernel_size = (1,1)),
        nn.Conv2d(10, 10, kernel_size = (1,1)),
        nn.Conv2d(10, 5, kernel_size = (1,1)),
        nn.Conv2d(5, 4, kernel_size = (1,1)),
        torch.nn.BatchNorm2d(4), 
        nn.LeakyReLU(inplace=True),
        nn.Dropout(p=0.05),
        torch.nn.MaxPool2d((2, 2)), 
        nn.Conv2d(4, 3, kernel_size = (1,1)),
        nn.Dropout(p=0.05), 
        torch.nn.MaxPool2d((2, 2)),    
        nn.Conv2d(3, 2, kernel_size = (1,1)),
        torch.nn.MaxPool2d((2, 2)), 
        torch.nn.BatchNorm2d(2),
        nn.LeakyReLU(inplace=True))
        self.fc = nn.Linear(in_features=8192,out_features=4096)
        self.fc1 = nn.Linear(in_features=4096,out_features=1)

    def forward(self, x):
        x= self.conv(x)
        x = torch.flatten(x,start_dim=1,end_dim=-1)
        x=self.fc(x)
        x=self.fc1(x)
        return x


model = Modelo()
