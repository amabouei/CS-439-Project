import torch
import torch.nn as nn
from torchvision import models
from models.small_cnn import SmallCNN


class Custom_model(nn.Module):
    def __init__(self, name):
        super().__init__()
        if name == 'mnist':
          self.model = SmallCNN()
        else:
          self.model = models.resnet18(weights= None)
          if name == 'cifar10':
            self.model.fc = nn.Linear(512, 10)  
          else:
            self.model.fc = nn.Linear(512, 100)  
       
   
    def forward(self, x):
        return self.model(x)
