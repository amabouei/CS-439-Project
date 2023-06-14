
from torch.utils.data import Dataset
import torch
from torchvision import datasets, models, transforms

class Custom_dataset(Dataset):
    """ Our Custom dataset, this class has some functions for downloading and partitioning datasets in train/valid/test.
    Arguments:
        name (string): Indicates the name of the dataset, which is in ['cifar10', 'mnist', 'cifar100']
        batch_size (int, optional): the size of batch
        train_valid_split (int, optional): ratio size of train/validation set 
    """
    def __init__(self, name, batch_size = 128, train_valid_split = 0.1):
      super().__init__()
      self.batch_size = batch_size
      self.train_valid_split = train_valid_split
      if name.lower() == 'cifar10':
          self.train_loader, self.valid_loader, self.test_loader = \
              self.get_cifar10_dataset()
      elif name.lower() == 'mnist':
           self.train_loader, self.valid_loader, self.test_loader = \
              self.get_mnist_dataset()
      elif name.lower() == 'cifar100':
           self.train_loader, self.valid_loader, self.test_loader= \
              self.get_cifar100_dataset()
      else:
        raise Exception ('Invalid dataset') 

    def split_train(self, trainset):
        train_size = int((1- self.train_valid_split) * len(trainset))
        valid_size = len(trainset) - train_size
        train_ds, valid_ds = torch.utils.data.random_split(trainset, (train_size, valid_size))
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=self.batch_size, shuffle = True)
        valid_loader =  torch.utils.data.DataLoader(valid_ds, batch_size=self.batch_size, shuffle = False)
        
        return train_loader, valid_loader 
        
    def get_cifar10_dataset(self):
        trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform= transforms.ToTensor())
        train_loader, valid_loader = self.split_train(trainset)
        testset = datasets.CIFAR10(root='./data', train=False, download=True, transform= transforms.ToTensor())
        test_loader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size, shuffle=False)
        return train_loader, valid_loader, test_loader
    
    def get_mnist_dataset(self):
        trainset = datasets.MNIST(root='./data', train=True, download=True, transform= transforms.ToTensor())
        train_loader, valid_loader = self.split_train(trainset)
        testset = datasets.MNIST(root='./data', train=False, download=True, transform= transforms.ToTensor())
        test_loader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size, shuffle=False)
        return train_loader, valid_loader, test_loader
    
    def get_cifar100_dataset(self):
        trainset = datasets.CIFAR100(root='./data', train=True, download=True, transform= transforms.ToTensor())
        train_loader, valid_loader = self.split_train(trainset)
        testset = datasets.CIFAR100(root='./data', train=False, download=True, transform= transforms.ToTensor())
        test_loader = torch.utils.data.DataLoader(testset, batch_size= self.batch_size, shuffle=False)
        return train_loader, valid_loader, test_loader

