import os
import pickle
import random
import numpy as np
import torch

## This function saves the logs during traning
def save_logs(logs, path):
	dir = os.path.dirname(path)
	os.makedirs(dir,  exist_ok=True) 
	with open(path, 'wb') as fp:
		pickle.dump(logs, fp)


## This function load the logs
def load_logs(path):
	with open(path, 'rb') as fp:
		logs = pickle.load(fp)
	return logs  


## This function save the state of optimizer and model
def save_state(model_state, optimizer_state, path):
	dir = os.path.dirname(path)
	os.makedirs(dir,  exist_ok=True) 
	torch.save ({'model_state_dict': model_state,
				'optimizer_state_dict': optimizer_state,
				}, path)  
		
## This function recover the state of model   
def load_state_model(path):
	checkpoint = torch.load(path)
	return checkpoint['model_state_dict']

## Print the logs.
def print_log(log):
	print(f"Epoch = {log['epoch']}")
	print(f"Train_loss = {log['train_loss']}, Train_acc = {log['train_acc']}, Train_grad = {log['train_grad']}")
	print(f"Valid_loss = {log['val_loss']}, valid_acc = {log['val_acc']}, valid_grad = {log['val_grad']}")
	print(f"Test_loss = {log['test_loss']}, Test_acc = {log['test_acc']}, Test_grad = {log['test_grad']}")
	print(f"learning rate {log['lr']}")
	print(f"Total time is {log['time']}")   
	print('-' * 10) 



## set all of seeds in order to reproducing the result
def set_all_seeds(SEED=100):
  random.seed(SEED)
  torch.manual_seed(SEED)
  np.random.seed(SEED)
  torch.cuda.manual_seed(SEED) 
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  # torch.backends.cudnn.enabled = False  	
	