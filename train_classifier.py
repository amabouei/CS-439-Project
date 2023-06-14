import wandb
import os
import argparse
import time
import random
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.optim.lr_scheduler import LambdaLR

###Our classes and functions
from optimizers.sophia import SophiaG
from optimizers.adahessian import Adahessian
from dataset import Custom_dataset
from models.custom_model import Custom_model
from utils import *



### The main procedure for training the model
def train(model, device, dataset, optimizer, epochs, dir, adaptive_lr = True, wandb_run = None):
	start = time.time()
	logs = []  
	if adaptive_lr: ### Scheduler for decreasing the learning during training.
		scheduler = LambdaLR(optimizer, lr_lambda= lambda x: 1 if x < epochs * 0.6 else 0.1 if x < epochs * 0.8 else 0.01)
	for epoch in range(1, epochs + 1):    
		model.train()
		for batch_idx, (x, y) in tqdm(enumerate(dataset.train_loader)):
				x, y = x.to(device), y.to(device)
				optimizer.zero_grad()
				loss = F.cross_entropy(model(x), y)
				loss.backward(create_graph = True)
				optimizer.step()
		if adaptive_lr:
				scheduler.step()
		log = dict()
		log['train_loss'], log['train_acc'], log['train_grad'] = eval_model(model, dataset.train_loader)
		log['val_loss'], log['val_acc'], log['val_grad'] = eval_model(model, dataset.valid_loader)
		log['test_loss'], log['test_acc'], log['test_grad'] = eval_model(model, dataset.test_loader)
		log['epoch'] = epoch
		log['time'] = time.time() - start
		log['lr'] = optimizer.param_groups[0]['lr']
		logs.append(log)
		print_log(log)
		if wandb_run != None:
				wandb_run.log(log)
		if epoch % 20 == 0:
			save_state(model.state_dict(), optimizer.state_dict(), dir + f"model_epoch_{epoch}.pt")
	return logs   


### Calculating the norm of the model's parametere
def get_gradient_norm(model):
	grad_norm = 0
	with torch.no_grad():
		for p in model.parameters():
			if p.grad is not None and p.requires_grad:
				p_norm = p.grad.detach().data.norm(2)
				grad_norm += p_norm.item() ** 2 
	return grad_norm


### This function evaluates the model on the input dataset. 
### The outputs are loss, accuracy, and average of norm of gradients of parameteres

def eval_model(model, data_loader):
	if len(data_loader.dataset) == 0:
		return 0, 0, 0
	model.eval()
	mean_loss, count, accuracy, sum_grad = 0, 0, 0, 0 
	for idx, (x, y) in enumerate(data_loader):
			x, y = x.to(device), y.to(device)    
			model.zero_grad()          
			pred = model(x)
			loss = F.cross_entropy(pred, y)
			loss.backward(create_graph = True)
			mean_loss += F.cross_entropy(pred, y, reduction='sum').item() / len(data_loader.dataset)
			accuracy += (y == pred.argmax(dim=1)).sum().item() / len(data_loader.dataset)     
			sum_grad += get_gradient_norm(model)

	return float(format(mean_loss, '.4f')), float(format(accuracy, '.4f')),  float(format(np.sqrt(sum_grad / len(data_loader.dataset)), '.4f'))   #### BIASED ESTIMATOR         


## Creating dataset, model, and optimizer corresponding to the inputs, then calling train function.
def train_classifier(config, wandb_run):
	dir = config['path']
	dataset = Custom_dataset(config['name_dataset'], config['batch_size'], config['val_ratio'])
	model = Custom_model(config['name_dataset']).to(device)
	if config['optimizer'].lower() == 'sgd':
		optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=config['momentum'], weight_decay=config['weight_decay'])
	elif  config['optimizer'].lower() == 'adam':
		optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
	elif  config['optimizer'].lower() == 'sophia':
		optimizer = SophiaG(model.parameters(), lr=config['lr'],  weight_decay=config['weight_decay'])
	elif config['optimizer'].lower() == 'adahessian':
		optimizer = Adahessian (model.parameters(), lr=config['lr'],  weight_decay=config['weight_decay'])
	else:
		raise Exception("Invalid optimizer")
	logs = train(model, device, dataset, optimizer, epochs = config['epochs'], dir = dir, wandb_run = wandb_run)
	save_logs(logs, dir + 'logs.pkl')
	save_state(model.state_dict(), optimizer.state_dict(), dir + "model_final.pt")


### Preparing the seed, wandb for starting the training thread
def run(config):
	set_all_seeds(config['seed'])
	name1 = f"lr_{config['lr']}_momentum_{config['momentum']}_weightdecay_{config['weight_decay']}"
	name2 = f"{config['name_dataset']}_{config['optimizer']}/"
	name = name2 + name1
	wandb_run = None
	if config['wandb'] == 1:
		print('sssss')
		wandb_run = wandb.init(
			project = 'Second_Order_Method',
			name = name,
			config = config
		)
	config['path'] = config['path'] + f"/{config['name_dataset']}/{config['optimizer']}/{name1}/"
	train_classifier(config, wandb_run = wandb_run)
	if wandb_run != None:
		wandb_run.finish()



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# torch.cuda.get_device_name(0)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Second_Order_Project')
  parser.add_argument('--epochs', type=int, default=50,
      help='number of epochs to train')
  parser.add_argument('--batch-size', type=int, default=128,
      help='input batch size for training')
  parser.add_argument('--optimizer', '--opt', default='SGD',
      help='Name of Optimizer SGD/ADAM/SOPHIA')
  parser.add_argument('--lr',  type= float, default=0.1,
      help='learning rate')
  parser.add_argument('--momentum', type= float,default=0.9,
      help='momentum')
  parser.add_argument('--weight-decay', type= float, default=2e-4,
      help='weight_decay')
  parser.add_argument('--name-dataset', '--dataset',  default='cifar10',
      help='Name of dataset')
  parser.add_argument('--val-ratio', default = 0.1)
  parser.add_argument('--wandb', type = int, default = 1)
  parser.add_argument('--seed', type = int, default = 100)
  parser.add_argument('--path', default='logs', help = 'Ratio of validation set')
  args = parser.parse_args()			
  config = vars(args)
  run(config)
	