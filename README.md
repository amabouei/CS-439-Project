# A Study of Second-Order Optimization Methods for Image Classification
In this project, we compare the performance of two second-order methods, Sophia and Adahessian, with the well-known first-order methods, SGD and Adam. We consider the image classification task on the benchmark datasets Cifar-10, Cifar-100, and Mnist.



## contents
* ```models```: includes a smallCNN for the Mnist dataset and a custom_classifier based on resnet18.
* ```run.sh```: script for reproducing the results.
* ```optimizers```: includes Sophia and Adahessian optimizers based on their main source.
* ```log```: results of different hyperparameters and settings.
* ```datasets.py```: custom datasets for Mnist, Cifar10, and Cifar100. 
* ```utils.py```: includes some functions for saving and loading the models.
* ```train_classifiers.py```: the main file for training the models. Take note to the next section for how to use it.
* ```requirments.txt```: list of packages should be installed.




## Install
We run all of our codes on Python (3.10.12). Make sure you have at least this version to prevent conflicts. For installing the dependencies, run the following code
```sh
pip install -r requirements.txt
```
Then to reproduce the results, run the following script. 
```sh
bash run.sh
```
Note that you can change the epoch, path, etc., in run.sh.
Moreover, you can train using the train_classifer.py file with the following command
```sh
python3 train_classifier.py --path log --wandb 1 --lr 0.01 --opt SGD --dataset cifar10
```
### Arguments
* ```path```: The folder name where logs and models' parameters will be saved.
* ```wandb```: If the input of this argument is 1, the logs will be synchronized with the wandb server. Make sure to login into wandb before running.
* ```lr```: The learning rate.
* ```dataset```: The name of the dataset you want to train on. The choices are "cifar10", "cifar100", and "mnist".
* ```opt```: The name of the optimizer in ("SGD", "ADAM", "adahessian", "Sophia").
* ```seed```: Indicate the seed value (default value is 100).
* ```weight-decay```: the weight-decay value (default value is 2e-4).
* ```momentum```: The momentum value (default value is 0.9).
* ```val-ratio```: The ratio of validation size / (size of all training samples)  (default value is 0.1).
* ```epochs```: Number of epochs (default value is 50).
* ```batch-size```: Batch size for train/valid/test sets (default value is 128).
## Contact
Please get in touch with amir.abouei@epfl.ch or sina.akbari@epfl.ch in case you have questions regarding the code.
