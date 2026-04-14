import numpy 
import pandas 
import torch
import os
from load_data1 import MnistDataset
from torch.utils.data import DataLoader

train_dataset=MnistDataset(r'C:\Users\siva-INC-5712\Desktop\Tiny Vision Transformer\Mnist\train-images.idx3-ubyte',
                           r'C:\Users\siva-INC-5712\Desktop\Tiny Vision Transformer\Mnist\train-labels.idx1-ubyte')

test_dataset=MnistDataset(r'C:\Users\siva-INC-5712\Desktop\Tiny Vision Transformer\Mnist\t10k-images.idx3-ubyte',
                          r'C:\Users\siva-INC-5712\Desktop\Tiny Vision Transformer\Mnist\t10k-labels.idx1-ubyte')

train_loader=DataLoader(train_dataset,batch_size=10,shuffle=True)
test_loader=DataLoader(test_dataset,batch_size=10,shuffle=False)