import numpy 
import pandas 
import torch
import os
from load_data import MnistDataset


loader=MnistDataset(r'C:\Users\siva-INC-5712\Desktop\Tiny Vision Transformer\Mnist\t10k-images.idx3-ubyte',
                    r'C:\Users\siva-INC-5712\Desktop\Tiny Vision Transformer\Mnist\t10k-labels.idx1-ubyte',
                    r'C:\Users\siva-INC-5712\Desktop\Tiny Vision Transformer\Mnist\train-images.idx3-ubyte',
                    r'C:\Users\siva-INC-5712\Desktop\Tiny Vision Transformer\Mnist\train-labels.idx1-ubyte'
)

(x_train,y_train),(x_test,y_test)=loader.load_data()