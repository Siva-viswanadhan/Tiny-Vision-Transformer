import numpy 
import pandas 
import torch
import os
from load_data import MnistDataset


image,label=MnistDataset()
train_dataset=MnistDataset(image,label)