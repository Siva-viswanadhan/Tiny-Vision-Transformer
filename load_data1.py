import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import os
import torchvision.transforms as T
import torch
import idx2numpy


train_images_path=r"C:\Users\siva-INC-5712\Desktop\Tiny Vision Transformer\Mnist\train-images.idx3-ubyte"
train_labels_path=r"C:\Users\siva-INC-5712\Desktop\Tiny Vision Transformer\Mnist\train-labels.idx1-ubyte"

test_images_path=r"C:\Users\siva-INC-5712\Desktop\Tiny Vision Transformer\Mnist\t10k-images.idx3-ubyte"
test_labels_path=r"C:\Users\siva-INC-5712\Desktop\Tiny Vision Transformer\Mnist\t10k-labels.idx1-ubyte"



x_train = idx2numpy.convert_from_file(train_images_path)
x_train = x_train.reshape(60000,1,28,28)/255
print(x_train.shape)
x_train = torch.tensor(x_train,dtype=torch.float32)

y_train = idx2numpy.convert_from_file(train_labels_path)
y_train=y_train
print(y_train.shape)

y_train = torch.tensor(y_train,dtype=torch.long)

x_test=idx2numpy.convert_from_file(test_images_path)
x_test=x_test.reshape(10000,1,28,28)/255
print(x_test.shape)

x_test=torch.tensor(x_test,dtype=torch.float32)

y_test=idx2numpy.convert_from_file(test_labels_path)

y_test=torch.tensor(y_test,dtype=torch.long)



class MnistDataset(Dataset):
    def __init__(self,images,labels):
        self.images= images
        self.labels=labels

    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, index):

        image=self.images[index]
        label=self.labels[index]
        image=image.unsqueeze(0)  #---> add channel dimension
        return image,label
