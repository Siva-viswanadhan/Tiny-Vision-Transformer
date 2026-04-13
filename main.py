import torch
import torch.nn as nn
import torch.nn.functional as F
from load_data1 import MnistDataset
from torch.utils.data import DataLoader

train_dataset=MnistDataset(r'C:\Users\siva-INC-5712\Desktop\Tiny Vision Transformer\Mnist\train-images.idx3-ubyte',
                           r'C:\Users\siva-INC-5712\Desktop\Tiny Vision Transformer\Mnist\train-labels.idx1-ubyte')

test_dataset=MnistDataset(r'C:\Users\siva-INC-5712\Desktop\Tiny Vision Transformer\Mnist\t10k-images.idx3-ubyte',
                          r'C:\Users\siva-INC-5712\Desktop\Tiny Vision Transformer\Mnist\t10k-labels.idx1-ubyte')


train_loader=DataLoader(train_dataset,batch_size=18,shuffle=True)
test_loader=DataLoader(test_dataset,batch_size=18,shuffle=False)



class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1=nn.Sequential(
            nn.Conv2d(1,64,kernel_size=7,padding=3),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        
        self.layer2=nn.Sequential(
            nn.Conv2d(64,128,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2) 
        )
        self.flatten=nn.Flatten()
        self.relu=nn.ReLU()
        self.l1=nn.Linear(128*7*7,64)
        self.l2=nn.Linear(64,32)
        self.ol=nn.Linear(32,10)

    def forward(self,x):
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.flatten(x)
        
        x=self.l1(x)
        x=self.relu(x)
        x=self.l2(x)
        x=self.relu(x)
        x=self.ol(x)