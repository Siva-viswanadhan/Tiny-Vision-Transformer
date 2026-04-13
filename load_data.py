import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import os
import torchvision.transforms as T
import torch

class MnistDataset(Dataset):
    def __init__(self,images,labels):
        self.images= images
        self.labels=labels

        

        

    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, index):

        image=self.images[index]
        label=self.labels[index]

        image= torch.tensor(image,dtype=torch.float32)
        image=image/255.0
        image=image.unsqueeze(0)  #---> add channel dimension

        label=torch.tensor(label,dtype=torch.long)


        return image,label
