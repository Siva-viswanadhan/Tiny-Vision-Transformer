import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import os
import torchvision.transforms as T

class MnistDataset(Dataset):
    def __init__(self,image,label):
        self.image= image
        self.label=label

        self.image_transform=T.compose(
            T.Resize
        )
