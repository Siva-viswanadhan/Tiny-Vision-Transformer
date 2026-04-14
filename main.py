import torch
import torch.nn as nn
import torch.nn.functional as F
from load_data1 import MnistDataset
from torch.utils.data import DataLoader
from load_data1 import x_train,y_train,x_test,y_test

train_dataset=MnistDataset(x_train,y_train)
                           

test_dataset=MnistDataset(x_test,y_test)


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
        
        return x
    
model=CNN()
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)


epochs = 5




for epoch in range (epochs):
    model.train()
    train_loss=0.0
    for x_batch, y_batch in train_loader:

        optimizer.zero_grad()
        outputs=model(x_batch)
        loss=criterion(outputs,y_batch)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    print(
        f"Epoch [{epoch+1}/{epochs}] | "
        f"Train Loss: {train_loss:.4f} | "
            )
    

    model.eval()
    test_loss=0.0
    correct = 0
    total=0

    with torch.no_grad():
        for x_batch,y_batch in test_loader:
            outputs=model(x_batch)
            loss=criterion(outputs,y_batch)
            test_loss+=loss.item()

            preds=outputs.argmax(dim=1)
            correct+=(preds==y_batch).sum().item()
            total+=y_batch.size(0)

    test_loss /= len(test_loader)
    test_acc =correct/total

    print(f'test loss:{test_loss:.4f} | test accuracy : {test_acc:.2f}')