import os
import torch
import torch.nn as nn
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
from model import get_model

DATA_DIR = os.getenv(
    "DATA_DIR",
    "/kaggle/input/bean-leaf-disease-dataset"
)

BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

train_ds=datasets.ImageFolder(
    os.path.join(DATA_DIR,"train"),
    transform=train_tfms
)
val_ds=datasets.ImageFolder(
    os.path.join(DATA_DIR,"val"),
    transform=val_tfms
)
train_loader=DataLoader(train_ds,batch_size=BATCH_SIZE,shuffle=True)  
val_loader=DataLoader(val_ds,batch_size=BATCH_SIZE,shuffle=False)

model=get_model(num_classes=len(train_ds.classes))
model=model.to(DEVICE)
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=LR)
for epoch in range(EPOCHS):
    model.train()
    running_loss=0.0
    for images,labels in train_loader:
        images=images.to(DEVICE)
        labels=labels.to(DEVICE)
        optimizer.zero_grad()
        outputs=model(images)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
    avg_loss=running_loss/len(train_loader)
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}")
