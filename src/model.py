import torch.nn as nn
from torchvision import models
def get_model(num_classes):
    model=models.googlenet(weights="DEFAULT")
    for param in model.parameters():
        param.requires_grad=True
    model.fc=nn.Linear(model.fc.in_features,num_classes)
    return model    