import torch.nn as nn
from torchvision import models

def get_model(num_classes, pretrained=True):
    model = models.efficientnet_b0(
        weights="DEFAULT" if pretrained else None
    )

    for param in model.features.parameters():
        param.requires_grad = False
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model
