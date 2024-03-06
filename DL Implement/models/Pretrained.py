from torchvision import models
import torch.nn as nn

class ResNet_pre(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet50_pre = models.resnet50(weights=None)
        self.fc = nn.Linear(in_features=1000, out_features=2, bias=True) #fine-tuning
    def forward(self,x):
        x = self.resnet50_pre(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x
        
class AlexNet_pre(nn.Module):
    def __init__(self):
        super().__init__()
        self.alexnet_pre = models.alexnet(weights=None)
        
        self.fc = nn.Linear(in_features=1000, out_features=2, bias=True) #fine tuning

    def forward(self,x):
        x = self.alexnet_pre(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x
        
class VGGNet_pre(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg11_pre = models.vgg11(weights=None)
        self.fc = nn.Linear(in_features=1000, out_features=2, bias=True) #fine tuning
    def forward(self,x):
        x = self.vgg11_pre(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x