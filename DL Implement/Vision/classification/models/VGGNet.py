import torch
import torch.nn as nn

class VGGNet(nn.Module):
    def __init__(self,config):
        super().__init__()
        print("CONFONF",config)
        self.features = self.make_layer(config)
        self.avgpool = nn.AdaptiveAvgPool2d(7)
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7,4096),
            nn.ReLU(inplace = True),
            nn.Dropout(0.5),
            nn.Linear(4096,4096),
            nn.ReLU(inplace = True),
            nn.Dropout(0.5),
            nn.Linear(4096,2),
        )
        
    def make_layer(self,conf):
        print("CONF",conf)
        layers=[]
        in_channels = 3
        for i in conf:
            if i == 'M':
                layers.append(nn.MaxPool2d(2))
            else:
                layers.append(nn.Sequential(
                    nn.Conv2d(in_channels, i , kernel_size=3,padding=1),
                    nn.BatchNorm2d(i),
                    nn.ReLU(inplace = True)
                ))
                in_channels = i
        return nn.Sequential(*layers)
        
    def forward(self , x):
        x = self.features(x)
        x = self.avgpool(x)
        return self.classifier(x.view(x.size(0),-1))