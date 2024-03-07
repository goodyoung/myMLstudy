import torch
import torch.nn as nn
class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels=96, kernel_size=11,stride=4),
            nn.ReLU(inplace = True),
            nn.LocalResponseNorm(2),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels = 96, out_channels = 256, kernel_size = 5, stride=  1, padding = 2),
            nn.ReLU(inplace = True),
            nn.LocalResponseNorm(2),
            nn.MaxPool2d(kernel_size=3,stride=2)
        )        
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels = 384, kernel_size=3, padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels=384, out_channels = 384, kernel_size=3, padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels=384, out_channels = 256, kernel_size=3, padding = 1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.avgpool = torch.nn.AdaptiveAvgPool2d((6,6))
        self.fc = nn.Sequential(
            nn.Linear(256*6*6,4096),
            nn.ReLU(inplace = True),
            nn.Dropout(0.5),
            nn.Linear(4096,4096),
            nn.ReLU(inplace = True),
            nn.Dropout(0.5),
            nn.Linear(4096,2)
        )

    def forward(self, x):
        x = self.layer1(x)       
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        # x = x.view(-1, 256 * 6* 6)
        x = torch.flatten(x,1)
        x = self.fc(x)
        return x