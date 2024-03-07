import torch
import torch.nn as nn
class LeNet_5(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels =3,out_channels=16,kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.layer3 =nn.Sequential(
            nn.Linear(53*53*32,512),
            nn.ReLU()
        )
        self.layer4 =nn.Sequential(
            nn.Linear(512,2),
            nn.ReLU()
        )
        self.output = nn.Softmax(dim = 1)
    def forward(self, x):

        x = self.layer1(x)

        x = self.layer2(x)
        x = self.layer3(x.view(x.size(0), -1))
        x = self.layer4(x)
        return self.output(x)