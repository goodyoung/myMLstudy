import timm
import torch
import torch.nn as nn
from transformers import ViTForImageClassification, ViTConfig
import torch
import torch.nn as nn

import torch
import torch.nn as nn
import timm

class CustomConv(nn.Module):
    def __init__(self, in_channels=1, out_channels=64):
        super().__init__()
        # 각 branch의 채널 수를 계산
        ch1 = out_channels // 3            # 예: 64//3 = 21
        ch2 = out_channels // 3            # 21
        ch3 = out_channels - ch1 - ch2     # 64 - 21 - 21 = 22

        # 세 개의 다른 커널 크기의 Conv 레이어 적용
        self.conv_3x3 = nn.Conv2d(in_channels, ch1, kernel_size=3, padding=1)
        self.conv_5x5 = nn.Conv2d(in_channels, ch2, kernel_size=5, padding=2)
        self.conv_7x7 = nn.Conv2d(in_channels, ch3, kernel_size=7, padding=3)
        
        # 1x1 Conv로 출력 채널을 64로 맞추기 (안정적인 학습을 위해)
        self.conv1x1 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False)
        
    def forward(self, x):
        # 여러 Conv 적용 후 병합
        out_3x3 = self.conv_3x3(x)
        out_5x5 = self.conv_5x5(x)
        out_7x7 = self.conv_7x7(x)
        
        # Concatenation 후 1x1 Conv로 차원 조정
        out = torch.cat([out_3x3, out_5x5, out_7x7], dim=1)
        out = self.conv1x1(out)

        return out



def get_model(num_channels, num_labels, device):
    # for ViT
    # config = ViTConfig.from_pretrained("google/vit-base-patch16-224-in21k")
    # config.num_channels = num_channels
    # config.num_labels = num_labels
    # model = ViTForImageClassification(
    #     # "google/vit-base-patch16-224-in21k",
    #     # ignore_mismatched_sizes=True,
    #     config=config,
    # )
    model = timm.create_model('timm/resnet50.a1_in1k', 
                              num_classes=num_labels, in_chans=num_channels,
                              pretrained=False)  # 사전 훈련된 가중치 로드
    model.conv1 = CustomConv()
    # model.bn1 = nn.BatchNorm2d(192)
    # model.layer1[0].conv1 = nn.Conv2d(192, 64, kernel_size=1, stride=1, bias=False)
    
    # model = timm.create_model('timm/efficientnet_b6.ra2_in1k', 
    #                           num_classes=num_labels, in_chans=num_channels,
    #                           pretrained=False)  # 사전 훈련된 가중치 로드

    # model = timm.create_model("beitv2_base_patch16_224.in1k_ft_in22k_in1k",
    #                           num_classes=num_labels, in_chans=num_channels,
    #                             pretrained=False)
    model.to(device)
    return model
    
def load_model(model, weights_path, device):
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only = True))
    return model
