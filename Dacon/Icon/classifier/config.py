import torch
from dataclasses import dataclass
@dataclass
class Config:
    image_size = 224  # 생성되는 이미지 해상도
    batch_size = 32
    num_epochs = 240
    learning_rate = 5e-5
    num_workers = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 42
    weights_name = f"vit-baseline-model-change-{num_epochs}.pth"
    num_labels = 10
    in_channel = 1
    early_stop_epoch = 40