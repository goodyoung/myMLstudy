import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class CustomDataset(Dataset):
    def __init__(self, df, transform=None, test=False):
        self.df = df
        self.transform = transform
        self.test = test
        self.image_columns = df.columns[1:] if test else df.columns[2:]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = row[self.image_columns].values.astype('uint8').reshape(32, 32)

        if self.transform:
            image = self.transform(image)

        return image if self.test else (image, row["label"])

def get_dataset():
    train = pd.read_csv("open/train.csv")
    test = pd.read_csv("open/test.csv")
    submission = pd.read_csv("open/sample_submission.csv")
    return train, test, submission

def get_dataloaders(batch_size=32):
    train, test, submission = get_dataset() # 데이터 가져오기
    


    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((config.image_size, config.image_size)),
        transforms.RandomHorizontalFlip(p=0.7),  # 50% 확률로 좌우 반전
        transforms.RandomVerticalFlip(p=0.3),  # 20% 확률로 상하 반전 (너무 강한 반전 방지)
        transforms.RandomRotation(degrees=(-25, 25)),  # 제한된 각도 회전
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_dataset = CustomDataset(train_df, transform=transform)
    val_dataset = CustomDataset(val_df, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
