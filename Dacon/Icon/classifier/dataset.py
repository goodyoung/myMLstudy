import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class CustomDataset(Dataset):
    def __init__(self, df, transform=None, test=False):
        self.df = df
        self.transform = transform
        self.test = test
        self.image_columns = df.columns if test else df.columns[2:]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = row[self.image_columns].values.astype('uint8').reshape(32, 32)

        if self.transform:
            image = self.transform(image)

        return image if self.test else (image, row["label"])

def get_data():
    train = pd.read_csv("../open/train.csv")
    test = pd.read_csv("../open/test.csv")
    return train, test
    
def preprocessing(data):
    # train data 전처리
    label_encoder = LabelEncoder()
    data["label"] = label_encoder.fit_transform(data["label"])  # 문자열 라벨을 숫자로 변환
    return data, label_encoder

def get_dataloaders(dataset, train_val_idx, batch_size=32, augment_num = 3,
                    image_size = 224, num_workers=4, SEED=0):
    train, test = dataset # train, test 분리
    # train, test = get_data() # 데이터 가져오기
    
    # # data 전처리
    # label_encoder = LabelEncoder()
    # train["label"] = label_encoder.fit_transform(train["label"])  # 문자열 라벨을 숫자로 변환
    
    if train_val_idx: # kfold 이면
        train_idx, val_idx = train_val_idx
    else:
        train_idx, val_idx = train_test_split(train.index, test_size=0.2, stratify=train["label"], random_state=SEED)    
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        # transforms.RandomHorizontalFlip(p=0.7),  # 50% 확률로 좌우 반전
        # transforms.RandomVerticalFlip(p=0.3),  # 20% 확률로 상하 반전 (너무 강한 반전 방지)
        # transforms.RandomRotation(degrees=(-25, 25)),  # 제한된 각도 회전
        # transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
        # transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
    ])
    
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # dataset
    train_dataset = CustomDataset(df = train.iloc[train_idx, :], transform=transform)
    val_dataset = CustomDataset(df = train.iloc[val_idx, :], transform=test_transform)
    test_dataset = CustomDataset(df = test.iloc[:, 1:], transform=test_transform, test = True)

    # aug dataset    
    # for _ in range(augment_num): train_dataset += CustomDataset(df = train.iloc[train_idx, :], transform=transform)

    # data loader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory = True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory = True, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory = True, shuffle=False)

    return train_loader, val_loader, test_loader