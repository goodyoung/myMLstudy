import argparse
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloaders
from model import get_model
from train import train_model
from utils import inference, submit
from config import Config

def main():
    config = Config()
    parser = argparse.ArgumentParser(description="Train Model")
    parser.add_argument("--epochs", type=int, default=config.num_epochs, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=config.batch_size, help="Batch size")
    parser.add_argument("--lr", type=float, default=config.learning_rate, help="Learning rate")
    parser.add_argument("--device", type=str, default=config.device, help="Device (cuda or cpu)")
    parser.add_argument("--early_stop", type=int, default=config.early_stop_epoch, help="Early stopping patience")
    parser.add_argument("--save_model_path", type=str, default=f"weights/{config.weights_name}", help="Path to save trained model")
    parser.add_argument("--submission_path", type=str, default="submission.csv", help="Path to save submission file")
    
    args = parser.parse_args()

    # 1. 데이터 로드
    train_loader, val_loader, test_loader, label_encoder = get_dataloaders(batch_size=args.batch_size, 
                                                                           augment_num = 3, 
                                                                           image_size = config.image_size,
                                                                           SEED = config.seed)

    # 2. 모델 로드
    model = get_model(num_channels = config.in_channel, 
                      num_labels = config.num_labels, device = args.device)
    
    # 3. 손실 함수 및 최적화 기법 설정
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 4. 학습 실행
    best_model = train_model(model, train_loader, val_loader, optimizer, criterion, scheduler, args)

    # 5. Inference
    preds = inference(best_model, test_loader, args.device)
    
    # 6. Submit
    submit(preds, label_encoder, args.submission_path)

if __name__ == "__main__":
    main()