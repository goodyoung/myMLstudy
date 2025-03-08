import argparse
import torch
from dataset import get_dataloaders
from model import get_model
from utils import train_one_epoch, val_one_epoch, save_checkpoint
from config import Config

def main():
    config = Config()
    parser = argparse.ArgumentParser(description="Train Model")
    parser.add_argument("--epochs", type=int, default=config.num_epochs, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=config.batch_size, help="Batch size")
    parser.add_argument("--lr", type=float, default=config.learning_rate, help="Learning rate")
    parser.add_argument("--device", type=str, default=config.device, help="Device (cuda or cpu)")
    parser.add_argument("--early_stop", type=int, default=config.early_stop_epoch, help="Early stopping patience")

    args = parser.parse_args()

    # 1. 데이터 로드
    train_loader, val_loader = get_dataloaders(batch_size=args.batch_size)

    # 2. 모델 로드
    model = get_model(num_classes=config.num_classes)
    model.to(args.device)

    # 3. 손실 함수 및 최적화 기법 설정
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # 4. 학습 실행
    best_loss = float('inf')
    patience = 0

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, args.device)
        val_loss, val_acc = val_one_epoch(model, val_loader, criterion, args.device)

        print(f"Epoch [{epoch+1}/{args.epochs}] | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%}")

        # 모델 저장 및 Early Stopping 체크
        if val_loss < best_loss:
            save_checkpoint(model, "weights/vit_best.pth")
            best_loss = val_loss
            patience = 0
        else:
            patience += 1
            if patience >= args.early_stop:
                print("Early stopping triggered!")
                break

        scheduler.step()

if __name__ == "__main__":
    main()
