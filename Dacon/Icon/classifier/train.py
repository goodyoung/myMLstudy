import torch
from utils import train_one_epoch, val_one_epoch, save_checkpoint

def train_model(model, train_loader, val_loader, optimizer, criterion, scheduler, args):
    best_loss = 1e9
    best_model = None
    early_stop_num = 0  # 개선되지 않은 epoch 수 카운트

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, args.device)
        val_loss, val_acc = val_one_epoch(model, val_loader, criterion, args.device)

        print(f"Epoch [{epoch+1}/{args.epochs}] | Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2%}")

        # 모델 저장 및 Early Stopping 체크
        if val_loss < best_loss:
            print(f'Epoch:{epoch} | Best: {val_loss}')
            save_checkpoint(model, arg.save_model_path)
            best_loss = val_loss
            best_model = model
            patience = 0
        else:
            patience += 1
            
        if patience >= args.early_stop:
            print("Early stopping triggered!")
            break