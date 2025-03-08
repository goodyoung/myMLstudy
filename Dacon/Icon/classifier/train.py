import time
from utils import train_one_epoch, val_one_epoch, save_checkpoint

def train_model(model, train_loader, val_loader, optimizer, criterion, scheduler, args):
    best_loss = 1e9
    best_model = None
    early_stop_num = 0  # 개선되지 않은 epoch 수 카운트
    total_train_time = 0  # 전체 학습 시간

    for epoch in range(args.epochs):
        start_time = time.time()  # 시작 시간 측정

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, args.device)
        val_loss, val_acc = val_one_epoch(model, val_loader, criterion, args.device)

        epoch_time = time.time() - start_time  # 한 Epoch 소요 시간
        total_train_time += epoch_time  # 전체 학습 시간 누적

        print(f"Epoch [{epoch+1}/{args.epochs}] | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Time: {epoch_time:.2f} sec")

        # 모델 저장 및 Early Stopping 체크
        if val_loss < best_loss:
            print(f'Best Model Saved!!')
            save_checkpoint(model, args.save_model_path)
            best_loss = val_loss
            best_model = model
            patience = 0
        else:
            patience += 1
            
        if patience >= args.early_stop:
            print("Early stopping triggered!")
            print(f"Total Training Time: {total_train_time:.2f} sec")
            return best_model

    print(f"Training complete. Total Training Time: {total_train_time:.2f} sec")
    return best_model