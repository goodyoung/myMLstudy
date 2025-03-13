import argparse
import numpy as np
from sklearn.model_selection import StratifiedKFold
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloaders, get_data, preprocessing
from model import get_model, load_model
from train import train_model
from utils import inference, submit
from config import Config

def run_model_pipeline(dataset, args, config, train_val_idx = None, fold_num = None):
    # 데이터 로드
    train_loader, val_loader, test_loader = get_dataloaders(dataset = dataset,train_val_idx = train_val_idx,
                                                            batch_size=args.batch_size, augment_num = 3, 
                                                            image_size = config.image_size,
                                                            num_workers = args.num_workers,
                                                            SEED = config.seed)
    # 모델 로드
    model = get_model(num_channels = config.in_channel, 
                      num_labels = config.num_labels, device = args.device)
    if args.mode in ["train", "total"]:
        # 손실 함수 및 최적화 기법 설정
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)

        T_max = min(10, args.epochs)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
        # 모델 학습
        best_model = train_model(model, train_loader, val_loader, optimizer, criterion, scheduler, args, fold_num)
        
    elif args.mode == "inference":
        best_model = load_model(model, args.save_model_path, args.device)
    return best_model, test_loader
        
def main():
    config = Config()
    parser = argparse.ArgumentParser(description="Train Model")
    parser.add_argument("--mode", type=str, choices=["train", "inference", "total"], 
                            required=True, help="Mode: total or train or inference")

    parser.add_argument("--epochs", type=int, default=config.num_epochs, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=config.batch_size, help="Batch size")
    parser.add_argument("--lr", type=float, default=config.learning_rate, help="Learning rate")
    parser.add_argument("--device", type=str, default=config.device, help="Device (cuda or cpu)")
    parser.add_argument("--num_workers", type=int, default=config.num_workers, help="Number of Worker")
    parser.add_argument("--early_stop", type=int, default=config.early_stop_epoch, help="Early stopping patience")
    parser.add_argument("--save_model_path", type=str, default=f"weights/{config.weights_name}", help="Path to save trained model")
    parser.add_argument("--submission_path", type=str, default="submission.csv", help="Path to save submission file")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of folds for StratifiedKFold")
    parser.add_argument("--use_kfold", action='store_true', help="Use StratifiedKFold for training")
    
    
    args = parser.parse_args()

    train, test = get_data() # 데이터 가져오기
    train, label_encoder = preprocessing(train) # train data 전처리
    best_models = []
    if args.use_kfold:
        skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=config.seed)
        for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(train['label'])), train['label'])):
            print(f"Fold {fold+1}/{args.n_splits}")
            args.save_model_path = f"weights/baseline-timm-aug-no-pretrained_fold{fold}.pth"
            # 훈련 파이프라인
            best_model, test_loader = run_model_pipeline(dataset = (train, test),
                                                         train_val_idx = (train_idx, val_idx), 
                                                         args = args, config = config, fold_num = fold+1)
            best_models.append(best_model)
    else:    
        # 훈련 파이프라인
        best_model, test_loader = run_model_pipeline(dataset = (train, test), 
                                                     args = args, config = config)
        
    if args.mode in ["total", "inference"]:
        if args.use_kfold:
            for idx, model in enumerate(best_models):
                preds = inference(model, test_loader, args.device) # Inference
                submit(preds, label_encoder, args.submission_path + f"_fold{str(idx+1)}" + ".csv") # Submit
        else:
            preds = inference(best_model, test_loader, args.device) # Inference
            submit(preds, label_encoder, args.submission_path + ".csv") # Submit

if __name__ == "__main__":
    main()