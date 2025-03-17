import torch
import pandas as pd
from tqdm.auto import tqdm

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    train_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)#.logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0) # 마지막 배치 크기가 작아도 정확하게 반영됨.
        
    one_epoch_loss = train_loss / len(loader.dataset)
    return one_epoch_loss

@torch.no_grad()
def val_one_epoch(model, loader, criterion, device):
    model.eval()
    val_loss, correct, total = 0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)#.logits
        loss = criterion(outputs, labels)

        val_loss += loss.item() * images.size(0)
        
        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
            
    one_epoch_loss = val_loss / len(loader.dataset)
    acc = correct / total
    return one_epoch_loss, acc
    
# @torch.no_grad()
# def inference(model, loader, device):
#     model.eval()    
#     preds = []
#     for images in loader:
#         images = images.to(device)
#         outputs = model(images)#.logits
#         _, predicted = torch.max(outputs, 1)
#         preds.extend(predicted.cpu().numpy())
#     return preds

@torch.no_grad()
def inference(model, loader, device):
    model.eval()    
    preds, probs = [], []  
    
    for images in loader:
        images = images.to(device)
        outputs = model(images) #.logits
        
        probabilities = torch.softmax(outputs, dim=1)  # 확률값 변환
        _, predicted = torch.max(outputs, 1)  # 최종 예측 클래스
        preds.extend(predicted.cpu().numpy())  
        probs.extend(probabilities.cpu().numpy().tolist())  # 확률값 리스트에 저장

    return preds, probs  # 예측값과 확률값 반환

def submit(preds, probs, encoder, file_name):
    submission = pd.read_csv("../open/sample_submission.csv")
    print(encoder.classes_)
    pred_labels = encoder.inverse_transform(preds)
    submission['label'] = pred_labels
    submission['probs'] = probs
    submission.to_csv(file_name, index=False)

def save_checkpoint(model, path):
    torch.save(model.state_dict(), path)
