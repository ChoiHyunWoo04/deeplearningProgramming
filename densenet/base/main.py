import os
import random
import numpy as np
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt

from torchvision.datasets.cifar import CIFAR100
from torchvision.transforms import ToTensor

from utils.earlystop import EarlyStop
import torchvision.transforms as transforms
from densenet import densenet_cifar

import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader

# 시드 고정 함수
def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

###################################### model setting #############################################################
DESCRIPTION = "DenseNet121, data argumentation(custom)" # 예시: 실험 내용 기록용(한글 작성시 깨짐)

LOAD_WEIGHT = False # 기존 모델 가중치를 가져올지 여부
WEIGHT_PATH = "./densenet/save/20250220_152155/weight/model_epoch_200.pt" # 기존 모델 가중치 경로

###################################### device setting ##########################################################
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]= "0"

def get_recommended_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

device = get_recommended_device()
print(device)

###################################### data setting ###########################################################

train = CIFAR100(root='./data', train=True, download=True, transform=ToTensor())
test = CIFAR100(root='./data', train=False, download=True, transform=ToTensor())

train_loader = DataLoader(train, batch_size=256, shuffle=True)
test_loader = DataLoader(test, batch_size=256, shuffle=False)

###################################### model setting ##############################################################

model = densenet_cifar()
model = model.to(device)

###################################### train parameter setting #########################################################
cfg = {
    'epoch': 100,
    'lr': 5e-4,
    'weight_decay': 5e-4
}

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])

# Scheduler
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epoch'])

###################################### creating folder for saving result #########################################################
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

save_folder = f"./densenet/save/{timestamp}" 
weight_folder = f"./densenet/save/{timestamp}/weight"

os.makedirs(weight_folder, exist_ok=True) # 현재 시간으로 폴더 생성
log_file_path = os.path.join(save_folder, 'log.txt')

# log.txt에 모델 정보 기록
with open(log_file_path, 'a') as log_file:
    log_file.write('model: Dense Net Cifar\n')
    log_file.write(f'description: {DESCRIPTION}\n\n')
    log_file.write(str(cfg) + '\n\n')
    
###################################### training loop #########################################################
earlystop = EarlyStop()
best_valid_loss = float('inf')

# Metrics tracking
train_losses, test_losses = [], []
train_accuracies, test_accuracies = [], []

for epoch in range(cfg['epoch']):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for inputs, labels in tqdm(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_loss = running_loss / total
    train_acc = 100. * correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    # Validation
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_loss = val_loss / total
    val_acc = 100. * correct / total
    test_losses.append(val_loss)
    test_accuracies.append(val_acc)

    scheduler.step()

    # save epoch info
    result = f"Epoch [{epoch+1}/{cfg['epoch']}] Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Valid Loss: {val_loss:.4f}, Valid Acc: {val_acc:.2f}% | Early Stop Count: {earlystop}"
    print(result)
    with open(log_file_path, 'a') as log_file:
        log_file.write(result + '\n')
        
    # Early Stopping
    if not earlystop.update_patience(best_valid_loss, val_loss):
        print("Early Stop.")
        break

    best_valid_loss = min(best_valid_loss, val_loss)

###################################### save Loss & Accuracy graph / Model weights #########################################################
# Saving model weights
if(LOAD_WEIGHT==False):
    MODEL_PATH = os.path.join(weight_folder, "DenseNet121.pth")
    torch.save(model.state_dict(), MODEL_PATH)

# Plotting
epochs = range(1, len(train_losses) + 1)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label='Train Loss')
plt.plot(epochs, test_losses, label='Valid Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies, label='Train Accuracy')
plt.plot(epochs, test_accuracies, label='Valid Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy over Epochs')
plt.legend()

plt.tight_layout()
plt.savefig(f'{save_folder}/loss&accuracy.png')
plt.savefig(f'{save_folder}/loss&accuracy.png')
