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
from densenet import densenet_custom, DenseNet121
from data_augmentation import Cutout

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
DESCRIPTION = "DenseNet growth=24, data argumentation(custom, cutout(nholes=1, size=8))" # 예시: 실험 내용 기록용(한글 작성시 깨짐)

LOAD_WEIGHT = False # 기존 모델 가중치를 가져올지 여부
WEIGHT_PATH = "./densenet/save/20250601_162825/weight/best_weight.pth" # 기존 모델 가중치 경로

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
'''train_transform = transforms.Compose([
    transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])'''

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=3),  # random crop + padding
    transforms.RandomHorizontalFlip(p=0.5),      # horizontal flip
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),  # color jitter (-30% ~ +30%)
    transforms.RandomRotation(15),         # random rotation (-15도 ~ + 15도)
    transforms.ToTensor(),                 # convert to tensor (img.shape를 (C, H, W)로 바꿔줌)
    transforms.RandomApply([Cutout(n_holes=1, length=8)], p=0.25), # Cutout 기법 확률적으로 적용
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))  # CIFAR-100 mean/std
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

train = CIFAR100(root='./data', train=True, download=True, transform=train_transform)
test = CIFAR100(root='./data', train=False, download=True, transform=test_transform)

train_loader = DataLoader(train, batch_size=256, shuffle=True)
test_loader = DataLoader(test, batch_size=256, shuffle=False)

###################################### model setting ##############################################################

model = densenet_custom()
model = model.to(device)

# 기존의 모델 로드할 경우
if LOAD_WEIGHT: 
    checkpoint = torch.load(WEIGHT_PATH)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    
###################################### train parameter setting #########################################################
cfg = {
    'epoch': 200, # epoch 크기가 달라지면 CosineAnnealingLR 부분도 달라짐..!
    'lr': 5e-4,
    'weight_decay': 5e-4
}

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])

# Scheduler
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epoch'])

###################################### creating folder for saving result #########################################################
timestamp = datetime.now().strftime('%m%d_%H%M')

save_folder = f"./densenet/save/{timestamp}" 
weight_folder = f"./densenet/save/{timestamp}/weight"

os.makedirs(weight_folder, exist_ok=True) # 현재 시간으로 폴더 생성
log_file_path = os.path.join(save_folder, 'log.txt')

# log.txt에 모델 정보 기록
with open(log_file_path, 'a') as log_file:
    log_file.write('model: Dense Net custom\n')
    log_file.write(f'description: {DESCRIPTION}\n\n')
    log_file.write(str(cfg) + '\n\n')
    
###################################### training loop #########################################################
earlystop = EarlyStop()
best_valid_loss = float('inf')

best_valid_acc = 0.0

# Metrics tracking
train_losses, test_losses = [], []
train_accuracies, test_accuracies = [], []

if (LOAD_WEIGHT==False):  
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
        
        # Save best model
        if val_acc > best_valid_acc:
            best_valid_acc = val_acc
            torch.save(model.state_dict(), weight_folder + '/best_weight.pth')
            print(f'--> Best model saved at epoch {epoch+1} with acc {best_valid_acc:.2f}')
            
        # Early Stopping
        if not earlystop.update_patience(best_valid_loss, val_loss):
            print("Early Stop.")
            break

        best_valid_loss = min(best_valid_loss, val_loss)
    
###################################### save Loss & Accuracy graph / Model weights #########################################################
# Saving model weights
if(LOAD_WEIGHT==False):
    MODEL_PATH = os.path.join(weight_folder, "DenseNet_16.pth")
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


if LOAD_WEIGHT:
    from PIL import Image

    # 폴더 경로
    image_folder = 'CImage'
    output_file = f'{save_folder}.txt'
    
    results = []

    for i in range(1, 3001):
        image_path = os.path.join(image_folder, f'{i}.jpg')
        if not os.path.isfile(image_path):
            print(f'Warning: {image_path} not found, skipping.')
            continue

        # 이미지 로드 및 전처리
        image = Image.open(image_path).convert('RGB')
        input_tensor = test_transform(image).unsqueeze(0).to(device)  # 배치 차원 추가

        # 추론
        with torch.no_grad():
            output = model(input_tensor)
            pred_class = output.argmax(dim=1).item()

        # 결과 저장
        results.append(f'{i:04d}, {pred_class}')

    # 결과를 텍스트 파일에 저장
    with open(output_file, 'w') as f:
        f.write('number, label\n')
        for line in results:
            f.write(line + '\n')

    print(f'Finished. Results saved to {output_file}')