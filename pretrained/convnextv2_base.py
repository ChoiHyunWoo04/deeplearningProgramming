import os
import random
import numpy as np
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt

from torchvision.datasets.cifar import CIFAR100
from torchvision.transforms import ToTensor

from earlystop import EarlyStop
import torchvision.transforms as transforms
from torchvision.transforms.autoaugment import RandAugment
from data_augmentation import Cutout

import timm
from timm.loss import SoftTargetCrossEntropy
from timm.data import Mixup

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
DESCRIPTION = "'convnextv2_base.fcmae_ft_in22k_in1k', data argumentation(RandAugment, mixup, cutout(nholes=1, size=56)), gradual unfreeze(epoch=5), smoothing=0.1" # 예시: 실험 내용 기록용(한글 작성시 깨짐)

LOAD_WEIGHT = False # 기존 모델 가중치를 가져올지 여부
weight_save_path = './pretrained/save/0605_0918/0605_0918'
WEIGHT_PATH = "./pretrained/save/0605_0918/weight/best_weight.pth" # 기존 모델 가중치 경로

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
'''
train_transform = transforms.Compose([
    transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

train_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.RandomCrop(224, padding=28),  # random crop + padding
    transforms.RandomHorizontalFlip(p=0.5),      # horizontal flip
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),  # color jitter (-30% ~ +30%)
    transforms.RandomRotation(15),         # random rotation (-15도 ~ + 15도)
    transforms.ToTensor(),                 # convert to tensor (img.shape를 (C, H, W)로 바꿔줌)
    transforms.RandomApply([Cutout(n_holes=1, length=56)], p=0.25), # Cutout 기법 확률적으로 적용
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
'''

def cutout_fn(inputs, n_holes=1, length=56):
    for i in range(inputs.size(0)):
        inputs[i] = Cutout(n_holes=n_holes, length=length)(inputs[i])
    return inputs

mixup_fn = Mixup(
    mixup_alpha=0.8,       # Mixup용 Beta 분포 계수
    cutmix_alpha=1.0,      # CutMix용 Beta 분포 계수
    cutmix_minmax=None,    # CutMix 박스 크기 제어 (None이면 무작위 비율)
    prob=0.5,              # Mixup/CutMix 적용 확률 (0.5로 낮추면 확률적 적용 가능)
    switch_prob=0.5,       # Mixup과 CutMix 중 무엇을 쓸지 선택하는 확률
    mode='batch',          # 'batch': 전체에 동일한 라벨 혼합, 'pair': 페어마다 혼합, 'elem': 각각 혼합
    label_smoothing=0.1,   # Label smoothing 값 (CrossEntropyLoss와 유사)
    num_classes=100        # 클래스 수
)

train_transform = transforms.Compose([
    transforms.Resize(224),
    RandAugment(num_ops=2, magnitude=9),  # or AutoAugment
    transforms.ToTensor(),                 # convert to tensor (img.shape를 (C, H, W)로 바꿔줌)
    #transforms.RandomApply([Cutout(n_holes=1, length=56)], p=0.25), # Cutout 기법 확률적으로 적용
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train = CIFAR100(root='./data', train=True, download=True, transform=train_transform)
test = CIFAR100(root='./data', train=False, download=True, transform=test_transform)

train_loader = DataLoader(train, batch_size=256, shuffle=True, num_workers=2)
test_loader = DataLoader(test, batch_size=256, shuffle=False, num_workers=2)

###################################### model setting ##############################################################

model = timm.create_model('convnextv2_base.fcmae_ft_in22k_in1k', pretrained=True)

num_features = model.head.in_features
# ConvNeXt-Tiny의 마지막 head는 일반적으로 Linear(in_features=768, out_features=1000) 구조

# 새 head 구성 (입력은 B x 768 이 되어야 함)
custom_head = nn.Sequential(
    nn.Linear(num_features, 3072),
    nn.GELU(), # ReLU는 단순하지만 정보 손실이 큼. GELU 또는 SiLU는 smoother하여 성능이 더 나은 경우가 많음 (특히 transformer 기반 백본에서).
    nn.BatchNorm1d(3072),
    nn.Dropout(p=0.4),

    nn.Linear(3072, 2048),
    nn.GELU(),
    nn.BatchNorm1d(2048),
    nn.Dropout(p=0.4),

    nn.Linear(2048, 1024),
    nn.GELU(),
    nn.BatchNorm1d(1024),
    nn.Dropout(p=0.3),

    nn.Linear(1024, 100),
)

# 기존 head 제거
model.head = nn.Identity()

# 최종 모델 정의 (풀링 포함 수동 정의)
class ConvNeXtForCIFAR(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.head = head

    def forward(self, x):
        x = self.backbone.forward_features(x)  # [B, C, H, W]
        x = self.pool(x)                       # [B, C, 1, 1]
        x = self.flatten(x)                    # [B, C]
        x = self.head(x)                       # [B, 100]
        return x
    
    def get_stage(self, idx):
        return self.backbone.stages[idx]

for param in model.parameters():
    param.requires_grad = False

# head는 항상 학습
for param in model.head.parameters():
    param.requires_grad = True
'''
for m in model.modules():
    if isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)): # 1 에폭에 5시간
        for p in m.parameters():
            p.requires_grad = True
'''         
model = ConvNeXtForCIFAR(model, custom_head).to(device)

# 기존의 모델 로드할 경우
if LOAD_WEIGHT: 
    checkpoint = torch.load(WEIGHT_PATH)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()

###################################### train parameter setting #########################################################
cfg = {
    'epoch': 100, # epoch 크기가 달라지면 CosineAnnealingLR 부분도 달라짐..!
    'lr': 0.0004,
    'weight_decay': 0.01
}

# Loss and optimizer
criterion = nn.CrossEntropyLoss() # label_smoothing=0.1
soft_criterion = SoftTargetCrossEntropy()
# 옵티마이저 정의 (head만 먼저 학습)
optimizer = torch.optim.AdamW([{"params": model.head.parameters(), "lr": cfg['lr']}], weight_decay=cfg['weight_decay'])
#optimizer = optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])

# Scheduler
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epoch'])

###################################### creating folder for saving result #########################################################

timestamp = datetime.now().strftime('%m%d_%H%M')

save_folder = f"./pretrained/save/{timestamp}" 
weight_folder = f"./pretrained/save/{timestamp}/weight"
if (LOAD_WEIGHT==False):  
    os.makedirs(weight_folder, exist_ok=True) # 현재 시간으로 폴더 생성
    log_file_path = os.path.join(save_folder, 'log.txt')

    # log.txt에 모델 정보 기록
    with open(log_file_path, 'a') as log_file:
        log_file.write('model: ConvNext v2 tiny\n')
        log_file.write(f'description: {DESCRIPTION}\n\n')
        log_file.write(str(cfg) + '\n\n')
        for name, param in model.named_parameters():
            log_file.write(f'{name} - {param.requires_grad}\n')
        log_file.write('\n')
    
###################################### training loop #########################################################
earlystop = EarlyStop()
best_valid_loss = float('inf')

best_valid_acc = 0.0

# Metrics tracking
train_losses, test_losses = [], []
train_accuracies, test_accuracies = [], []
if (LOAD_WEIGHT==False):  
    for epoch in range(cfg['epoch']):
        if epoch == 5: # 점진적으로 Unfreeze
            for param in model.get_stage(3).parameters():
                param.requires_grad = True
            optimizer.add_param_group({'params': model.get_stage(3).parameters(), "lr": 5e-5})
        '''
        if epoch == 14:
            print("Unfreezing normalization layers in stage2 blocks 6-8 only")
            norm_params = []

            for name, module in model.get_stage(2).named_modules():
                if any(name.startswith(f'blocks.{i}') for i in [5, 6, 7, 8]):
                    if isinstance(module, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
                        for param in module.parameters():
                            param.requires_grad = True
                        norm_params += list(module.parameters())
                        print(f"Unfrozen: stage2.{name}")

            # 옵티마이저에 선택적 norm 레이어 추가
            if norm_params:
                optimizer.add_param_group({'params': norm_params})
        '''
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # hard label 저장 (int tensor, shape: [B])
            hard_labels = labels
            
            # 확률적으로 mixup 적용
            if mixup_fn is not None:
                inputs, labels = mixup_fn(inputs, labels)
                
            # Cutout 확률적으로 적용
            if np.random.rand() < 0.25:
                inputs = cutout_fn(inputs)
                
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = soft_criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 정확도 계산 (soft label이 아니라 hard label 기준)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(hard_labels).sum().item()
            total += hard_labels.size(0)

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
    MODEL_PATH = os.path.join(weight_folder, "ConvNextv2_tiny.pth")
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
    image_folder = './CImages'
    output_file = f'{weight_save_path}.txt'
    
    results = []

    for i in tqdm(range(1, 3001)):
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