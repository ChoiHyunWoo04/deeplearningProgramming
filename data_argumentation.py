# 필요한지 잘 모르겠음.

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100
import os

# 저장 경로
save_path = './data/augmented_cifar100'
os.makedirs(save_path, exist_ok=True)

# CIFAR-100 원본 train셋
original_transform = transforms.Compose([
    transforms.ToTensor()
])

flip_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=1.0),  # 항상 flip
    transforms.ToTensor()
])

train_dataset = CIFAR100(root='./data', train=True, download=True, transform=original_transform)

# 저장용 리스트
augmented_images = []
augmented_labels = []

for img, label in train_dataset:
    # 원본
    augmented_images.append(img)
    augmented_labels.append(label)
    
    # flip된 이미지
    flipped_img = flip_transform(img)
    augmented_images.append(flipped_img)
    augmented_labels.append(label)

# 리스트를 tensor로 변환
augmented_images = torch.stack(augmented_images)
augmented_labels = torch.tensor(augmented_labels)

# 저장
torch.save({
    'images': augmented_images,
    'labels': augmented_labels
}, os.path.join(save_path, 'cifar100_train_with_flip.pt'))

print(f"Saved augmented dataset with shape {augmented_images.shape}")
