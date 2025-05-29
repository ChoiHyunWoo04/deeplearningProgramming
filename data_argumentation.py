from torchvision import datasets, transforms
from torch.utils.data import ConcatDataset

# 기본 transform (no augmentation)
base_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

flip = transforms.Compose([
    transforms.RandomHorizontalFlip(p=1.0),
    base_transform
])


# 원본 dataset
base_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=base_transform)

# 각 augmentation dataset
flip_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=flip)