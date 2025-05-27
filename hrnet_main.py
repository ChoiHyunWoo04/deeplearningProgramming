import os
import numpy as np
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt

from torchvision.datasets.cifar import CIFAR100
from torchvision.transforms import ToTensor

from config.models import MODEL_EXTRAS

import torch
import torch.nn as nn
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader

###################################### model setting #############################################################
LOAD_WEIGHT = False # 기존 모델 가중치를 가져올지 여부
WEIGHT_PATH = "./save/20250220_152155/weight/model_epoch_200.pt" # 기존 모델 가중치 경로

###################################### device setting ##########################################################
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]= "1"

###################################### data setting ###########################################################
train = CIFAR100(root='./data', train=True, download=True, transform=ToTensor())
test = CIFAR100(root='./data', train=False, download=True, transform=ToTensor())

train_loader = DataLoader(train, batch_size=32, shuffle=True)
test_loader = DataLoader(test, batch_size=32, shuffle=False)

###################################### model setting ##############################################################
