# deeplearningProgramming
* git clone 해오신 후, /deeplearningProgramming 폴더 위에서 main.py를 수행하시면 됩니다!

## base model(hrnet_w18)
* Epoch [26/150] Train Loss: 0.2530, Train Acc: 91.88% | Valid Loss: 3.7361, Valid Acc: 40.37%

* Data Augmentation(AutoAugmentPolicy.CIFAR10) - Epoch [35/200] Train Loss: 0.9539, Train Acc: 75.24% | Valid Loss: 2.4266, Valid Acc: 47.99% (20250529_180959)
    * Drop Path in BasicBlock - Epoch [35/200] Train Loss: 0.8683, Train Acc: 77.70% | Valid Loss: 2.4806, Valid Acc: 47.85% (20250529_220227)

## smaller model(hrnet_w18_small_v2)

## base model(dense_cifar)

## larger model(dense_custom)
* 기존 dense_cifar 함수에서 growth를 6, 10, 16, 20, 24로 변경해가며 실험
* 기존 dense_cifar는 Cifar10 데이터셋 기준 모델이므로 Cifar100에 맞는 모델은 growth가 더 높을 때 성능이 더 좋아짐.
