# --------------- import ---------------
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchsummary import summary

from model import CnnNetwork


# --------------- Data Load ---------------
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64

# -------------- Data & Device Setting --------------
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


model = CnnNetwork().to(device)
print(model)

summary(model,input_size=(1,28,28))

# ------------------ loss function & optimizer setting ------------------
loss_fn = nn.CrossEntropyLoss()
#loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

# ----------------- Train & Test function -----------------
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # 예측 오류 계산
        pred = model(X)
        loss = loss_fn(pred, y)
        #MSE cost function를 사용할 시 변경해야함
        #y_onehot = torch.nn.functional.one_hot(y, num_classes=10).float()
        #loss = loss_fn(pred, y_onehot)
        
        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    # inference 모드로 실행하기 위해 학습시에 필요한 Drouout, batchnorm등의 기능을 비활성화함
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad(): # autograd engine(gradinet를 계산해주는 context)을 비활성화함
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            #MSE cost function를 사용할 시 변경해야함
            #y_onehot = torch.nn.functional.one_hot(y, num_classes=10).float()
            #test_loss += loss_fn(pred, y_onehot).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
# ---------- epoch setting ----------
epochs = 40
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)

print("Done!")


# --------------- save model --------------
torch.save(model.state_dict(), "save/CNN_model_for_FASHION_MNIST.pth")
print("Saved PyTorch Model State to model.pth")

# --------------- Inference ---------------
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

#model.load_state_dict(torch.load("CNN_model_for_FASHION_MNIST.pth", map_location=device)) 
model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    x = x.unsqueeze(0).to(device)
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')

"""
"""
