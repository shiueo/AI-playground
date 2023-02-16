"""
MNIST데이터는 0~9까지의 이미지로 이루어진 손글씨 데이터이다.
60000개의 데이터, 레이블
그리고 10000개의 테스트 데이터와 레이블로 구성되어있다.

목표: 손글씨 이미지가 들어오면 이 이미지가 무슨 숫자인지 맞추는 것.
-> 머신러닝이다.
"""

import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import random

CUDA_AVAILABLE = torch.cuda.is_available()  # 버전에 맞는 pytorch 다운 받을 것
device = torch.device("cuda" if CUDA_AVAILABLE else "cpu")
print(f"{device} 사용")

SEED = 922
random.seed(SEED)
torch.manual_seed(SEED)
if device == "cuda":
    torch.cuda.manual_seed_all(SEED)

# hyper parameters
training_epochs = 15
batch_size = 100

# MNIST 다운
mnist_train = dsets.MNIST(
    root="MNIST_data/", train=True, transform=transforms.ToTensor(), download=True
)

mnist_test = dsets.MNIST(
    root="MNIST_data/", train=False, transform=transforms.ToTensor(), download=True
)

# dataset loader
data_loader = DataLoader(
    dataset=mnist_train,
    batch_size=batch_size,  # 배치 크기는 100
    shuffle=True,
    drop_last=True,
)

# MNIST 이미지 사이즈가 28*28이라 784이다
linear = nn.Linear(in_features=784, out_features=10, bias=True).to(device=device)

citerion = nn.CrossEntropyLoss().to(device=device)
optimizer = torch.optim.SGD(linear.parameters(), lr=0.1)

for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = len(data_loader)

    for X, Y in data_loader:
        X = X.view(-1, 28 * 28).to(device=device)  # 이게 28*28 이미지이고
        Y = Y.to(device=device)  # 이게 레이블이다. 즉 0-9까지의 수이다.

        optimizer.zero_grad()
        hypothesis = linear(X)
        cost = citerion(hypothesis, Y)
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch

    print(f"Epoch: {epoch}, cost: {avg_cost}")
print("Learning Done")

# 여기까지가 학습 이제부터는 모델 테스트

with torch.no_grad():
    X_test = mnist_test.test_data.view(-1, 28 * 28).float().to(device=device)
    Y_test = mnist_test.test_labels.to(device=device)

    prediction = linear(X_test)
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    accuracy = correct_prediction.float().mean()
    print(f"Accuracy: {accuracy.item()}")

    r = random.randint(0, len(mnist_test) - 1)  # mnist test 데이터셋에서 아무거나 하나 뽑는다
    X_random_data = (
        mnist_test.test_data[r : r + 1].view(-1, 28 * 28).float().to(device=device)
    )
    Y_random_data = mnist_test.test_labels[r : r + 1].to(device=device)
    print(f"Label: {Y_random_data.item()}")
    r_prediction = linear(X_random_data)
    print(f"Prediction: {torch.argmax(r_prediction, 1).item()}")

    plt.imshow(
        mnist_test.test_data[r : r + 1].view(28, 28),
        cmap="Greys",
        interpolation="nearest",
    )
    plt.show()
