import numpy as np
import shap
import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F
from torchvision import datasets, transforms

from explain import get_shap_values, plot_explanation

batch_size = 128 # batch size is important not only for training but also for explaining
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "mnist_data",
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    ),
    batch_size=batch_size,
    shuffle=True,
)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "mnist_data", train=False, transform=transforms.Compose([transforms.ToTensor()])
    ),
    batch_size=batch_size,
    shuffle=True,
)
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.Dropout(),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(320, 50),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(50, 10),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 320)
        x = self.fc_layers(x)
        return x

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output.log(), target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )

device = "cuda:0"
# backbone_checkpoint_path = "Densenet_epoch2.pth"
# model = torch.load(backbone_checkpoint_path, map_location=device)

model = Net().to(device)

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
num_epochs = 2
for epoch in range(1, num_epochs + 1):
    train(model, device, train_loader, optimizer, epoch)

device = "cpu"
model.to(device)
batch = next(iter(test_loader))
images, _ = batch

background = images[:100]
test_images = images[100:103]

shap_values = get_shap_values(model, background, test_images)
plot_explanation(shap_values, test_images)
