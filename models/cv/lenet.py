import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)  # 28x28 → 28x28
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)  # 28x28 → 14x14
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)  # 14x14 → 10x10 → 10x10
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)  # 10x10 → 5x5

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = F.tanh(self.conv1(x))
        x = self.pool1(x)
        x = F.tanh(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)
        return x
