import torch
import torch.nn as nn
from hiragana.sample import DownSample, UpSample

class SE(nn.Module):
    def __init__(self, channels, reduction):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction, bias = False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias = False)
        self.act1 = nn.ReLU()
        self.act2 = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x)
        y = y.view(b, c)
        y = self.fc1(y)
        y = self.act1(y)
        y = self.fc2(y)
        y = self.act2(y)
        y = y.view(b, c, 1, 1)
        return x * y.expand_as(x)

class AbstractBlock(nn.Module):
    def __init__(self):
        super().__init__()
        
    def block(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.se(self.bn2(self.conv2(x)))
        return x
    
    def forward(self, x):
        x = self.sample(x) + self.block(x)
        x = self.relu(x)
        return x

class BasicBlock(AbstractBlock):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1, bias = False)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias = False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SE(channels, 16)
        self.relu = nn.ReLU()
        
    def sample(self, x):
        return x

class ContractingBlock(AbstractBlock):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels * 2, 3, 2, 1, bias = False)
        self.conv2 = nn.Conv2d(channels * 2, channels * 2, 3, 1, 1, bias = False)
        self.bn1 = nn.BatchNorm2d(channels * 2)
        self.bn2 = nn.BatchNorm2d(channels * 2)
        self.se = SE(channels * 2, 16)
        self.relu = nn.ReLU()
        self.sample = DownSample(channels, channels * 2, 2)

class ExpansiveBlock(AbstractBlock):
    def __init__(self, channels, width):
        super().__init__()
        padding = {2: 2, 4: 3, 8: 5, 16: 9, 32: 17}[width]
        self.conv1 = nn.Conv2d(channels, channels // 2, 3, 1, padding, bias = False)
        self.conv2 = nn.Conv2d(channels // 2, channels // 2, 3, 1, 1, bias = False)
        self.bn1 = nn.BatchNorm2d(channels // 2)
        self.bn2 = nn.BatchNorm2d(channels // 2)
        self.se = SE(channels // 2, 16)
        self.relu = nn.ReLU()
        self.sample = UpSample(channels, channels // 2, 2)

