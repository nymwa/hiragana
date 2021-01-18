import torch
import torch.nn as nn
from hiragana.layer import BasicLayer, ContractingLayer, ExpansiveLayer
from hiragana.init import init_params

class Encoder(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.conv = nn.Conv2d(1, 16, 3, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.layer1 = BasicLayer(n, 16)
        self.layer2 = ContractingLayer(n, 16)
        self.layer3 = ContractingLayer(n, 32)
        self.avgpool = nn.AvgPool2d(8)
        
    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

class Decoder(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.fc = nn.Linear(64, 64 * 8 * 8)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.layer1 = ExpansiveLayer(n, 64, 8)
        self.layer2 = ExpansiveLayer(n, 32, 16)
        self.layer3 = BasicLayer(n, 16)
        self.conv = nn.Conv2d(16, 1, 3, 1, 1, bias = False)
        
    def forward(self, x):
        x = self.fc(x).reshape(-1, 64, 8, 8)
        x = self.relu(self.bn(x))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.conv(x)
        return x

class AutoEncoder(nn.Module):
    def __init__(self, n=18):
        super().__init__()
        self.encoder = Encoder(n)
        self.decoder = Decoder(n)
        self.relu = nn.ReLU()
        self.apply(init_params)

    def forward(self, x):
        x = self.encoder(x)
        x = self.relu(x)
        x = self.decoder(x)
        return x

class Classifier(nn.Module):
    def __init__(self, n=18, num_classes=46):
        super().__init__()
        self.encoder = Encoder(n)
        self.ln = nn.LayerNorm(64)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.20)

    def forward(self, x):
        x = self.encoder(x)
        x = self.ln(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc(x)
        return x

