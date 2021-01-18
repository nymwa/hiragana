import torch
import torch.nn as nn
from hiragana.block import BasicBlock, ContractingBlock, ExpansiveBlock

class AbstractLayer(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return self.blocks(x)
    
class BasicLayer(AbstractLayer):
    def __init__(self, num_layers, channels):
        super().__init__()
        self.blocks = nn.Sequential(*[BasicBlock(channels) for _ in range(num_layers)])
        
class ContractingLayer(AbstractLayer):
    def __init__(self, num_layers, channels):
        super().__init__()
        self.blocks = nn.Sequential(ContractingBlock(channels),
                *[BasicBlock(channels * 2) for _ in range(num_layers - 1)])

class ExpansiveLayer(AbstractLayer):
    def __init__(self, num_layers, channels, width):
        super().__init__()
        self.blocks = nn.Sequential(ExpansiveBlock(channels, width),
                *[BasicBlock(channels // 2) for _ in range(num_layers - 1)])

