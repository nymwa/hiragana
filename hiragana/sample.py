import torch
import torch.nn as nn

class DownSample(nn.Module):
    def __init__(self, in_channels, channels, stride):
        super().__init__()
        width = channels - in_channels
        self.stride = stride
        self.pad = nn.ConstantPad3d((0, 0, 0, 0, 0, width), 0)
        
    def forward(self, x):
        x = x[:, :, ::self.stride, ::self.stride]
        return self.pad(x)

class UpSample(nn.Module):
    def __init__(self, in_channels, channels, scale_factor):
        super().__init__()
        self.width = in_channels - channels
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='nearest')
        
    def forward(self, x):
        x = x[:, :self.width, :, :]
        return self.upsample(x)

