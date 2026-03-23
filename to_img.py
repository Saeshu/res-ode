
import torch
import torch.nn as nn
from upsample import UpsampleBlock
class ToImage(nn.Module):
    def __init__(self, base_ch=128):
        super().__init__()
        self.net = nn.Sequential(
            UpsampleBlock(base_ch, 64),   # 4 → 8
            UpsampleBlock(64, 32),       # 8 → 16
            UpsampleBlock(32, 16),       # 16 → 32
            nn.Conv2d(16, 3, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)
      
