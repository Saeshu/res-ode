import torch
import torch.nn as nn

class LatentToFeature(nn.Module):
    def __init__(self, z_dim=128, base_ch=128):
        super().__init__()
        self.fc = nn.Linear(z_dim, base_ch * 4 * 4)

    def forward(self, z):
        x = self.fc(z)
        return x.view(z.size(0), -1, 4, 4)  # (B, C, 4, 4)
