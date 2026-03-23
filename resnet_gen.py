import torch
import torch.nn as nn
from latent_to_feature import LatentToFeature
from to_img import ToImage
from resblock import ResNetEvolution
class ResNetGenerator(nn.Module):
    def __init__(self, z_dim=128, ch=128):
        super().__init__()
        self.init = LatentToFeature(z_dim, ch)
        self.evolve = ResNetEvolution(ch)
        self.to_img = ToImage(ch)

    def forward(self, z):
        x = self.init(z)
        x = self.evolve(x)
        return self.to_img(x)
