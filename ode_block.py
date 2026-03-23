from torchdiffeq import odeint
import torch
import torch.nn as nn
class ODEEvolution(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        t = torch.tensor([0, 1]).float().to(x.device)
        out = odeint(self.func, x, t, method='rk4')  # or 'dopri5'
        return out[-1]
