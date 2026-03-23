class ODEFunc(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GroupNorm(8, ch),
            nn.Tanh(),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GroupNorm(8, ch)
        )

    def forward(self, t, x):
        return self.net(x)
