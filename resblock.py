class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.BatchNorm2d(ch),
            nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.BatchNorm2d(ch)
        )

    def forward(self, x):
        return x + self.net(x)

class ResNetEvolution(nn.Module):
    def __init__(self, ch=128, depth=6):
        super().__init__()
        self.blocks = nn.Sequential(*[ResBlock(ch) for _ in range(depth)])

    def forward(self, x):
        return self.blocks(x)
