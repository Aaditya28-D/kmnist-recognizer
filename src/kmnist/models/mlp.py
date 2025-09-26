import torch.nn as nn

class MLPWide(nn.Module):
    def __init__(self, p=0.35, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(p),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(p),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(p),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):  # x: [B,1,28,28]
        return self.net(x)