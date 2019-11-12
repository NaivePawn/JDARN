from torch import nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.restored = False
        self.discriminator = nn.Sequential(
            nn.Linear(256, 3072),
            nn.ReLU(True),
            nn.Linear(3072, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 31 + 1)
        )

    def forward(self, x):
        x = F.dropout(F.relu(x), training=self.training)
        out = self.discriminator(x)
        return out