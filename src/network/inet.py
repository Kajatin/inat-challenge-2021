import torch
import torch.nn as nn
from torchvision.models import resnext50_32x4d


class iNet(nn.Module):
    def __init__(self, cfg):
        super(iNet, self).__init__()
        self.base = resnext50_32x4d(pretrained=cfg["pretrained"])
        self.base.fc = nn.Linear(self.base.fc.in_features, cfg["out_dim"])

    def forward(self, x):
        x = self.base(x)
        return x
