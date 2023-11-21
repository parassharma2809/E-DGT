import torch
from torch import nn


class ContrastiveLoss(nn.Module):
    def __init__(self):
        super(ContrastiveLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, x, target):
        return self.ce_loss(x, target)


class NCELoss(nn.Module):
    def __init__(self):
        super(NCELoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, x):
        bsz = len(x)
        target = torch.arange(bsz)
        x = torch.cat((x, x.t()), dim=1)
        return self.ce_loss(x, target)
