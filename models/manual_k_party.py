import torch.nn as nn
from torchvision import models
import torch


class Manual_A(nn.Module):

    def __init__(self, input_dim, u_dim=64, k=2):
        super(Manual_A, self).__init__()
        self.linear = nn.Linear(input_dim, u_dim)
        self.classifier = nn.Linear(u_dim, 1)
        self.dist_loss = 0

    def forward(self, input, U_B):
        out = self.linear(input)
        if U_B is not None:
            # self.dist_loss = torch.sum((out - U_B[0])**2) / out.size(0)
            self.dist_loss = torch.abs(torch.sum(out * U_B[0]) / out.size(0))
            out = out * U_B[0]
            # out = out + U_B[0]
            # out = torch.cat([out] + [U for U in U_B], dim=1)
        logits = self.classifier(out)
        return logits, self.dist_loss


class Manual_B(nn.Module):

    def __init__(self, input_dim, u_dim=64):
        super(Manual_B, self).__init__()
        self.linear = nn.Linear(input_dim, u_dim)

    def forward(self, input):
        out = self.linear(input)
        return out