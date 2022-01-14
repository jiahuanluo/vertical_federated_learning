import torch.nn as nn
import torch
from models.transformer import TransformerEncoder


class Manual_A(nn.Module):
    def __init__(self, num_classes, input_dim ,layers, u_dim=30, k=2):
        super(Manual_A, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 30, kernel_size=1, padding=0, bias=False)
        self.net = TransformerEncoder(embed_dim=30, num_heads=5, layers=5)
        self.proj1 = nn.Linear(30, 30)
        self.relu = nn.ReLU()
        self.proj2 = nn.Linear(30, 30)
        self.classifier = nn.Linear(u_dim * k, num_classes)

    def forward(self, input, U_B):
        x = input.transpose(1, 2)
        x = self.conv1(x)
        x = x.permute(2, 0, 1)
        x = self.net(x)
        x = x[-1]
        x = self.proj1(x)
        x = self.relu(x)
        out = self.proj2(x)
        if U_B is not None:
            out = torch.cat([out] + [U for U in U_B], dim=1)
        logits = self.classifier(out)
        return logits


class Manual_B(nn.Module):

    def __init__(self, input_dim, u_dim=64):
        super(Manual_B, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 30, kernel_size=1, padding=0, bias=False)
        self.net = TransformerEncoder(embed_dim=30, num_heads=5, layers=5)
        self.proj1 = nn.Linear(30, 30)
        self.relu = nn.ReLU()
        self.proj2 = nn.Linear(30, 30)

    def forward(self, input):
        x = input.transpose(1, 2)
        x = self.conv1(x)
        x = x.permute(2, 0, 1)
        x = self.net(x)
        x = x[-1]
        x = self.proj1(x)
        x = self.relu(x)
        out = self.proj2(x)
        return out
