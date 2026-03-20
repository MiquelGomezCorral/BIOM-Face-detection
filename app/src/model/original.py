import torch
from torch import nn

class RowleyFaceNN(nn.Module):
    def __init__(self, in_channels=1):
        super(RowleyFaceNN, self).__init__()
        # 4 units: 10x10 subregions
        self.path1 = nn.Conv2d(in_channels, 1, kernel_size=10, stride=10)
        # 16 units: 5x5 subregions
        self.path2 = nn.Conv2d(in_channels, 1, kernel_size=5, stride=5)
        # 6 units: overlapping 20x5 horizontal stripes
        self.path3 = nn.Conv2d(in_channels, 1, kernel_size=(5, 20), stride=(3, 1))
        
        # 4 + 16 + 6 = 26 total hidden units
        self.fc = nn.Linear(26, 1)

    def forward(self, x):
        # The 1998 paper used Tanh activations instead of ReLU
        x1 = torch.flatten(torch.tanh(self.path1(x)), 1)
        x2 = torch.flatten(torch.tanh(self.path2(x)), 1)
        x3 = torch.flatten(torch.tanh(self.path3(x)), 1)
        
        x_cat = torch.cat((x1, x2, x3), dim=1)
        return torch.sigmoid(self.fc(x_cat))