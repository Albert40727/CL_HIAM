import torch 
import torch.nn as nn
import torch.nn.functional as F

class FcLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(0.4)
        self.fc1 = nn.Linear(1280, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        output = torch.sigmoid(x)
        return output