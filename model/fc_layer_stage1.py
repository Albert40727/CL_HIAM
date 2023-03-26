import torch 
import torch.nn as nn
import torch.nn.functional as F

class FcLayerStage1(nn.Module):
    def __init__(self, args):
      super().__init__()
      self.dropout = nn.Dropout(0.1)
      self.fc1 = nn.Linear(256, 128)
      self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        output = torch.sigmoid(x)
        return output