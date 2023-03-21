import torch 
import torch.nn as nn
import torch.nn.functional as F

class FcLayer(nn.Module):
    def __init__(self, args):
      super(FcLayer, self).__init__()
      self.dropout = nn.Dropout(0.2)
      self.fc1 = nn.Linear(768, 384)
      self.fc2 = nn.Linear(384, 1)

    def forward(self, x):
      x = self.fc1(x)
      x = F.relu(x)
      x = self.dropout(x)
      x = self.fc2(x)
      output = torch.sigmoid(x)
      return output