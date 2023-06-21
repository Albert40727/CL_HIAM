import torch 
import torch.nn as nn
import torch.nn.functional as F

class FcLayerStage1(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.fc1 = nn.Linear(512, 1)

        self.fc1_1 = nn.Linear(512, 1)

        self.fc1_2 = nn.Linear(512, 1)

        self.fc1_3 = nn.Linear(512, 1)

    def forward(self, x, x1=None, x2=None, x3=None):

        x = self.fc1(x)
        x = torch.sigmoid(x)
        
        if self.training:

            x1 = self.fc1_1(x1)
            x1 = torch.sigmoid(x1)

            x2 = self.fc1_2(x2)
            x2 = torch.sigmoid(x2)

            x3 = self.fc1_3(x3)
            x3 = torch.sigmoid(x3)

            return x, x1, x2, x3
        
        return x