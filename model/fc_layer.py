import torch 
import torch.nn as nn
import torch.nn.functional as F

class FcLayer(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(1280, 512)
        self.dropout = nn.Dropout(0.4)
        self.l_relu = nn.LeakyReLU()

        self.fc2 = nn.Linear(512, 1)
        self.dropout_2 = nn.Dropout(0.2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.l_relu(x)

        x = self.fc2(x)
        x = self.dropout_2(x)
        output = torch.sigmoid(x)
        return output