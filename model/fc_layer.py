import torch 
import torch.nn as nn

class FcLayer(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(1280, 640)
        self.tanh = nn.Tanh()

        self.fc2 = nn.Linear(640, 1)

        # self.fc1 = nn.Linear(1280, 1)
        # self.dropout = nn.Dropout(0.4)

    def forward(self, x):

        x = self.fc1(x)
        x = self.tanh(x)

        x = self.fc2(x)
        output = torch.sigmoid(x)

        # x = self.fc1(x)
        # x = self.dropout(x)
        # output = torch.sigmoid(x)

        return output