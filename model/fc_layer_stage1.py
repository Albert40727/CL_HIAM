import torch 
import torch.nn as nn
import torch.nn.functional as F

class FcLayerStage1(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(512, 1)

        self.dropout_1 = nn.Dropout(0.2)
        self.fc1_1 = nn.Linear(512, 1)

        self.dropout_2 = nn.Dropout(0.2)
        self.fc1_2 = nn.Linear(512, 1)

        self.dropout_3 = nn.Dropout(0.2)
        self.fc1_3 = nn.Linear(512, 1)

    def forward(self, x, x1, x2, x3):
        x = self.fc1(x)
        x = self.dropout(x)
        output = torch.sigmoid(x)

        x1 = self.fc1_1(x1)
        
        x1 = self.dropout_1(x1)
        soft_label_1 = torch.sigmoid(x1)

        x2 = self.fc1(x2)
        x2 = self.dropout(x2)
        soft_label_2 = torch.sigmoid(x2)

        x3 = self.fc1(x3)
        x3 = self.dropout(x3)
        soft_label_3 = torch.sigmoid(x3)

        return output, soft_label_1, soft_label_2, soft_label_3