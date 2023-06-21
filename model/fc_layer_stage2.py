import torch 
import torch.nn as nn
import torch.nn.functional as F
from .fc_layer import FcLayer

class FcLayerStage2(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_layer_stage2 = FcLayer()
        self.fc_layer_1_stage2 = FcLayer()
        self.fc_layer_2_stage2 = FcLayer()
        self.fc_layer_3_stage2 = FcLayer()

    def forward(self, x, x1 = None, x2 = None, x3 = None):
        output = self.fc_layer_stage2(x)
        if self.training:
            soft_label_1 = self.fc_layer_1_stage2(x1)
            soft_label_2 = self.fc_layer_2_stage2(x2)
            soft_label_3 = self.fc_layer_3_stage2(x3)
            return output, soft_label_1, soft_label_2, soft_label_3

        return output