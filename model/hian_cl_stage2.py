import torch.nn as nn
import torch.nn.functional as F
from .hian import HianModel
class HianCollabStage2(HianModel):
    def __init__(self, args):
        super().__init__(args)

        # Review-Level Network
        self.review_attention_1 = nn.MultiheadAttention(256, num_heads=1)

    def forward(self, x, batch_size):
        
        #Review-Level Network
        x_rf = self.review_level_network(x, self.review_attention, batch_size=batch_size)
        x_rf_1 = self.review_level_network(x, self.review_attention_1, batch_size=batch_size)

        return x_rf, x_rf_1
    


        