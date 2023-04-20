from .hian import HianModel
from .bp_gate import BackPropagationGate
from .attention_utils import Multihead_Cross_attention
class ReviewNetworkStage2(HianModel):
    def __init__(self, args):
        super().__init__(args)

        # Review-Level Network
        self.review_cross_attention_1 = Multihead_Cross_attention(512, 512, 512, num_heads=2)

    def forward(self, x, review_mask, batch_size):
        
        x = x.reshape(batch_size, -1, x.size(1))
        
        #Review-Level Network
        x_rf = self.review_level_network(x, review_mask, self.review_cross_attention)
        x_rf = BackPropagationGate.apply(x_rf)

        if self.training:
            x_rf_1 = self.review_level_network(x, review_mask, self.review_cross_attention_1)
            x_rf_1 = BackPropagationGate.apply(x_rf_1)
            return x_rf, x_rf_1
        
        return x_rf
    


        