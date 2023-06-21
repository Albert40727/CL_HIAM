import torch
from .hian import HianModel
from .bp_gate import BackPropagationGate
import torch.nn as nn
from .attention_utils import Multihead_Cross_attention

class HianCollabStage1(HianModel):
    def __init__(self, args):
        super().__init__(args)

        # Sentence-Level Network
        self.sentence_cnn_network_1 = nn.Sequential(
            nn.Conv1d(512, 512, self.args["sentence_cnn_ksize"]),
            nn.ReLU(),
        )
        self.sent_cross_attention_1 = Multihead_Cross_attention(512, 512, 512, num_heads=2)

        self.aspect_cross_attention_1 = Multihead_Cross_attention(512, 512, 512, num_heads=2)
        self.aspect_cross_attention_2 = Multihead_Cross_attention(512, 512, 512, num_heads=2)
        self.aspect_cross_attention_3 = Multihead_Cross_attention(512, 512, 512, num_heads=2)

    def forward(self, x, lda_groups):
        
        x = x.reshape(-1, x.size(2), x.size(3))

        # Word-Level Network
        x_s = self.word_level_network(x, self.word_cnn_network, self.word_attention)
        x_s = BackPropagationGate.apply(x_s)
        
        # Sentence-Level Network
        x_as = self.sentence_level_network(x_s, self.sentence_cnn_network, self.sent_cross_attention, lda_groups)
        x_as = BackPropagationGate.apply(x_as)
        if self.training:
            x_as_1 = self.sentence_level_network(x_s, self.sentence_cnn_network_1, self.sent_cross_attention_1, lda_groups)
            x_as_1 = BackPropagationGate.apply(x_as_1)

        # Aspect-Level Network
        x_ar = self.aspect_level_network(x_as, lda_groups, self.aspect_cross_attention)
        if self.training:
            x_ar_1 = self.aspect_level_network(x_as, lda_groups, self.aspect_cross_attention_1)
            x_ar_2 = self.aspect_level_network(x_as_1, lda_groups, self.aspect_cross_attention_2)
            x_ar_3 = self.aspect_level_network(x_as_1, lda_groups, self.aspect_cross_attention_3)
            return x_ar, x_ar_1, x_ar_2, x_ar_3
        
        return x_ar