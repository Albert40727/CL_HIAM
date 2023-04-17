from .hian import HianModel
from .bp_gate import BackPropagationGate
import torch.nn as nn
import torch.nn.functional as F
import torch

class HianCollabStage1(HianModel):
    def __init__(self, args):
        super().__init__(args)

        # Sentence-Level Network
        self.sentence_cnn_network_1 = nn.Sequential(
            nn.Conv1d(512, 512, self.args["sentence_cnn_ksize"]),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )
        self.sentence_attention_1 = nn.MultiheadAttention(512, num_heads=1, batch_first=True)

        # Aspect-Level Network
        self.aspect_attention_1 = nn.MultiheadAttention(512, num_heads=1, batch_first=True)
        self.aspect_attention_2 = nn.MultiheadAttention(512, num_heads=1, batch_first=True)
        self.aspect_attention_3 = nn.MultiheadAttention(512, num_heads=1, batch_first=True)

        # FC layers for stage1 predictions
        self.dropout_stage1 = nn.Dropout(0.4)
        self.fc1 = nn.Linear(512, 512)
        self.fc1_1 = nn.Linear(512, 512)
        self.fc1_2 = nn.Linear(512, 512)
        self.fc1_3 = nn.Linear(256, 512)

        self.fc2 = nn.Linear(512, 1)
        self.fc2_1 = nn.Linear(512, 1)
        self.fc2_2 = nn.Linear(512, 1)
        self.fc2_3 = nn.Linear(512, 1)

    def fc_layer(self, x, fc1, fc2, dropout):
        x = fc1(x)
        x = F.tanh(x)
        x = dropout(x)
        x = fc2(x)
        output = torch.sigmoid(x)

        return output

    def forward(self, x, lda_groups):

        x = x.reshape(-1, x.size(2), x.size(3))

        # Word-Level Network
        x_s = self.word_level_network(x, self.word_cnn_network, self.word_attention)
        x_s = BackPropagationGate.apply(x_s)

        # Sentence-Level Network
        x_as = self.sentence_level_network(x_s, self.sentence_cnn_network, self.sentence_attention, lda_groups)
        x_as = BackPropagationGate.apply(x_as)
        if self.training:
            x_as_1 = self.sentence_level_network(x_s, self.sentence_cnn_network_1, self.sentence_attention_1, lda_groups)
            x_as_1 = BackPropagationGate.apply(x_as_1)
        
        # Aspect-Level Network
        x_ar = self.aspect_level_network(x_as, lda_groups, self.aspect_attention)
        if self.training:
            x_ar_1 = self.aspect_level_network(x_as, lda_groups, self.aspect_attention_1)
            x_ar_2 = self.aspect_level_network(x_as_1, lda_groups, self.aspect_attention_2)
            x_ar_3 = self.aspect_level_network(x_as_1, lda_groups, self.aspect_attention_3)
            return x_ar, x_ar_1, x_ar_2, x_ar_3

        return x_ar