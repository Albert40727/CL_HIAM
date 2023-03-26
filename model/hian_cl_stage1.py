from .hian import HianModel
import torch.nn as nn
import torch.nn.functional as F
import torch

class HianCollabStage1(HianModel):
    """
    B: Batch, N: User/Item max review num, W: Word, S:Sentence, A:Aspect, D:Output Dimension 
            input:  x: 32, N, 250, 768 
                            |reshape
                            v
                    x: 32*N, 250, 768 (B, N, W*S, D)
                            |
                            |w_cnn_network + attention
                            v
                    x: 32*N, 10, D  (B, N, S, D)
                            |
                            |s_cnn_network + attention
                            v 
                    x: 32*N, 10, D  (B, N, S, D)
                            |
                            |(LDA) + attention
                            v             
                    x: 32*N, 6, D   (B, N, A, D)
                            |
                            |aspect attention weighted sum
                            v
                    x: 32, N, D  (B, N, D)
                            |
                            |fc_layer
                            v  
                        x: 32, N, 1          
    """
    """
    (Some emb might be permuted during training due to the Conv1d input format)
    Word Emb:               torch.Size([50, 250, 768])
    Sentence Emb:           torch.Size([50, 10, 512])
    Weighted Sentence Emb:  torch.Size([50, 10, 256])
    Aspect Emb:             torch.Size([50, 6, 256])
    Aspect Review Emb:      torch.Size([50, 256])
    """
    def __init__(self, args):
        super().__init__(args)

        # Sentence-Level Network
        self.sentence_cnn_network_1 = nn.Sequential(
            nn.Conv1d(512, 256, self.args["sentence_cnn_ksize"]),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )
        self.sentence_attention_1 = nn.MultiheadAttention(256, num_heads=1)

        # Aspect-Level Network
        self.aspect_attention_1 = nn.MultiheadAttention(256, num_heads=1)
        self.aspect_attention_2 = nn.MultiheadAttention(256, num_heads=1)
        self.aspect_attention_3 = nn.MultiheadAttention(256, num_heads=1)

        # FC layers for stage1 predictions
        self.dropout_stage1 = nn.Dropout(0.1)
        self.fc1 = nn.Linear(256, 128)
        self.fc1_1 = nn.Linear(256, 128)
        self.fc1_2 = nn.Linear(256, 128)
        self.fc1_3 = nn.Linear(256, 128)

        self.fc2 = nn.Linear(128, 1)
        self.fc2_1 = nn.Linear(128, 1)
        self.fc2_2 = nn.Linear(128, 1)
        self.fc2_3 = nn.Linear(128, 1)

    def fc_layer(self, x, fc1, fc2, dropout):
        x = fc1(x)
        x = F.relu(x)
        x = dropout(x)
        x = fc2(x)
        output = torch.sigmoid(x)

        return output

    def forward(self, x, lda_groups):

        x = x.reshape(-1, x.size(2), x.size(3))

        # Word-Level Network
        x_s = self.word_level_network(x, self.word_cnn_network, self.word_attention)

        # Sentence-Level Network
        x_as = self.sentence_level_network(x_s, self.sentence_cnn_network, self.sentence_attention)
        x_as_1 = self.sentence_level_network(x_s, self.sentence_cnn_network_1, self.sentence_attention_1)

        # Aspect-Level Network
        x_ar = self.aspect_level_network(x_as, lda_groups, self.aspect_attention)
        x_ar_1 = self.aspect_level_network(x_as, lda_groups, self.aspect_attention_1)
        x_ar_2 = self.aspect_level_network(x_as_1, lda_groups, self.aspect_attention_2)
        x_ar_3 = self.aspect_level_network(x_as_1, lda_groups, self.aspect_attention_3)

        return x_ar, x_ar_1, x_ar_2, x_ar_3