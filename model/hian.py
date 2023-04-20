import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from .attention_utils import Multihead_Cross_attention


class HianModel(nn.Module):
    """
    B: Batch, R: User/Item max review, W: Max Word, S:Max Sentence, A:Aspect, D:Emb Dimension 

            input:  x: 32, R, 250, 768 (B, R, W*S, D)
                            |reshape
                            v
                    x: 32*R, 250, 768 (B*R, W*S, D)
                            |
                            |w_cnn_network + attention
                            v
                    x: 32*R, 10, D  (B*R, S, D)
                            |
                            |s_cnn_network + attention
                            v 
                    x: 32*R, 10, D  (B*R, S, D)
                            |
                            |get_aspect_emb_from_sent (LDA) 
                            v             
                    x: 32*R, 6, D   (B*R, A, D)
                            |
                            |aspect attention
                            v
                    x: 32, R, D  (B, R, D)
                            |
                            |review_network
                            v  
                        x: 32, D
                            |
                            |co_attention (Outside HianModel)
                            v
                        x: 32, D                 

    Item network for example:
    (Some emb might be permuted during training due to the Conv1d input format)
    Input Emb:              torch.Size([32, 50, 250, 768])
    Word Emb:               torch.Size([32*50, 250, 768])
    Sentence Emb:           torch.Size([32*50, 10, 512])
    Weighted Sentence Emb:  torch.Size([32*50, 10, 512])
    Aspect Emb:             torch.Size([32*50, 6, 512])
    Aspect Review Emb:      torch.Size([32, 50, 512])
    Item Reiew feature:     torch.Size([32, 512])
    Item Emb:               torch.Size([32, 512]) (after co-attention) 
    """

    def __init__(self, args):
        super().__init__()
        self.args = args

        # Word-Level Network
        self.word_pad_size = int((self.args["word_cnn_ksize"]-1)/2)
        self.word_cnn_network = nn.Sequential(
            nn.Conv1d(768, 512, self.args["word_cnn_ksize"]),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )
        self.word_attention = nn.MultiheadAttention(512, num_heads=2, batch_first=True)
        
        # Sentence-Level Network
        self.sent_pad_size = int((self.args["sentence_cnn_ksize"]-1)/2)
        self.sentence_cnn_network = nn.Sequential(
            nn.Conv1d(512, 512, self.args["sentence_cnn_ksize"]),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )
        self.sent_cross_attention = Multihead_Cross_attention(512, 512, 512, num_heads=2)

        # Aspect-Level Network
        self.lda_group_num = self.args["lda_group_num"]
        self.aspect_cross_attention = Multihead_Cross_attention(512, 512, 512, num_heads=2)

        
        # Review-Level Network
        self.review_cross_attention = Multihead_Cross_attention(512, 512, 512, num_heads=2)

    def word_level_network(self, x, word_cnn, word_attention):
        x = torch.permute(x, (0, 2, 1))
        x = F.pad(x, (self.word_pad_size, self.word_pad_size), "constant", 0) # same to keras: padding = same
        x = word_cnn(x)
        x = torch.permute(x, [0, 2, 1])
        x, att_weight = word_attention(x, x, x, need_weights=True)
        x = self.word_weighted_sum(x, self.args["max_sentence"])
        return x
        
    def word_weighted_sum(self, input_tensor, max_sentence):
        """
        Weighted sum words' emb into sentences' emb.
        """
        batch, word_num, word_dim = input_tensor.shape
        input_tensor = input_tensor.reshape(batch, max_sentence, -1, word_dim)
        sentence_tensor = torch.sum(input_tensor, dim=2)

        return sentence_tensor
    
    def sentence_level_network(self, x, sent_cnn, sent_attention, lda_groups):
        """
        Be careful that we're using self defined attention not torch.nn.MultiheadAttention
        """
        x = torch.permute(x, [0, 2, 1])
        x = F.pad(x, (self.sent_pad_size, self.sent_pad_size), "constant", 0) # same to keras: padding = same
        x = sent_cnn(x)
        x = torch.permute(x, [0, 2, 1]) 
        # Mask generate
        k_sent_mask = (lda_groups == False).reshape(-1, lda_groups.size(2))
        sent_mask = torch.logical_or(k_sent_mask.unsqueeze(dim=-1), k_sent_mask.unsqueeze(dim=1))
        x, att_weight = sent_attention(x, x, mask=~sent_mask.to(self.args["device"]))
        # x, sent_att_weight = sent_attention(x, x, x, attn_mask=sent_mask, need_weights=True)
        # x, sent_att_weight = sent_attention(x, x, x, need_weights=True)
        # x = torch.nan_to_num(x, nan=0)
        return x 
    
    def aspect_level_network(self, x, lda_groups, aspect_attention):
        """
        Be careful that we're using self defined attention not torch.nn.MultiheadAttention
        """
        x, aspect_att_mask = self.get_aspect_emb_from_sent(x, lda_groups, self.lda_group_num)
        # x, att_weight = aspect_attention(x, x, x, attn_mask=aspect_att_mask.to(self.args["device"]), need_weights=True)
        x, att_weight = aspect_attention(x, x, mask=~aspect_att_mask.to(self.args["device"])) # the input mask should be reversed compare to nn.MultiheadAttention
        x = torch.sum(x, dim=1)

        return x
    
    def get_aspect_emb_from_sent(self, input_tensor, lda_groups, group_num):
        """
        Weighted sum sentences' emb according to their LDA groups respectively.  
        """
        k_aspect_att_mask = torch.zeros((input_tensor.size(0), group_num), dtype=torch.bool)
        lda_groups = torch.unsqueeze(lda_groups.reshape(-1, lda_groups.size(2)), dim=-1)
        group_tensor_list = []

        for group in range(group_num):
            mask = lda_groups == group
            mask_sum = torch.sum(mask, dim=1)
            mask_sum[mask_sum == 0] = 1
            group_tensor = torch.where(mask , input_tensor, 0.)
            group_tensor = torch.sum(group_tensor, dim = 1)
            group_tensor = group_tensor/mask_sum
            group_tensor_list.append(group_tensor)

            # Get ignore value index (False index)
            if group == 0:
                k_aspect_att_mask[:,group] = False
            else:
                group_mask = torch.any((lda_groups.squeeze(dim=-1))==group ,dim=1)
                k_aspect_att_mask[:,group] = group_mask
        # Mask generate
        k_aspect_att_mask = ~k_aspect_att_mask
        aspect_att_mask = torch.logical_or(k_aspect_att_mask.unsqueeze(dim=-1), k_aspect_att_mask.unsqueeze(dim=1))
        aspect_review_tensor = torch.stack(group_tensor_list, dim=1)

        return aspect_review_tensor, aspect_att_mask

    def review_level_network(self, x, review_mask, review_attention):
        """
        Be careful that we're using self defined attention not torch.nn.MultiheadAttention
        """
        x, _ = review_attention(x, x, mask=review_mask)

        return x 

    def forward(self, x, review_mask, lda_groups):

        batch_size, num_review, num_words, word_dim = x.shape
        x = x.reshape(-1, x.size(2), x.size(3))
        x = self.word_level_network(x, self.word_cnn_network, self.word_attention)
        x = self.sentence_level_network(x, self.sentence_cnn_network, self.sent_cross_attention, lda_groups)
        x = self.aspect_level_network(x, lda_groups, self.aspect_cross_attention)
        x = x.reshape(batch_size, num_review, -1)
        x = self.review_level_network(x, review_mask, self.review_cross_attention)

        return x