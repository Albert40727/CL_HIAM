import torch.nn as nn
import torch.nn.functional as F
import torch

class HianModel(nn.Module):
    """
    B: Batch, N: User/Item max review, W: Max Word, S:Max Sentence, A:Aspect, D:Emb Dimension 

            input:  x: 32, N, 250, 768 (B, N, W*S, D)
                            |reshape
                            v
                    x: 32*N, 250, 768 (B*N, W*S, D)
                            |
                            |w_cnn_network + attention
                            v
                    x: 32*N, 10, D  (B*N, S, D)
                            |
                            |s_cnn_network + attention
                            v 
                    x: 32*N, 10, D  (B*N, S, D)
                            |
                            |get_aspect_emb_from_sent (LDA) 
                            v             
                    x: 32*N, 6, D   (B*N, A, D)
                            |
                            |aspect attention
                            v
                    x: 32, N, D  (B, N, D)
                            |
                            |review_network
                            v  
                        x: 32, N, D
                            |
                            |co_attention (Outside HianModel)
                            v
                        x: 32, D                 

    (Some emb might be permuted during training due to the Conv1d input format)
    Word Emb:               torch.Size([50, 250, 768])
    Sentence Emb:           torch.Size([50, 10, 512])
    Weighted Sentence Emb:  torch.Size([50, 10, 512])
    Aspect Emb:             torch.Size([50, 6, 512])
    Aspect Review Emb:      torch.Size([50, 512])
    User/Item Emb:          torch.Size([512]) (after co-attention) 
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
        self.word_attention = nn.MultiheadAttention(512, num_heads=1, batch_first=True)
        
        # Sentence-Level Network
        self.sent_pad_size = int((self.args["sentence_cnn_ksize"]-1)/2)
        self.sentence_cnn_network = nn.Sequential(
            nn.Conv1d(512, 512, self.args["sentence_cnn_ksize"]),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )
        self.sentence_attention = nn.MultiheadAttention(512, num_heads=1, batch_first=True)

        # Aspect-Level Network
        self.lda_group_num = self.args["lda_group_num"]
        self.aspect_attention = nn.MultiheadAttention(512, num_heads=1, batch_first=True)

        
        # Review-Level Network
        self.review_attention = nn.MultiheadAttention(512, num_heads=1, batch_first=True)

    def word_level_network(self, x, word_cnn, word_attention):
        x = torch.permute(x, (0, 2, 1))
        x = F.pad(x, (self.word_pad_size, self.word_pad_size), "constant", 0) # same to keras: padding = same
        x = word_cnn(x)
        x = torch.permute(x, [0, 2, 1])
        attn_output = word_attention(x, key=x, value=x, need_weights=False)
        x = x * attn_output[0] 
        x = self.word_weighted_sum(x, self.args["max_sentence"])
        x = torch.permute(x, [0, 2, 1])
        return x
    
    def sentence_level_network(self, x, sent_cnn, sent_attention, lda_groups):
        x = F.pad(x, (self.sent_pad_size, self.sent_pad_size), "constant", 0) # same to keras: padding = same
        x = sent_cnn(x)
        x = torch.permute(x, [0, 2, 1])
        sent_mask = (lda_groups == False).reshape(-1, lda_groups.size(2))
        attn_output = sent_attention(x, x, x, key_padding_mask=sent_mask, need_weights=False)
        x = x * attn_output[0]
        return x 
    
    def aspect_level_network(self, x, lda_groups, aspect_attention):
        lda_groups = lda_groups.reshape(-1, lda_groups.size(2))
        x, aspect_att_mask = self.get_aspect_emb_from_sent(x, lda_groups, self.lda_group_num) 
        attn_output = aspect_attention(x, x, x, key_padding_mask=aspect_att_mask.to(self.args["device"]), need_weights=False)
        x = torch.sum(x * attn_output[0], 1) #weighted sum
        return x

    def review_level_network(self, x, review_attetion, *, batch_size):
        attn_output = review_attetion(x, x, x, need_weights=False)
        x = x * attn_output[0]
        x = x.reshape(batch_size, -1, x.size(1))
        return x 

    
    def word_weighted_sum(self, input_tensor, max_sentence):
        """
        Weighted sum words' emb into sentences' emb.
        """
        batch, word_num, word_dim = input_tensor.shape
        input_tensor = input_tensor.reshape(batch, max_sentence, -1, word_dim)
        sentence_tensor = torch.sum(input_tensor, dim=2)

        return sentence_tensor
    
    def get_aspect_emb_from_sent(self, input_tensor, lda_groups, group_num):
        """
        Weighted sum sentences' emb according to their LDA groups respectively.  
        """
        aspect_att_mask = torch.zeros((input_tensor.size(0), group_num))
        lda_groups = torch.unsqueeze(lda_groups, dim=-1)
        group_tensor_list = []

        for group in range(group_num):
            mask = lda_groups == group
            mask_sum = torch.sum(mask, dim=1)
            mask_sum[mask_sum == 0] = 1
            group_tensor = torch.where(mask , input_tensor, 0.0)
            group_tensor = torch.sum(group_tensor, dim = 1)
            group_tensor = group_tensor/mask_sum
            group_tensor_list.append(group_tensor)

            # Mask generate
            group_mask = torch.any((lda_groups.squeeze(dim=-1))==group ,dim=1)
            aspect_att_mask[:,group] = group_mask

        aspect_review_tensor = torch.stack(group_tensor_list).permute(1, 0, 2)  
        aspect_att_mask = torch.where(aspect_att_mask==0, 1., 0.)  

        return aspect_review_tensor, aspect_att_mask
    

    def forward(self, x, lda_groups):

        batch_size, num_review, num_words, word_dim = x.shape
        x = x.reshape(-1, x.size(2), x.size(3))
        x = self.word_level_network(x, self.word_cnn_network, self.word_attention)
        x = self.sentence_level_network(x, self.sentence_cnn_network, self.sentence_attention, lda_groups)
        x = self.aspect_level_network(x, lda_groups, self.aspect_attention)
        x = self.review_level_network(x, self.review_attention, batch_size=batch_size)

        return x