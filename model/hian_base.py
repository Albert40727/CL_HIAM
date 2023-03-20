import torch.nn as nn
import torch.nn.functional as F
import torch


class HianModel(nn.Module):
    """
    B: Batch, D:Output Dimension, W: Word, S:Sentence, A:Aspect, N: User/Item max review num
                          input   
                            |
                            v
                    x: 32, N, 250, 768 (B, W*S, D)
                            |
                            |w_cnn_network + attention
                            v
                    x: 32, N, 10, D  (B, S, D)
                            |
                            |s_cnn_network + attention
                            v 
                    x: 32, N, 10, D  (B, S, D)
                            |
                            |(LDA) + attention
                            v             
                    x: 32, N, 6, D   (B, A, D)
                            |
                            |aspect weighted sum
                            v
                        x: 32, N, D
                            |
                            |Review_network
                            v  
                        x: 32, D               
    """
    """
    Word Emb:               torch.Size([50, 250, 768])
    Sentence Emb:           torch.Size([50, 10, 512])
    Weighted Sentence Emb:  torch.Size([50, 10, 256])
    Aspect Emb:             torch.Size([50, 6, 256])
    Aspect Review Emb:      torch.Size([50, 256])
    User/Item Emb:          torch.Size([256])
    """

    def __init__(self, args):
        super().__init__()
        self.args = args

        #Word-Level Network
        self.word_pad_size = int((self.args["word_cnn_ksize"]-1)/2)
        self.word_cnn_network = nn.Sequential(
            nn.Conv1d(768, 512, self.args["word_cnn_ksize"]),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )
        self.word_attention = nn.MultiheadAttention(512, num_heads=1, batch_first =True)
        
        #Sentence-Level Network
        self.sent_pad_size = int((self.args["sentence_cnn_ksize"]-1)/2)
        self.sentence_cnn_network = nn.Sequential(
            nn.Conv1d(512, 256, self.args["sentence_cnn_ksize"]),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )
        self.sentence_attention = nn.MultiheadAttention(256, num_heads=1)

        #Aspect-Level Network
        self.lda_group_num = self.args["lda_group_num"]
        self.aspect_attention = nn.MultiheadAttention(256, num_heads=1)

        
        #Review-Level Network
        self.review_attention = nn.MultiheadAttention(256, num_heads=1)
    
    def word_weighted_sum(self, input_tensor, max_word):
        """
        Weighted sum words' emb into sentences' emb.
        input:  input_tensor = (max_review, word*sentence, emb_dim)
        output: tensor(max_review, sentence, emb_dim)
        """
        tensor_list = []
        for i in range(0, input_tensor.size(dim=1), max_word):
            start, end = i, i+max_word
            tensor_list.append(torch.sum(input_tensor[:, start:end, :], dim=1))
        sentence_tensor = torch.stack(tensor_list, dim=1)
        return sentence_tensor
    
    def get_aspect_emb_from_sent(self, input_tensor, lda_groups, group_num):
        """
        Weighted sum sentences' emb according to their LDA groups respectively.  
        input:  input_tensor = (max_review, sentence, emb_dim)
        output: tensor(max_review, aspect, emb_dim)
        """
        aspect_review_tensor = torch.zeros(input_tensor.size(0), group_num, input_tensor.size(2)).to(self.args["device"])
        for i, groups in enumerate(lda_groups):
            for group in range(group_num):
                # torch.nonzero(condition, as_tuple=True) is identical to torch.where(condition).
                select_indices = torch.nonzero(groups==group, as_tuple=False).squeeze().to(self.args["device"])
                aspect_review_tensor[i, group, :] = torch.sum(torch.index_select(input_tensor[i], 0, select_indices), 0)        
        return aspect_review_tensor

    def forward(self, xs, lda_groups):

        x_list = []

        for i, x in enumerate(xs):

            # Word-Level Network
            x = torch.permute(x, (0, 2, 1))
            x = F.pad(x, (self.word_pad_size, self.word_pad_size), "constant", 0) # same to keras: padding = same
            x = self.word_cnn_network(x)
            x = torch.permute(x, [0, 2, 1])
            attn_output = self.word_attention(x, key=x, value=x, need_weights=False)
            x = x * attn_output[0] 
            x = self.word_weighted_sum(x, self.args["max_word"])
            x = torch.permute(x, [0, 2, 1])

            #Sentence-Level Network
            x = F.pad(x, (self.sent_pad_size, self.sent_pad_size), "constant", 0) # same to keras: padding = same
            x = self.sentence_cnn_network(x)
            x = torch.permute(x, [0, 2, 1])
            attn_output = self.sentence_attention(x, key=x, value=x, need_weights=False)
            x = x * attn_output[0]

            #Aspect-Level Network
            x = self.get_aspect_emb_from_sent(x, lda_groups[i], self.lda_group_num) 
            attn_output = self.aspect_attention(x, key=x, value=x, need_weights=False)
            x = torch.sum(x * attn_output[0], 1) #weighted sum

            #Review-Level Network
            attn_output = self.review_attention(x, key=x, value=x, need_weights=False)
            x = x * attn_output[0]
            x_list.append(x)

        result_x = torch.stack(x_list)
        return result_x