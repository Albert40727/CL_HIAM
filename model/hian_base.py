import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class HianModel(nn.Module):
    """
    B: Batch, D:Emb Dimension, W: Word, S:Sentence, A:Aspect
                          input   
                            |
                            v
                    x: 32, 728, 250 (B, D, W*S)
                            |
                            |w_cnn_network + attention
                            v
                    x: 32, 256, 10  (B, D, S)
                            |
                            |s_cnn_network + attention
                            v 
                    x: 32, 10, 256  (B, D, S)
                            |
                            |(LDA) + attention
                            v             
                    x: 32, 6, 256   (B, D, A)                
    """

    def __init__(self, args):
        super().__init__()
        self.args = args

        #Word-Level Network
        self.word_pad_size = int((self.args["word_cnn_ksize"]-1)/2)
        self.word_cnn_network = nn.Sequential(
            nn.Conv1d(768, 256, self.args["word_cnn_ksize"]),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )
        self.word_attention = nn.MultiheadAttention(256, num_heads=1, batch_first =True)
        
        #Sentence-Level Network
        self.sent_pad_size = int((self.args["sentence_cnn_ksize"]-1)/2)
        self.sentence_cnn_network = nn.Sequential(
            nn.Conv1d(256, 256, self.args["sentence_cnn_ksize"]),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )
        self.sentence_attention = nn.MultiheadAttention(256, num_heads=1)

        #Aspect-Level Network
        self.lda_group_num = self.args["lda_group_num"]
        self.aspect_attention = nn.MultiheadAttention(256, num_heads=1)

        
        #Review-Level Network
        self.aspect_attention = nn.MultiheadAttention(256, num_heads=1)
    
    def word_level_mean(self, input_tensor, max_word):
        """
        input:  input_tensor = (batch, word*sentence, emb_dim)
        output: tensor(batch, sentence, emb_dim)
        """
        tensor_list = []
        for i in range(0, input_tensor.size(dim=1), max_word):
            start = i
            end = start + max_word
            # torch.mean will reduce a dimension, so it need to be unsqueezed then concated
            tensor_list.append(torch.unsqueeze(torch.mean(input_tensor[:, start:end, :], dim=1), dim=1))
        sentence_tensor = torch.cat(tensor_list, dim=1)
        return sentence_tensor
    
    def aspect_level_mean(self, input_tensor, lda_groups, group_num):
        """
        
        """
        batch_review_tensor = []
        batch_review_tensor = torch.zeros(input_tensor.size(0), group_num, input_tensor.size(2))
        for i, groups in enumerate(lda_groups):
            avg_list = [[] for i in range(group_num)]
            for group in range(group_num):
                avg_list[group].append(input_tensor[i, np.where(groups == group), :])
            for j, tensors in enumerate(avg_list):
                batch_review_tensor[i, j, :] = torch.mean(torch.stack(tensors))
        batch_review_tensor = batch_review_tensor.to(self.args["device"])
            

        # //////////////////////////////////////////////////////////////
        # for i, groups in enumerate(lda_groups):
        #     avg_list = [[] for i in range(group_num)]
            # for j, group in enumerate(groups):
            #     avg_list[group].append(input_tensor[i, j, :])
            # for k, group_tensors in enumerate(avg_list):
            #     if not avg_list[k]:
            #         avg_list[k] = torch.zeros(256).to(self.args["device"])
            #     else:
            #         avg_list[k] = torch.mean(torch.stack(group_tensors), dim=0).to(self.args["device"])
        #     single_review_tensor = torch.stack(avg_list) # dim: group_num*dim
        #     batch_review_tensor.append(single_review_tensor)
        # batch_review_tensor = torch.stack(batch_review_tensor).to(self.args["device"])
            
        return batch_review_tensor




    def forward(self, x, lda_groups):

        # Word-Level Network
        x = F.pad(x, (self.word_pad_size, self.word_pad_size), "constant", 0) # same to keras: padding = same
        x = self.word_cnn_network(x)
        x = torch.permute(x, [0, 2, 1])
        attn_output = self.word_attention(x, key=x, value=x, need_weights=False)
        x = x * attn_output[0] 
        x = self.word_level_mean(x, self.args["max_word"])
        x = torch.permute(x, [0, 2, 1])

        #Sentence-Level Network
        x = F.pad(x, (self.sent_pad_size, self.sent_pad_size), "constant", 0) # same to keras: padding = same
        x = self.sentence_cnn_network(x)
        x = torch.permute(x, [0, 2, 1])
        attn_output = self.sentence_attention(x, key=x, value=x, need_weights=False)
        x = x * attn_output[0]

        #Aspect-Level Network
        x = self.aspect_level_mean(x, lda_groups, self.lda_group_num) 
        attn_output = self.aspect_attention(x, key=x, value=x, need_weights=False)
        x = x * attn_output[0]

        #Review-Level Network
        attn_output = self.aspect_attention(x, key=x, value=x, need_weights=False)
        x = x * attn_output[0]

        return x