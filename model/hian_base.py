import torch.nn as nn
import torch.nn.functional as F
import torch


class HianModel(nn.Module):
    """
    input   x: 32, 728, 250 (Batch, emb_dim, word*sentence)
                            |
                            |word_cnn_network + attention
                            v
                    x: 32, 256, 250
                            |
                            |sentence_cnn_network + attention
                            v 

    """
    def __init__(self, args):
        super().__init__()
        self.args = args

        #Word-Level Network
        self.word_ksize = args["word_cnn_ksize"]
        self.word_cnn_network = nn.Sequential(
            nn.Conv1d(768, 256, self.word_ksize),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )
        self.word_attention = nn.MultiheadAttention(256, num_heads=1, batch_first =True)
        
        #Sentence-Level Network
        self.word_ksize = args["sentence_cnn_ksize"]
        self.sentence_cnn_network = nn.Sequential(
            nn.Conv1d(256, 256, self.word_ksize),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )
        self.sentence_attention = nn.MultiheadAttention(256, num_heads=1)
    
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
    
    def sentence_level_mean(self, input_tensor, max_sentence):
        """
        input:  input_tensor = (batch, sentence, emb_dim)
        output: tensor(batch, sentence, emb_dim)
        """
        tensor_list = []
        for i in range(0, input_tensor.size(dim=1), max_sentence):
            start = i
            end = start + max_sentence
            # torch.mean will reduce a dimension, so it need to be unsqueezed then concated
            tensor_list.append(torch.unsqueeze(torch.mean(input_tensor[:, start:end, :], dim=1), dim=1))
        sentence_tensor = torch.cat(tensor_list, dim=1)
        return sentence_tensor
    


    def forward(self, x):

        # Word-Level Network
        x = F.pad(x, (1,1), "constant", 0) # same to keras: padding = same, if kernal_size =3
        x = self.word_cnn_network(x)
        x = torch.permute(x, [0, 2, 1])
        attn_output = self.word_attention(x, key=x, value=x, need_weights=False)
        x = x * attn_output[0] 
        x = self.word_level_mean(x, self.args["max_word"])
        x = torch.permute(x, [0, 2, 1])

        #Sentence_Level Network
        x = F.pad(x, (1,1), "constant", 0) # same to keras: padding = same, if kernal_size =3
        x = self.sentence_cnn_network(x)
        x = torch.permute(x, [0, 2, 1])
        attn_output = self.sentence_attention(x, key=x, value=x, need_weights=False)
        x = x * attn_output[0]
        x = torch.permute(x, [0, 2, 1])


        return x