import torch
import torch.nn as nn
import torch.nn.functional as fn

# git repo src: https://github.com/SkyOL5/VQA-CoAttention/blob/master/coatt/coattention_net.py

class CoattentionNet(nn.Module):
    """
    Predicts an answer to a question about an image using the Hierarchical Question-Image Co-Attention
    for Visual Question Answering (Lu et al, 2017) paper.
    """
    def __init__(self, args, embed_dim=512, k=30): # Original paper: k=30
        super().__init__()
        self.tanh = nn.Tanh()
        self.W_b = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.W_v = nn.Parameter(torch.randn(k, embed_dim))
        self.W_q = nn.Parameter(torch.randn(k, embed_dim))
        self.w_hv = nn.Parameter(torch.randn(k, 1))
        self.w_hq = nn.Parameter(torch.randn(k, 1))

    def forward(self, user_emb, item_emb):
        q_user, v_item = self.parallel_co_attention(user_emb, item_emb)

        return q_user, v_item


    def parallel_co_attention(self, Q, V):  
        # Original paper:   V : B x 512 x 196(Seq), Q : B x L x 512
        # Our paper:        V : B x 50 x 256, Q : B x 10 x 256
        # V = item, Q=User

        V = V.permute(0, 2, 1) # permute to fit original paper's input format

        C = torch.matmul(Q, torch.matmul(self.W_b, V)) # B x L x 196

        H_v = self.tanh(torch.matmul(self.W_v, V) + torch.matmul(torch.matmul(self.W_q, Q.permute(0, 2, 1)), C))                            # B x k x 196
        H_q = self.tanh(torch.matmul(self.W_q, Q.permute(0, 2, 1)) + torch.matmul(torch.matmul(self.W_v, V), C.permute(0, 2, 1)))           # B x k x L

        a_v = fn.softmax(torch.matmul(torch.t(self.w_hv), H_v), dim=2) # B x 1 x 196
        a_q = fn.softmax(torch.matmul(torch.t(self.w_hq), H_q), dim=2) # B x 1 x L

        v = torch.squeeze(torch.matmul(a_v, V.permute(0, 2, 1))) # B x 512
        q = torch.squeeze(torch.matmul(a_q, Q))                  # B x 512

        return q, v