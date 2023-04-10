import torch
import torch.nn as nn
import torch.nn.functional as fn
from .co_attention import CoattentionNet

# git repo src: https://github.com/SkyOL5/VQA-CoAttention/blob/master/coatt/coattention_net.py

class CoattentionNetStage2(CoattentionNet):
    """
    Predicts an answer to a question about an image using the Hierarchical Question-Image Co-Attention
    for Visual Question Answering (Lu et al, 2017) paper.
    """
    def __init__(self, args, embed_dim=512, k=30): # Original paper: k=30
        super().__init__(args, embed_dim, k)

        self.tanh_1 = nn.Tanh()
        self.W_b_1 = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.W_v_1 = nn.Parameter(torch.randn(k, embed_dim))
        self.W_q_1 = nn.Parameter(torch.randn(k, embed_dim))
        self.w_hv_1 = nn.Parameter(torch.randn(k, 1))
        self.w_hq_1 = nn.Parameter(torch.randn(k, 1))

        self.tanh_2 = nn.Tanh()
        self.W_b_2 = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.W_v_2 = nn.Parameter(torch.randn(k, embed_dim))
        self.W_q_2 = nn.Parameter(torch.randn(k, embed_dim))
        self.w_hv_2 = nn.Parameter(torch.randn(k, 1))
        self.w_hq_2 = nn.Parameter(torch.randn(k, 1))

        self.tanh_3 = nn.Tanh()
        self.W_b_3 = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.W_v_3 = nn.Parameter(torch.randn(k, embed_dim))
        self.W_q_3 = nn.Parameter(torch.randn(k, embed_dim))
        self.w_hv_3 = nn.Parameter(torch.randn(k, 1))
        self.w_hq_3 = nn.Parameter(torch.randn(k, 1))

    def forward(self, user_emb, item_emb, user_emb_1=None, user_emb_2=None, user_emb_3=None, item_emb_1=None, item_emb_2=None, item_emb_3=None):
        q_user, v_item = self.parallel_co_attention(user_emb, item_emb, self.W_b, self.W_v, self.W_q, self.w_hv, self.w_hq, self.tanh)
        if user_emb_1 != None or item_emb_1 != None:
            q_user_1, v_item_1 = self.parallel_co_attention(user_emb_1, item_emb_1, self.W_b_1, self.W_v_1, self.W_q_1, self.w_hv_1, self.w_hq_1, self.tanh_1)
        if user_emb_2 != None or item_emb_2 != None:
            q_user_2, v_item_2 = self.parallel_co_attention(user_emb_2, item_emb_2, self.W_b_2, self.W_v_2, self.W_q_2, self.w_hv_2, self.w_hq_2, self.tanh_2)
        if user_emb_3 != None or item_emb_3 != None:
            q_user_3, v_item_3 = self.parallel_co_attention(user_emb_3, item_emb_3, self.W_b_3, self.W_v_3, self.W_q_3, self.w_hv_3, self.w_hq_3, self.tanh_3)

        return q_user, q_user_1, q_user_2, q_user_3, v_item, v_item_1, v_item_2, v_item_3


    def parallel_co_attention(self, Q, V, W_b, W_v, W_q, w_hv, w_hq, tanh):  
        # Original paper:   V : B x 512 x 196(Seq), Q : B x L x 512
        # Our paper:        V : B x 50 x 256, Q : B x 10 x 256
        # V = item, Q=User

        V = V.permute(0, 2, 1) # permute to fit original paper's input format

        C = torch.matmul(Q, torch.matmul(W_b, V)) # B x L x 196

        H_v = tanh(torch.matmul(W_v, V) + torch.matmul(torch.matmul(W_q, Q.permute(0, 2, 1)), C))                            # B x k x 196
        H_q = tanh(torch.matmul(W_q, Q.permute(0, 2, 1)) + torch.matmul(torch.matmul(W_v, V), C.permute(0, 2, 1)))           # B x k x L

        a_v = fn.softmax(torch.matmul(torch.t(w_hv), H_v), dim=2) # B x 1 x 196
        a_q = fn.softmax(torch.matmul(torch.t(w_hq), H_q), dim=2) # B x 1 x L

        v = torch.squeeze(torch.matmul(a_v, V.permute(0, 2, 1))) # B x 512
        q = torch.squeeze(torch.matmul(a_q, Q))                  # B x 512

        return q, v