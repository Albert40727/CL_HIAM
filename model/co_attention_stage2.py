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
        if self.training:
            q_user_1, v_item_1 = self.parallel_co_attention(user_emb_1, item_emb_1, self.W_b_1, self.W_v_1, self.W_q_1, self.w_hv_1, self.w_hq_1, self.tanh_1)
            q_user_2, v_item_2 = self.parallel_co_attention(user_emb_2, item_emb_2, self.W_b_2, self.W_v_2, self.W_q_2, self.w_hv_2, self.w_hq_2, self.tanh_2)
            q_user_3, v_item_3 = self.parallel_co_attention(user_emb_3, item_emb_3, self.W_b_3, self.W_v_3, self.W_q_3, self.w_hv_3, self.w_hq_3, self.tanh_3)
            return q_user, q_user_1, q_user_2, q_user_3, v_item, v_item_1, v_item_2, v_item_3
        return q_user, v_item