"""https://uvadlc-notebooks.readthedocs.io/en/latest/
tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html"""
import torch
import torch.nn.functional as F
from torch import nn
import math


def scale_dot_product(q,k,v,mask=None, dropout = None):
    '''
    mask: [b, num_q, num_k], True: meaningful vector, False: mean
    '''
    dk = k.size()[-1]
    # original_precision = q.dtype
    att_score = (torch.matmul(q, k.transpose(-1, -2))/math.sqrt(dk))
    if mask is not None:

        ## check the q,v dimension
        assert mask.ndim == 3, f"att mask shape should be [batch, num_q, num_k], our output {mask.shape}"
        assert mask.shape[0] == att_score.shape[0], f"mask has the wrong shape! mask:{mask.shape}, att_score:{att_score.shape}"
        assert mask.shape[-2:] == att_score.shape[-2:], f"mask has the wrong shape! mask:{mask.shape}, att_score:{att_score.shape}"
        mask = mask.unsqueeze(-3) # to make up for the absence of the multiple head dimension
        # att_score.masked_fill_(mask == 0, -1e9)
        att_score.masked_fill_(mask == 0, torch.finfo(att_score.dtype).min)
            # filled in the negative number

    att_prob = F.softmax(att_score, dim=-1)

    if dropout is not None:
        att_prob = dropout(att_prob)
    output_vec = torch.matmul(att_prob, v)

    return att_prob, output_vec


class Multihead_Cross_attention(nn.Module):
    '''
    感謝室友 Liu Yi, Chang 幫忙，不然論文早炸了
    param:
        q_input_len: q's last dim length
        kv_input_len: kv's last dim length
        output_len: The output last dim length
        num_heads: int      
    '''
    def __init__(self, q_input_len, kv_input_len, output_len, num_heads = 1, qk_hidden_len=None, dropout=0.1):
        super(Multihead_Cross_attention, self).__init__()
        self.q_input_len = q_input_len
        self.kv_input_len = kv_input_len
        self.dk = kv_input_len
        self.num_heads = num_heads
        self.qk_hidden_len = qk_hidden_len if qk_hidden_len is not None else q_input_len
        self.output_len = output_len
        assert self.output_len%self.num_heads == 0, "the output_len should be the multiple of num_heads!"
        self.v_hidden_len = output_len//num_heads
        self.q_proj = nn.Linear(q_input_len, self.num_heads*self.qk_hidden_len)
        self.kv_proj = nn.Linear(kv_input_len, self.num_heads*(self.qk_hidden_len+self.v_hidden_len))
        self.o_proj = nn.Linear(self.output_len, self.output_len)
        self.dropout = nn.Dropout(dropout)


    def forward(self, q_input_data, kv_input_data, mask=None):
        # q_input_data:[batch_size, q_candds, q_input_len]
        # kv_input_data:[batch_size, kv_candds, kv_input_len]
        # mask: [batch_size, q_candds, k_candds]
        batch_size, q_num_candidates, q_input_len = q_input_data.size()
        kv_batch_size, kv_num_candidates, kv_input_len = kv_input_data.size()
        assert batch_size == kv_batch_size, "batch size are not equal"

        q = self.q_proj(q_input_data)

        # q: [batch_size, q_candds, num_heads*qv_hidden_len]
        q = q.reshape(batch_size,q_num_candidates, self.num_heads, self.qk_hidden_len)
        # q: [batch_size, q_candds, num_heads, qv_hidden_len]
        q = q.permute(0,2,1,3)
        # q: [batch_size, num_heads, q_candds, qv_hidden_len]
        kv = self.kv_proj(kv_input_data)
        # kv: [batch_size, kv_candds, num_heads*(qv_hidden_len+v_hidden_len) ]
        kv = kv.reshape(batch_size, kv_num_candidates, self.num_heads, self.qk_hidden_len+self.v_hidden_len)
        # kv: [batch_size, kv_candds, num_heads, (qv_hidden_len+v_hidden_len) ]
        kv = kv.permute(0,2,1,3)
        # kv: [batch_size, num_heads, kv_candds, (qv_hidden_len+v_hidden_len) ]
        k, v = torch.tensor_split(kv, (self.qk_hidden_len,),dim = -1)
        # k: [batch_size, num_heads, kv_candds, qv_hidden_len ]
        # v: [batch_size, num_heads, kv_candds, v_hidden_len ]

        att_prob, output_data = scale_dot_product(q,k,v, mask, dropout=self.dropout)
        # att_prob: [batch_size, num_heads, q_candds, kv_candds]
        # output_data: [batch_size, num_heads, kv_candds, v_hidden_len]
        output_data = output_data.permute(0,2,1,3)
        # output_data: [batch_size, kv_candds, num_heads, v_hidden_len]
        output_data = output_data.reshape(batch_size, q_num_candidates, self.num_heads*self.v_hidden_len)
        # output_data: [batch_size, kv_candds, num_heads*v_hidden_len]
        output_data = self.o_proj(output_data)
        # output_data: [batch_size, kv_candds, num_heads*v_hidden_len]
        
        # To zero those padding query
        if mask is not None:
            padded_query_mask = ~torch.any(mask, dim=-1)# [b, q_num_candidates], True if needs to be pad zero
            output_data[padded_query_mask,:] = 0

        return output_data, att_prob


def test_Multihead_Cross_attention():
    batch_size = 2
    num_real_q = [0,0]
    num_real_k = [0,0]
    q_num_candidates = max(num_real_q)
    k_num_candidates = max(num_real_k)
    # create agent mask
    mask = torch.ones((batch_size,q_num_candidates,k_num_candidates), dtype=torch.bool)
    
    for i in range(batch_size):
        mask[i,num_real_q[i]:,:] = False
        mask[i,:,num_real_k[i]:] = False

    q_input_len = 6
    kv_input_len = 6
    output_len = 8
    q_data_vec = torch.randn(batch_size, q_num_candidates, q_input_len)
    k_data_vec = torch.randn(batch_size, k_num_candidates, kv_input_len)
    num_heads = 2
    model = Multihead_Cross_attention(q_input_len=q_input_len, 
                                kv_input_len=kv_input_len,
                                output_len= output_len,
                                num_heads=num_heads,
                                dropout=0.1
                            )
    output_data, att_prob = model(q_data_vec, k_data_vec, mask=mask)

    # # try with no mask and observe the mask
    # att_prob, output = model(q_data_vec, k_data_vec)
    assert att_prob.shape == (batch_size, num_heads, q_num_candidates, k_num_candidates), f"att_prob shape is wrong\
            output: {att_prob.shape}  desired: ({batch_size},{num_heads},{q_num_candidates},{k_num_candidates})"
    assert output_data.shape == (batch_size, q_num_candidates, output_len), f"output shape is wrong\
            output: {output_data.shape}  desired: ({batch_size}, {q_num_candidates},{output_len})"
    

    # check mask effect
    for i in range(batch_size):
        if num_real_q[i] == 0:
            assert (output_data[i]==0).all(), f"output_data should be all zeros for padding query"
            print(f"{i}: correct!")
        else:
            assert (att_prob[i,:,num_real_k[i]:] == 0).all(), f"att_prob should be zeros for padding k for i:{i}, {att_prob}"
            print(f"{i}: correct!")
    print("output_data:", output_data.shape)
    print(output_data)
    print("att_prob:", att_prob.shape)
    print(att_prob)

    # # try backward to check if alright
    loss = torch.sum(output_data)
    loss.backward()
    return

# print(list(enc.parameters()))
def test_model():
    batch_size = 4
    q_num_candidates = 10
    k_num_candidates = 9
    q_input_len = 10
    kv_input_len = 13
    hidden_len = 10
    q_data_vec = torch.randn(batch_size, q_num_candidates, q_input_len)
    k_data_vec = torch.randn(batch_size, k_num_candidates, kv_input_len)

    model = Multihead_Cross_attention(q_input_len=q_input_len, 
                                kv_input_len=kv_input_len,
                                num_heads=2,
                                qk_hidden_len=hidden_len,
                                v_hidden_len=hidden_len
                            )
    agt_mask = torch.zeros(batch_size,q_num_candidates,k_num_candidates)
    for batch in range(batch_size):
        for i in range(agt_mask.shape[1]):
            for j in range(agt_mask.shape[2]):
                if (i+j)%batch_size == batch:
                    agt_mask[batch,i,j] = 1

if __name__ == '__main__':
    test_Multihead_Cross_attention()