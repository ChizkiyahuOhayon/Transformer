# implement self attention
import torch
import torch.nn as nn
class SelfAttention(nn.Module):
    def __init__(self, dim_a, d_k, d_v):
        # dim_a: length of embedding
        # d_k, d_v: length of each key and value(d_k equals to d_q)
        super(SelfAttention, self).__init__()
        self.dim_a = dim_a
        self.d_k = d_k
        self.d_q = d_k
        self.d_v = d_v
        self.scale = d_k ** (-0.5)
        self.w_k = nn.Linear(self.dim_a, self.d_k)
        self.w_q = nn.Linear(self.dim_a, self.d_q)
        self.w_v = nn.Linear(self.dim_a, self.d_v)
    # three W(a->q, a->k, a->v)
    def forward(self, a):  # a: (batch_size, num_embedding, dim_a)
        query = self.w_q(a)
        key = self.w_k(a)
        value = self.w_v(a)

        attention = query @ key.transpose(-2, -1)
        attention_scaled = attention * self.scale
        attention_softmax = attention.softmax(-1)
        x = attention_softmax @ value
        return x

a = torch.randn((1, 3, 2))
self_attention = SelfAttention(2, 3, 2)
b = self_attention(a)
print(a)
print(b)
