import os
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as f

from modules import FeedForwardNetwork

# Decoder module for CPTR

'''
Sinusoid positional encoding for txt sequence
'''
def sinusoid_position_encoding(seq_len, 
                    dim_model, 
                    device: torch.device = torch.device("cpu"),
                    ) -> Tensor:
    pos = torch.arange(seq_len, dtype=torch.float, device=device).reshape(1, -1, 1)
    dim = torch.arange(dim_model, dtype=torch.float, device=device).reshape(1, 1, -1)
    phase = pos / 1e4 ** (dim // dim_model)

    return torch.where(dim.long() % 2 == 0, torch.sin(phase), torch.cos(phase))


class SingleAttention(nn.Module):
    def __init__(self, dim_in, dim_k, dim_v):
        super().__init__()

        self.q = nn.Linear(dim_in, dim_k)
        self.k = nn.Linear(dim_in, dim_k)
        self.v = nn.Linear(dim_in, dim_v)

        self.attend = nn.Softmax(dim = -1)
        self.scale = dim_k ** -0.5

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        dots = torch.matmul(self.q(query), self.k(key).transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, value)
        return out


class MultiAttentionHead(nn.Module):
    def __init__(self, heads, dim_in, dim_k, dim_v):
        super().__init__()

        # E.g: nn.Linear(8 * 64, 64/256)
        self.to_out = nn.Linear(heads * dim_v, dim_in)
        self.net = nn.Sequential(*[SingleAttention(dim_in, dim_k, dim_v) for _ in heads])

    def forward(self, query, key, value):
        return self.net(query, key, value)


class SingleDecoder(nn.Module):
    def __init__(self, dim, mlp_dim, heads, dim_in, dim_k, dim_v, drop_out):
        super().__init__()
        self.masked_attention = MultiAttentionHead(heads, dim_in, dim_k, dim_v)
        self.cross_attention = MultiAttentionHead(heads, dim_in, dim_k, dim_v)
        self.ffn = FeedForwardNetwork(dim, mlp_dim, drop_out)
        self.norm = nn.LayerNorm()
    
    def forward(self, tgt, enc_output):
        '''
        Note: Please mask the target beforehand
        '''
        x = tgt
        masked_att = self.masked_attention(tgt, tgt, tgt)
        x = self.norm(masked_att + x)
        cross_att = self.cross_attention(x, enc_output, enc_output)
        x = self.norm(cross_att + x)
        mlp_out = self.ffn(x)
        # Return enc_output for stack decoders inputs
        return self.norm(mlp_out+x), enc_output


class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, dim, mlp_dim, heads, dim_head, dropout):
        super().__init__()
        module_list = [SingleDecoder(dim, mlp_dim, heads, dim_head, dropout) for _ in range(num_layers)]
        self.net = nn.Sequential(*module_list)

    def forward(self, x, enc_output):
        return self.net(x, enc_output)


class Decoder(nn.Module):
    def __init__(self, 
                vocab_size, 
                num_layers, 
                dim, 
                mlp_dim, 
                heads, 
                dim_head, 
                dropout):
        super().__init__()

        self.transformer = TransformerDecoder(num_layers, dim, mlp_dim, heads, dim_head, dropout)

        self.emb = nn.Embedding(vocab_size, dim)
        self.vocab_size = vocab_size

    


