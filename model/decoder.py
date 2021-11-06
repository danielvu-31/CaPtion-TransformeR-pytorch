import os
import math
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as f

from modules import FeedForwardNetwork, PositionalEncoding

# Decoder module for CPTR
class SingleAttention(nn.Module):
    def __init__(self, dim_in, dim_k, dim_v):
        super().__init__()

        self.q = nn.Linear(dim_in, dim_k)
        self.k = nn.Linear(dim_in, dim_k)
        self.v = nn.Linear(dim_in, dim_v)

        self.attend = nn.Softmax(dim = -1)
        self.scale = dim_k ** -0.5

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask = None) -> Tensor:
        dots = torch.matmul(self.q(query), self.k(key).transpose(-1, -2)) * self.scale
        if mask is not None:  # For txt decoders branch only
            attn = dots.masked_fill(mask == 0, -1e9)
        attn = self.attend(dots)
        out = torch.matmul(attn, value)
        return out


class MultiAttentionHead(nn.Module):
    def __init__(self, heads, dim_in, dim_k, dim_v):
        super().__init__()

        # E.g: nn.Linear(8 * 64, 64/256)
        self.to_out = nn.Linear(heads * dim_v, dim_in)
        self.net = nn.Sequential(*[SingleAttention(dim_in, dim_k, dim_v) for _ in heads])

    def forward(self, query, key, value, mask = None):
        return self.net(query, key, value, mask)


class SingleDecoder(nn.Module):
    def __init__(self, dim, mlp_dim, heads, dim_in, dim_k, dim_v, drop_out):
        super().__init__()
        self.masked_attention = MultiAttentionHead(heads, dim_in, dim_k, dim_v)
        self.cross_attention = MultiAttentionHead(heads, dim_in, dim_k, dim_v)
        self.ffn = FeedForwardNetwork(dim, mlp_dim, drop_out)
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, tgt, enc_output, slf_attn_mask):
        '''
        Note: Please mask the target beforehand
        '''
        x = tgt
        masked_att = self.masked_attention(tgt, tgt, tgt, slf_attn_mask)
        x = self.norm(masked_att + x)
        cross_att = self.cross_attention(x, enc_output, enc_output, None)
        x = self.norm(cross_att + x)
        mlp_out = self.ffn(x)
        # Return enc_output for stack decoders inputs
        return self.norm(mlp_out+x), enc_output


class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, dim, mlp_dim, heads, dim_head, dropout):
        super().__init__()
        module_list = [SingleDecoder(dim, mlp_dim, heads, dim_head, dropout) for _ in range(num_layers)]
        self.net = nn.Sequential(*module_list)

    def forward(self, tgt, enc_output, slf_attn_mask):
        return self.net(tgt, enc_output, slf_attn_mask)


class Decoder(nn.Module):
    def __init__(self, 
                vocab_size, # Vocab Size of tgt dictionary
                num_layers, # Num of stacked decoders
                dim, # Dimension of output
                mlp_dim, # Dimension of FFN layers
                heads, # Multi-head attention
                dim_head,
                max_seq_len, # Max sequence length
                tgt_pad_idx,
                dropout):
        super().__init__()

        self.tgt_pad_idx = tgt_pad_idx

        self.transformer = TransformerDecoder(num_layers, dim, mlp_dim, heads, dim_head, dropout)
        self.pe = PositionalEncoding(dim, max_seq_len)

        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, dim, padding_idx=tgt_pad_idx)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(dim, eps=1e-6)
    
    def forward(self, tgt, enc_output, slf_attn_mask):
        # Initial Hidden states
        # Embedding + Positional Encoding
        dec_output = self.embed(tgt)
        dec_output = self.layer_norm(self.dropout(self.pe(dec_output)))

        out_features, _ = self.transformer(dec_output, enc_output, slf_attn_mask)
        return out_features
        
        