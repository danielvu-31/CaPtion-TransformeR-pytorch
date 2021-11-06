import os
import sys

import torch
import torch.nn as nn

from decoder import Decoder
from encoder import Encoder
from modules import get_pad_mask, get_subsequent_mask

class CPTR(nn.Module):
    def __init__(self,
                image_size,
                vocab_size, # Vocab Size of tgt dictionary
                patch_size, 
                dim, # Dimension of output
                num_encoders, # Num of stacked encoders
                num_decoders, # Num of stacked decoders
                mlp_dim,
                heads, # Multi-head attention, 
                dim_head,
                max_seq_len, # Max sequence length
                tgt_pad_idx,
                pool = 'cls', 
                channels = 3,
                dropout = 0., 
                emb_dropout = 0.,
                trg_emb_prj_weight_sharing=False):
    
        self.encoder = Encoder(image_size, 
                                patch_size, 
                                dim, 
                                num_encoders,
                                heads, 
                                mlp_dim, 
                                pool, 
                                channels, 
                                dim_head, 
                                dropout, 
                                emb_dropout)
        
        self.decoder = Decoder(vocab_size,
                                num_decoders, 
                                dim, # Dimension of output
                                mlp_dim, # Dimension of FFN layers
                                heads, # Multi-head attention
                                dim_head,
                                max_seq_len, # Max sequence length
                                tgt_pad_idx,
                                dropout)
        
        self.word_linear = nn.Linear(dim, vocab_size, bias=False)
        self.tgt_pad_idx = tgt_pad_idx

        # Xavier Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 
        
        if trg_emb_prj_weight_sharing:
            # Share the weight between target word embedding & last dense layer
            self.word_linear.weight = self.decoder.embed.weight

    def forward(self, image, tgt):
        enc_output = self.encoder(image)
        trg_mask = get_pad_mask(tgt, self.tgt_pad_idx) & get_subsequent_mask(tgt)
        out_features = self.decoder(tgt, enc_output, trg_mask)
        
        return nn.Softmax(self.word_linear(out_features))