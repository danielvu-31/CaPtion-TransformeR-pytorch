# Code based on lucidrains's pytorch visual transformer
# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py

import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes
class FeedForwardNetwork(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.ffn(x)


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        # Multi head check
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads

        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim = -1)

        # Get qkv vector before split
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)') # (b n innerdim)
        return self.to_out(out)


class SingleTransformer(nn.Module):
    def __init__(self, dim, mlp_dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        self.multi_head_attention = Attention(dim, heads, dim_head, dropout)
        # Could try nn.MultiheadAttention class
        # self.multi_head_attention = nn.MultiheadAttention(dim, heads, dropout)
        self.norm = nn.LayerNorm()
        self.ffn = FeedForwardNetwork(dim, mlp_dim, dropout)

    def forward(self, x):
        att = self.multi_head_attention(x)
        x = self.norm(att + x)
        mlp_out = self.ffn(x)
        x = self.norm(mlp_out + x)
        return x


class Transformer(nn.Module):
    def __init__(self, num_layers, dim, mlp_dim, heads, dim_head, dropout):
        super().__init__()
        module_list = [SingleTransformer(dim, mlp_dim, heads, dim_head, dropout) for _ in range(num_layers)]
        self.net = nn.Sequential(*module_list)

    def forward(self, x):
        return self.net(x)


class Encoder(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, 
                heads, mlp_dim, pool = 'cls', channels = 3, 
                dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            # B*3*H*W to B*N*(P^2*3)
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        # Positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        # self.transformer = nn.Transformer(
        #     d_model=dim,
        #     nhead=heads,
        #     num_encoder_layers=depth,
        #     num_decoder_layers=0,
        #     dim_feedforward=mlp_dim,
        #     dropout=dropout,
        #     activation=nn.GELU()
        #     )
        self.transformer = Transformer(depth, dim, mlp_dim, heads, dim_head, dropout)
        
        self.pool = pool

    def forward(self, img):
        # No more cls token due to no need of cls
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        # Add positional encoding (1d encoding)
        x += self.pos_embedding[:, :n]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        return x