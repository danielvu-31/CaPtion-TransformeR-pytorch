import torch
import torch.nn as nn


class FeedForwardNetwork(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )
    
    def forward(self, x):
        return self.ffn(x)