import torch.nn as nn

from .attention import RelativeSelfAttentionLayer
from .common import FFN


class TransformerLayer(nn.Module):
    def __init__(self,
                 channels,
                 n_heads,
                 dropout):
        super(TransformerLayer, self).__init__()
        self.mha = RelativeSelfAttentionLayer(channels, n_heads, dropout)
        self.ff = FFN(channels, dropout)

    def forward(self, x, pos_emb, x_mask):
        x += self.mha(x, pos_emb, x_mask)
        x += self.ff(x, x_mask)
        x *= x_mask
        return x


class Transformer(nn.Module):
    def __init__(self,
                 channels=192,
                 n_layers=6,
                 n_heads=2,
                 dropout=0.1):
        super(Transformer, self).__init__()

        self.layers = nn.ModuleList([
            TransformerLayer(
                channels=channels,
                n_heads=n_heads,
                dropout=dropout
            ) for _ in range(n_layers)
        ])

    def forward(self, x, pos_emb, x_mask):
        for layer in self.layers:
            x = layer(x, pos_emb, x_mask)
        return x
