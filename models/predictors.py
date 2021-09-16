import torch
import torch.nn as nn

from .common import LayerNorm
from .utils import sequence_mask, generate_path


class VarianceAdopter(nn.Module):
    def __init__(self, channels, dropout):
        super(VarianceAdopter, self).__init__()
        self.duration_predictor = VariancePredictor(
            channels=channels,
            n_layers=2,
            kernel_size=3,
            dropout=dropout
        )
        self.length_regulator = LengthRegulator()

    def forward(
        self,
        x,
        x_mu,
        x_mask,
        path
    ):
        dur_pred = self.duration_predictor(x, x_mask)
        x_mu = self.length_regulator(x_mu, path)
        return x_mu, dur_pred

    def infer(self, x, x_mu, x_mask):
        dur_pred = self.duration_predictor(x, x_mask)
        dur_pred = torch.exp(dur_pred)
        dur_pred = torch.round(dur_pred) * x_mask
        y_lengths = torch.clamp_min(torch.sum(dur_pred, [1, 2]), 1).long()
        y_mask = sequence_mask(y_lengths).unsqueeze(1).to(x_mask.device)
        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(y_mask, 2)

        path = generate_path(dur_pred.squeeze(1), attn_mask.squeeze(1))

        x_mu = self.length_regulator(x_mu, path)
        return x_mu, y_mask


class VariancePredictor(nn.Module):
    def __init__(self, channels, n_layers, kernel_size, dropout):
        super(VariancePredictor, self).__init__()

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2),
                nn.SiLU(),
                LayerNorm(channels),
                nn.Dropout(dropout)
            ) for _ in range(n_layers)
        ])

        self.out = nn.Conv1d(channels, 1, 1)

    def forward(self, x, x_mask):
        for layer in self.layers:
            x = layer(x)
            x *= x_mask
        x = self.out(x)
        x *= x_mask
        return x


class LengthRegulator(nn.Module):
    def forward(self, x, path):
        x = torch.matmul(x, path)
        return x
