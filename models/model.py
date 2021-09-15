import torch
import torch.nn as nn

from .common import EmbeddingLayer, RelPositionalEncoding
from .transformer import Transformer
from .predictors import VarianceAdopter
from .flow import Glow
from .utils import sequence_mask, generate_path


class TTSModel(nn.Module):
    def __init__(self, params):
        super(TTSModel, self).__init__()

        self.emb = EmbeddingLayer(**params.embedding, channels=params.encoder.channels // 3)
        self.relative_pos_emb = RelPositionalEncoding(
            params.encoder.channels,
            params.encoder.dropout
        )
        self.encoder = Transformer(**params.encoder)
        self.proj_mu = nn.Conv1d(params.encoder.channels, params.n_mel, 1)
        self.variance_adopter = VarianceAdopter(**params.variance_adopter)
        self.decoder = Glow(in_channels=params.n_mel, **params.decoder)

    def forward(
        self,
        phoneme,
        a1,
        f2,
        x_length,
        y,
        y_length,
        duration,
        pitch,
        energy
    ):
        x = self.emb(phoneme, a1, f2)
        x, pos_emb = self.relative_pos_emb(x)

        x_mask = sequence_mask(x_length).unsqueeze(1).to(x.dtype)
        z_mask = sequence_mask(y_length).unsqueeze(1).to(x.dtype)

        x = self.encoder(x, pos_emb, x_mask)
        x_mu = self.proj_mu(x)

        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(z_mask, 2)
        path = generate_path(duration.squeeze(1), attn_mask.squeeze(1))

        x_mu, (dur_pred, pitch_pred, energy_pred) = self.variance_adopter(
            x,
            x_mu,
            x_mask,
            z_mask,
            pitch,
            energy,
            path
        )
        z, log_df_dz = self.decoder(y, z_mask)
        z *= z_mask

        return x, (z, log_df_dz), (dur_pred, pitch_pred, energy_pred), (x_mask, z_mask)

    def infer(self, phoneme, a1, f2, x_length):
        x = self.emb(phoneme, a1, f2)
        x, pos_emb = self.relative_pos_emb(x)

        x_mask = sequence_mask(x_length).unsqueeze(1).to(x.dtype)
        x, pos_emb = self.relative_pos_emb(x)
        x = self.encoder(x, pos_emb, x_mask)
        x_mu = self.proj_mu(x)

        x, y_mask = self.variance_adopter.infer(x, x_mu, x_mask)
        x, _ = self.decoder.backward(x, y_mask)
        return x
