import torch
import torch.nn as nn

from .common import EmbeddingLayer, RelPositionalEncoding, PreNet
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
        self.pre_net = PreNet(params.encoder.channels)
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
        duration
    ):
        x = self.emb(phoneme, a1, f2)
        x, pos_emb = self.relative_pos_emb(x)

        x_mask = sequence_mask(x_length).unsqueeze(1).to(x.dtype)
        z_mask = sequence_mask(y_length).unsqueeze(1).to(x.dtype)

        x = self.pre_net(x, x_mask)
        x = self.encoder(x, pos_emb, x_mask)
        x_mu = self.proj_mu(x)
        x_logs = torch.zeros_like(x_mu)

        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(z_mask, 2)
        path = generate_path(duration.squeeze(1), attn_mask.squeeze(1))

        z_mu, z_logs, dur_pred = self.variance_adopter(x, x_mu, x_logs, x_mask, path)

        z, log_df_dz, z_mask = self.decoder(y, z_mask)
        z *= z_mask

        z_mu = z_mu[:, :, :z.size(-1)]
        z_logs = z_logs[:, :, :z.size(-1)]
        return (z_mu, z_logs), (z, log_df_dz), dur_pred, (x_mask, z_mask)

    def infer(self, phoneme, a1, f2, x_length):
        x = self.emb(phoneme, a1, f2)
        x, pos_emb = self.relative_pos_emb(x)

        x_mask = sequence_mask(x_length).unsqueeze(1).to(x.dtype)
        x, pos_emb = self.relative_pos_emb(x)

        x = self.pre_net(x, x_mask)
        x = self.encoder(x, pos_emb, x_mask)
        x_mu = self.proj_mu(x)

        x_mu, y_mask = self.variance_adopter.infer(x, x_mu, x_mask)

        y, *_ = self.decoder.backward(x_mu, y_mask)
        return y
