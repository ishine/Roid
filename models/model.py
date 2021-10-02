import math
import torch
import torch.nn as nn

from .common import EmbeddingLayer, RelPositionalEncoding, PreNet
from .transformer import Transformer
from .predictors import VarianceAdopter
from .glow import Glow
from monotonic_align import maximum_path
from .utils import sequence_mask


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
        y_length
    ):
        x = self.emb(phoneme, a1, f2)
        x, pos_emb = self.relative_pos_emb(x)

        x_mask = sequence_mask(x_length).unsqueeze(1).to(x.dtype)
        z_mask = sequence_mask(y_length).unsqueeze(1).to(x.dtype)

        x = self.pre_net(x, x_mask)
        x = self.encoder(x, pos_emb, x_mask)
        x_mu = self.proj_mu(x)
        x_logs = torch.zeros_like(x_mu)

        z, log_df_dz, z_mask = self.decoder(y, z_mask)

        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(z_mask, 2)
        with torch.no_grad():
            x_s_sq_r = torch.exp(-2 * x_logs)
            logp1 = torch.sum(-0.5 * math.log(2 * math.pi) - x_logs, [1]).unsqueeze(-1)
            logp2 = torch.matmul(x_s_sq_r.transpose(1, 2), -0.5 * (z ** 2))
            logp3 = torch.matmul((x_mu * x_s_sq_r).transpose(1, 2), z)
            logp4 = torch.sum(-0.5 * (x_mu ** 2) * x_s_sq_r, [1]).unsqueeze(-1)
            logp = logp1 + logp2 + logp3 + logp4
            path = maximum_path(logp, attn_mask.squeeze(1)).unsqueeze(1).detach()

        z_mu, z_logs, dur_pred = self.variance_adopter(x, x_mu, x_logs, x_length, x_mask, path.squeeze(1))

        z_mu = z_mu[:, :, :z.size(-1)]
        z_logs = z_logs[:, :, :z.size(-1)]
        duration = torch.sum(path, dim=-1)
        return (z_mu, z_logs), (z, log_df_dz), (dur_pred, duration), (x_mask, z_mask)

    def infer(self, phoneme, a1, f2, x_length, noise_scale=0):
        x_emb = self.emb(phoneme, a1, f2)

        x_mask = sequence_mask(x_length).unsqueeze(1).to(x_emb.dtype)
        x, pos_emb = self.relative_pos_emb(x_emb)

        x = self.pre_net(x, x_mask)
        x = self.encoder(x, pos_emb, x_mask)
        x_mu = self.proj_mu(x)
        x_logs = torch.zeros_like(x_mu)

        z_mu, z_logs, z_mask = self.variance_adopter.infer(x_emb, x_mu, x_logs, x_length, x_mask)

        z = (z_mu + torch.exp(z_logs) * torch.randn_like(z_mu) * noise_scale) * z_mask

        y, *_ = self.decoder.backward(z, z_mask)
        return y
