import math
import torch
import torch.nn as nn

from .common import EmbeddingLayer, RelPositionalEncoding, PreNet
from .transformer import Transformer
from .predictors import VariancePredictor
from .flow import Glow
from monotonic_align import maximum_path
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
        self.duration_predictor = VariancePredictor(**params.variance_predictor)
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

        dur_pred = self.duration_predictor(x.detach(), x_mask)

        z, log_df_dz, z_mask = self.decoder(y, z_mask)
        z *= z_mask
        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(z_mask, 2)
        with torch.no_grad():
            x_logs = torch.zeros_like(x_mu)
            x_s_sq_r = torch.exp(-2 * x_logs)
            logp1 = torch.sum(-0.5 * math.log(2 * math.pi) - x_logs, [1]).unsqueeze(-1)  # [b, t, 1]
            logp2 = torch.matmul(x_s_sq_r.transpose(1, 2), -0.5 * (z ** 2))  # [b, t, d] x [b, d, t'] = [b, t, t']
            logp3 = torch.matmul(x_mu.transpose(1, 2), z)  # [b, t, d] x [b, d, t'] = [b, t, t']
            logp4 = torch.sum(-0.5 * (x_mu ** 2), [1]).unsqueeze(-1)  # [b, t, 1]
            logp = logp1 + logp2 + logp3 + logp4  # [b, t, t']
            path = maximum_path(logp, attn_mask.squeeze(1)).unsqueeze(1).detach()
        x_mu = torch.matmul(x_mu, path)
        duration = torch.log(1e-8 + torch.sum(path, -1)) * x_mask
        return x_mu, (z, log_df_dz), (dur_pred, duration), (x_mask, z_mask)

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
