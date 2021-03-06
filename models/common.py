import math
import torch
import torch.nn as nn


class EmbeddingLayer(nn.Module):
    def __init__(self, n_phoneme, n_accent, channels):
        super(EmbeddingLayer, self).__init__()
        self.scale = math.sqrt(channels)

        self.phoneme_emb = nn.Embedding(n_phoneme, channels)
        self.f2_emb = nn.Embedding(n_accent, channels)
        nn.init.normal_(self.phoneme_emb.weight, 0.0, channels ** -0.5)
        nn.init.normal_(self.f2_emb.weight, 0.0, channels ** -0.5)

    def forward(self, phoneme, a1, f2):
        phoneme_emb = self.phoneme_emb(phoneme) * self.scale
        f2_emb = self.f2_emb(f2) * self.scale
        a1_emb = a1.unsqueeze(-1).expand(-1, -1, phoneme_emb.size(-1))
        x = torch.cat([phoneme_emb, f2_emb, a1_emb], dim=-1).transpose(-1, -2)
        return x


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dropout=0.5):
        super(ConvLayer, self).__init__()
        self.norm = LayerNorm(out_channels)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_mask):
        x = self.norm(x)
        x = self.conv(x * x_mask)
        x = self.act(x)
        x = self.dropout(x)
        return x


class PreNet(nn.Module):
    def __init__(self, channels, n_layers=3, kernel_size=5, dropout=0.5):
        super(PreNet, self).__init__()

        self.layers = nn.ModuleList([
            ConvLayer(
                channels,
                channels,
                kernel_size,
                dropout
            ) for _ in range(n_layers)
        ])
        self.out = nn.Conv1d(channels, channels, 1)
        self.out.weight.data.zero_()
        self.out.bias.data.zero_()

    def forward(self, x, x_mask):
        residual = x
        for layer in self.layers:
            x = layer(x, x_mask)
        x = residual + self.out(x)
        x *= x_mask
        return x


def PostNet(params):
    return nn.Sequential(
        nn.Conv1d(80, params.decoder.channels, 5, padding=2),
        nn.BatchNorm1d(params.decoder.channels),
        nn.Tanh(),
        nn.Dropout(0.5),
        nn.Conv1d(params.decoder.channels, params.decoder.channels, 5, padding=2),
        nn.BatchNorm1d(params.decoder.channels),
        nn.Tanh(),
        nn.Dropout(0.5),
        nn.Conv1d(params.decoder.channels, params.decoder.channels, 5, padding=2),
        nn.BatchNorm1d(params.decoder.channels),
        nn.Tanh(),
        nn.Dropout(0.5),
        nn.Conv1d(params.decoder.channels, params.decoder.channels, 5, padding=2),
        nn.BatchNorm1d(params.decoder.channels),
        nn.Tanh(),
        nn.Dropout(0.5),
        nn.Conv1d(params.decoder.channels, 80, 5, padding=2)
    )


class WaveNet(nn.Module):
    def __init__(self, channels, kernel_size, num_layers, dilation_rate=1, gin_channels=0, dropout=0):
        super(WaveNet, self).__init__()

        self.channels = channels
        self.num_layers = num_layers

        self.dilated_convs = nn.ModuleList()
        for i in range(num_layers):
            dilation = dilation_rate ** i
            padding = int((kernel_size * dilation - dilation) / 2)
            conv = nn.Conv1d(channels, channels * 2, kernel_size, padding=padding, dilation=dilation)
            conv = nn.utils.weight_norm(conv)
            self.dilated_convs.append(conv)

        self.out_convs = nn.ModuleList()
        for i in range(num_layers):
            conv = nn.Conv1d(channels, channels * 2 if i < num_layers-1 else channels, 1)
            conv = nn.utils.weight_norm(conv)
            self.out_convs.append(conv)

        self.dropout = nn.Dropout(dropout)

        if gin_channels > 0:
            self.cond_layer = nn.Conv1d(gin_channels, channels, 1)

    def forward(self, x, x_mask, g=None):
        if g is not None:
            g = self.cond_layer(g)
        out = 0
        for i, (d_conv, o_conv) in enumerate(zip(self.dilated_convs, self.out_convs)):
            x_in = d_conv(x)
            if g is not None:
                x_in += g
            x_in_a, x_in_b = x_in.chunk(2, dim=1)
            x_in = x_in_a.sigmoid() * x_in_b.tanh()
            if i < self.num_layers - 1:
                o1, o2 = o_conv(x_in).chunk(2, dim=1)
                x = (x + o1) * x_mask
                x = self.dropout(x)
                out += o2 * x_mask
            else:
                out += o_conv(x_in)
        return out * x_mask

    def remove_weight_norm(self):
        for l in self.dilated_convs:
            nn.utils.remove_weight_norm(l)
        for l in self.out_convs:
            nn.utils.remove_weight_norm(l)


class FFN(nn.Module):
    def __init__(self, channels, kernel_size=3, dropout=0.1):
        super(FFN, self).__init__()

        self.norm = LayerNorm(channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2)
        self.act = nn.SiLU()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_mask):
        x = self.norm(x)
        x = self.conv1(x * x_mask)
        x = self.act(x)
        x = self.dropout(x)
        x = self.conv2(x * x_mask)
        x = self.dropout(x)
        return x * x_mask


class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        mean = torch.mean(x, 1, keepdim=True)
        variance = torch.mean((x - mean) ** 2, 1, keepdim=True)

        x = (x - mean) * torch.rsqrt(variance + self.eps)

        x = x * self.gamma.view(1, -1, 1) + self.beta.view(1, -1, 1)
        return x


class RelPositionalEncoding(nn.Module):
    def __init__(self, channels, dropout=0.1, max_len=10000):
        super(RelPositionalEncoding, self).__init__()
        self.d_model = channels
        self.scale = math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x):
        if self.pe is not None:
            if self.pe.size(2) >= x.size(2) * 2 - 1:
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        pe_positive = torch.zeros(x.size(1), self.d_model)
        pe_negative = torch.zeros(x.size(1), self.d_model)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)
        self.pe = pe.transpose(-1, -2).to(device=x.device, dtype=x.dtype)

    def forward(self, x):
        self.extend_pe(x)
        pos_emb = self.pe[
            :,
            :,
            self.pe.size(2) // 2 - x.size(2) + 1 : self.pe.size(2) // 2 + x.size(2),
        ]
        return x, self.dropout(pos_emb)
