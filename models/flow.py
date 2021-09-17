import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import WaveNet


class Glow(nn.Module):
    def __init__(self, in_channels, channels, kernel_size, num_flows, num_layers, n_sqz=2, gin_channels=0, dropout=0.05):
        super(Glow, self).__init__()

        self.n_sqz = n_sqz

        self.flows = nn.ModuleList()
        for _ in range(num_flows):
            self.flows.append(ActNorm(in_channels * n_sqz))
            self.flows.append(Invertible1x1Conv(in_channels * n_sqz))
            self.flows.append(AffineCoupling(in_channels * n_sqz, channels, kernel_size, num_layers, gin_channels, dropout))

    def forward(self, z, z_mask, g=None):
        if self.n_sqz > 1:
            z, z_mask = self.squeeze(z, z_mask, self.n_sqz)
        log_df_dz = 0
        for flow in self.flows:
            z, log_df_dz = flow(z=z, z_mask=z_mask, log_df_dz=log_df_dz, g=g)
        if self.n_sqz > 1:
            z, z_mask = self.unsqueeze(z, z_mask, self.n_sqz)
        return z, log_df_dz, z_mask

    def backward(self, y, y_mask, g=None):
        if self.n_sqz > 1:
            y, y_mask = self.squeeze(y, y_mask, self.n_sqz)
        log_df_dz = 0
        for flow in reversed(self.flows):
            y, log_df_dz = flow.backward(y=y, y_mask=y_mask, log_df_dz=log_df_dz, g=g)
        if self.n_sqz > 1:
            y, y_mask = self.unsqueeze(y, y_mask, self.n_sqz)
        return y, log_df_dz, y_mask

    @staticmethod
    def squeeze(x, x_mask=None, n_sqz=2):
        b, c, t = x.size()

        t = (t // n_sqz) * n_sqz
        x = x[:, :, :t]
        x_sqz = x.view(b, c, t // n_sqz, n_sqz)
        x_sqz = x_sqz.permute(0, 3, 1, 2).contiguous().view(b, c * n_sqz, t // n_sqz)

        if x_mask is not None:
            x_mask = x_mask[:, :, n_sqz - 1::n_sqz]
        else:
            x_mask = torch.ones(b, 1, t // n_sqz).to(device=x.device, dtype=x.dtype)
        return x_sqz * x_mask, x_mask

    @staticmethod
    def unsqueeze(x, x_mask=None, n_sqz=2):
        b, c, t = x.size()

        x_unsqz = x.view(b, n_sqz, c // n_sqz, t)
        x_unsqz = x_unsqz.permute(0, 2, 3, 1).contiguous().view(b, c // n_sqz, t * n_sqz)

        if x_mask is not None:
            x_mask = x_mask.unsqueeze(-1).repeat(1, 1, 1, n_sqz).view(b, 1, t * n_sqz)
        else:
            x_mask = torch.ones(b, 1, t * n_sqz).to(device=x.device, dtype=x.dtype)
        return x_unsqz * x_mask, x_mask


class ActNorm(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super(ActNorm, self).__init__()
        self.channels = channels
        self.eps = eps

        self.dimensions = [1, channels, 1]
        self.register_parameter('log_scale', nn.Parameter(torch.zeros(self.dimensions)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(self.dimensions)))
        self.initialized = False

    def forward(self, z, z_mask, log_df_dz, **kwargs):
        if not self.initialized:
            self.initialize(z, z_mask)
            self.initialized = True

        z = z * torch.exp(self.log_scale) + self.bias

        length = torch.sum(z_mask, dim=[1, 2])
        log_df_dz += torch.sum(self.log_scale) * length
        return z, log_df_dz

    def backward(self, y, y_mask, log_df_dz, **kwargs):
        y = (y - self.bias) * torch.exp(-self.log_scale)
        length = torch.sum(y_mask, dim=[1, 2])
        log_df_dz -= torch.sum(self.log_scale) * length
        return y, log_df_dz

    @torch.no_grad()
    def initialize(self, x, x_mask):
        denom = torch.sum(x_mask, [0, 2])
        m = torch.sum(x * x_mask, [0, 2]) / denom
        m_sq = torch.sum(x * x * x_mask, [0, 2]) / denom
        v = m_sq - (m ** 2)
        logs = 0.5 * torch.log(torch.clamp_min(v, 1e-6))

        bias_init = (-m * torch.exp(-logs)).view(self.dimensions)
        logs_init = (-logs).view(self.dimensions)

        self.bias.data.copy_(bias_init)
        self.log_scale.data.copy_(logs_init)


class Invertible1x1Conv(nn.Module):
    def __init__(self, channels):
        super(Invertible1x1Conv, self).__init__()
        self.channels = channels

        w_init = torch.linalg.qr(torch.FloatTensor(self.channels, self.channels).normal_())[0]
        self.weight = nn.Parameter(w_init)

    def forward(self, z, z_mask, log_df_dz, **kwargs):
        weight = self.weight
        z = F.conv1d(z, weight.unsqueeze(-1))

        length = torch.sum(z_mask, dim=[1, 2])
        log_df_dz += torch.slogdet(weight)[1] * length
        return z, log_df_dz

    def backward(self, y, y_mask, log_df_dz, **kwargs):
        weight = self.weight.inverse()
        y = F.conv1d(y, weight.unsqueeze(-1))

        length = torch.sum(y_mask, dim=[1, 2])
        log_df_dz -= torch.slogdet(weight)[1] * length
        return y, log_df_dz


class Invertible1x1ConvLU(nn.Module):
    def __init__(self, channels):
        super(Invertible1x1ConvLU, self).__init__()

        W = torch.zeros((channels, channels), dtype=torch.float32)
        nn.init.orthogonal_(W)
        LU, pivots = torch.lu(W)

        P, L, U = torch.lu_unpack(LU, pivots)
        self.P = nn.Parameter(P, requires_grad=False)
        self.L = nn.Parameter(L, requires_grad=True)
        self.U = nn.Parameter(U, requires_grad=True)
        self.I = nn.Parameter(torch.eye(channels), requires_grad=False)
        self.pivots = nn.Parameter(pivots, requires_grad=False)

        L_mask = np.tril(np.ones((channels, channels), dtype='float32'), k=-1)
        U_mask = L_mask.T.copy()
        self.L_mask = nn.Parameter(torch.from_numpy(L_mask), requires_grad=False)
        self.U_mask = nn.Parameter(torch.from_numpy(U_mask), requires_grad=False)

        s = torch.diag(U)
        sign_s = torch.sign(s)
        log_s = torch.log(torch.abs(s))
        self.log_s = nn.Parameter(log_s, requires_grad=True)
        self.sign_s = nn.Parameter(sign_s, requires_grad=False)

    def forward(self, z, z_mask, log_df_dz, **kwargs):
        L = self.L * self.L_mask + self.I
        U = self.U * self.U_mask + torch.diag(self.sign_s * torch.exp(self.log_s))
        W = self.P @ L @ U
        z = torch.matmul(W, z)

        length = torch.sum(z_mask, dim=[1, 2])
        log_df_dz += torch.sum(self.log_s, dim=0) * length

        return z, log_df_dz

    def backward(self, y, y_mask, log_df_dz, **kwargs):
        with torch.no_grad():
            LU = self.L * self.L_mask + self.U * self.U_mask + torch.diag(self.sign_s * torch.exp(self.log_s))

            y_reshape = y.view(y.size(0), y.size(1), -1)
            y_reshape = torch.lu_solve(y_reshape, LU.unsqueeze(0), self.pivots.unsqueeze(0))
            y = y_reshape.view(y.size())
            y = y.contiguous()

        length = torch.sum(y_mask, dim=[1, 2])
        log_df_dz -= torch.sum(self.log_s, dim=0) * length

        return y, log_df_dz


class AffineCoupling(nn.Module):

    def __init__(self, in_channels, channels, kernel_size, num_layers, gin_channels=0, dropout=0.05):
        super(AffineCoupling, self).__init__()

        self.split_channels = in_channels // 2

        self.start = torch.nn.utils.weight_norm(nn.Conv1d(in_channels // 2, channels, 1))
        self.net = WaveNet(channels, kernel_size, num_layers, gin_channels=gin_channels, dropout=dropout)
        self.end = nn.Conv1d(channels, in_channels, 1)
        self.end.weight.data.zero_()
        self.end.bias.data.zero_()

    def forward(self, z, z_mask, log_df_dz, g=None):
        z0, z1 = self.squeeze(z)
        z0, z1, log_df_dz = self._transform(z0, z1, z_mask, log_df_dz, g=g)
        z = self.unsqueeze(z0, z1)
        return z, log_df_dz

    def backward(self, y, y_mask, log_df_dz, g=None):
        y0, y1 = self.squeeze(y)
        y0, y1, log_df_dz = self._inverse_transform(y0, y1, y_mask, log_df_dz, g=g)
        y = self.unsqueeze(y0, y1)
        return y, log_df_dz

    def _transform(self, z0, z1, z_mask, log_df_dz, g):
        params = self.start(z1) * z_mask
        params = self.net(params, z_mask, g=g)
        params = self.end(params)
        t = params[:, :self.split_channels, :]
        logs = params[:, self.split_channels:, :]

        z0 = z0 * torch.exp(logs) + t
        log_df_dz += torch.sum(logs, dim=[1, 2])

        return z0, z1, log_df_dz

    def _inverse_transform(self, y0, y1, y_mask, log_df_dz, g):
        params = self.start(y1) * y_mask
        params = self.net(params, y_mask, g=g)
        params = self.end(params)
        t = params[:, :self.split_channels, :]
        logs = params[:, self.split_channels:, :]

        y0 = (y0 - t) * torch.exp(-logs)
        log_df_dz -= torch.sum(logs, dim=[1, 2])

        return y0, y1, log_df_dz

    @staticmethod
    def squeeze(z, dim=1):
        C = z.size(dim)
        z0, z1 = torch.split(z, C // 2, dim=dim)
        return z0, z1

    @staticmethod
    def unsqueeze(z0, z1, dim=1):
        z = torch.cat([z0, z1], dim=dim)
        return z
