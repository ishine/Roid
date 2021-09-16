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
            self.flows.append(InvertibleConv1x1(in_channels * n_sqz))
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
        s = params[:, self.split_channels:, :]

        z0 = z0 * torch.exp(s) + t
        log_df_dz += torch.sum(s, dim=[1, 2])

        return z0, z1, log_df_dz

    def _inverse_transform(self, y0, y1, y_mask, log_df_dz, g):
        params = self.start(y1) * y_mask
        params = self.net(params, y_mask, g=g)
        params = self.end(params)
        t = params[:, :self.split_channels, :]
        s = params[:, self.split_channels:, :]

        y0 = torch.exp(-s) * (y0 - t)
        log_df_dz -= torch.sum(s, dim=[1, 2])

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
            log_std = torch.log(torch.std(z, dim=[0, 2]) + self.eps)
            mean = torch.mean(z, dim=[0, 2])
            self.log_scale.data.copy_(log_std.view(self.dimensions))
            self.bias.data.copy_(mean.view(self.dimensions))
            self.initialized = True

        z = (z - self.bias) / torch.exp(self.log_scale)

        length = torch.sum(z_mask, dim=[1, 2])
        log_df_dz += torch.sum(self.log_scale) * length
        return z, log_df_dz

    def backward(self, y, y_mask, log_df_dz, **kwargs):
        y = y * torch.exp(self.log_scale) + self.bias
        length = torch.sum(y_mask, dim=[1, 2])
        log_df_dz -= torch.sum(self.log_scale) * length
        return y, log_df_dz


class InvertibleConv1x1(nn.Module):
    def __init__(self, channels):
        super(InvertibleConv1x1, self).__init__()

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


class InvConvNear(nn.Module):
    def __init__(self, channels, n_split=4):
        super().__init__()
        assert (n_split % 2 == 0)
        self.channels = channels
        self.n_split = n_split

        w_init = torch.linalg.qr(torch.FloatTensor(self.n_split, self.n_split).normal_())[0]
        if torch.det(w_init) < 0:
            w_init[:, 0] = -1 * w_init[:, 0]
        self.weight = nn.Parameter(w_init)

    def forward(self, z, z_mask, log_df_dz, **kwargs):
        return self.process(z, z_mask, log_df_dz, forward=True)

    def backward(self, y, y_mask, log_df_dz, **kwargs):
        return self.process(y, y_mask, log_df_dz, forward=False)

    def process(self, x, x_mask, log_df_dz, forward=True):
        B, C, T = x.size()
        x = x.view(B, 2, C // self.n_split, self.n_split // 2, T)
        x = x.permute(0, 1, 3, 2, 4).contiguous().view(B, self.n_split, C // self.n_split, T)

        length = torch.sum(x_mask, [1, 2])
        if forward:
            weight = self.weight
            log_df_dz += torch.logdet(weight) * (C / self.n_split) * length
        else:
            weight = torch.inverse(self.weight.float()).to(dtype=self.weight.dtype)
            log_df_dz += torch.logdet(weight) * (C / self.n_split) * length

        weight = weight.view(self.n_split, self.n_split, 1, 1)
        z = F.conv2d(x, weight)

        z = z.view(B, 2, self.n_split // 2, C // self.n_split, T)
        z = z.permute(0, 1, 3, 2, 4).contiguous().view(B, C, T) * x_mask
        return z, log_df_dz


