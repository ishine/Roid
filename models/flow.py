import numpy as np
import torch
import torch.nn as nn

from .common import WaveNet


def squeeze(z, dim=1):
    C = z.size(dim)
    z0, z1 = torch.split(z, C // 2, dim=dim)
    return z0, z1


def unsqueeze(z0, z1, dim=1):
    z = torch.cat([z0, z1], dim=dim)
    return z


class Glow(nn.Module):
    def __init__(self, in_channels, channels, kernel_size, num_flows, num_layers, gin_channels=0, dropout=0.05):
        super(Glow, self).__init__()
        self.flows = nn.ModuleList()
        for _ in range(num_flows):
            self.flows.append(ActNorm(in_channels))
            self.flows.append(InvertibleConv1x1(in_channels))
            self.flows.append(AffineCoupling(in_channels, channels, kernel_size, num_layers, gin_channels, dropout))

    def forward(self, z, z_mask, g=None):
        log_df_dz = 0
        for flow in self.flows:
            z, log_df_dz = flow(z=z, z_mask=z_mask, log_df_dz=log_df_dz, g=g)
        return z, log_df_dz

    def backward(self, y, y_mask, g=None):
        log_df_dz = 0
        for flow in self.flows:
            y, log_df_dz = flow.backward(y=y, y_mask=y_mask, log_dz_df=log_df_dz, g=g)
        return y, log_df_dz


class AffineCoupling(nn.Module):

    def __init__(self, in_channels, channels, kernel_size, num_layers, gin_channels=0, dropout=0.05):
        super(AffineCoupling, self).__init__()

        self.split_channels = in_channels // 2

        self.s_log_scale = nn.Parameter(torch.randn(1) * 0.01)
        self.s_bias = nn.Parameter(torch.randn(1) * 0.01)

        self.start = torch.nn.utils.weight_norm(nn.Conv1d(in_channels // 2, channels, 1))
        self.net = WaveNet(channels, kernel_size, num_layers, gin_channels=gin_channels, dropout=dropout)
        self.end = nn.Conv1d(channels, in_channels, 1)
        self.end.weight.data.zero_()
        self.end.bias.data.zero_()

    def forward(self, z, z_mask, log_df_dz, g=None):
        z0, z1 = squeeze(z)
        z0, z1, log_df_dz = self._transform(z0, z1, z_mask, log_df_dz, g=g)
        z = unsqueeze(z0, z1)
        return z, log_df_dz

    def backward(self, y, y_mask, log_df_dz, g=None):
        y0, y1 = squeeze(y)
        y0, y1, log_df_dz = self._inverse_transform(y0, y1, y_mask, log_df_dz, g=g)
        y = unsqueeze(y0, y1)
        return y, log_df_dz

    def _transform(self, z0, z1, z_mask, log_df_dz, g):
        params = self.start(z1) * z_mask
        params = self.net(params, z_mask, g=g)
        params = self.end(params)
        t = params[:, :self.split_channels, :]
        s = torch.tanh(params[:, self.split_channels:]) * self.s_log_scale + self.s_bias

        z0 = z0 * torch.exp(s) + t
        log_df_dz += torch.sum(s.view(z0.size(0), -1), dim=1)

        return z0, z1, log_df_dz

    def _inverse_transform(self, y0, y1, y_mask, log_df_dz, g):
        params = self.start(y1) * y_mask
        params = self.net(params, y_mask, g=g)
        params = self.end(params)
        t = params[:, :self.split_channels]
        s = torch.tanh(params[:, self.split_channels:]) * self.s_log_scale + self.s_bias

        y0 = torch.exp(-s) * (y0 - t)
        log_df_dz -= torch.sum(s.view(y0.size(0), -1), dim=1)

        return y0, y1, log_df_dz


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

        length = torch.sum(z_mask)
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
