import math
import torch


def mle_loss(z, m, logdet, length):
    l = 0.5 * torch.sum((z - m) ** 2)  # neg normal likelihood w/o the constant term
    l = l - torch.sum(logdet)  # log jacobian determinant
    l = l / torch.sum(length)  # averaging across batch, channel and time axes
    l = l + 0.5 * math.log(2 * math.pi)  # add the remaining constant term
    return l
