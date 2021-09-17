import math
import torch


def mle_loss(z, m, logs, log_df_dz, mask):
    loss = torch.sum(logs) + 0.5 * torch.sum(torch.exp(-2 * logs) * ((z - m) ** 2))
    loss = loss - torch.sum(log_df_dz)
    loss = loss / torch.sum(torch.ones_like(z) * mask)
    loss = loss + 0.5 * math.log(2 * math.pi)
    return loss
