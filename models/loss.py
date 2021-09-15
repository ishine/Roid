import math
import torch

constant_term = 0.5 * math.log(2 * math.pi)


def mle_loss(z, m, log_df_dz, mask):
    loss = 0.5 * torch.sum((z - m) ** 2)
    loss -= torch.sum(log_df_dz)
    loss /= torch.sum(mask)
    loss += constant_term
    return loss
