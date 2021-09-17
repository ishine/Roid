import math
import torch

constant_term = 0.5 * math.log(2 * math.pi)


def mle_loss(z, m, log_df_dz, length):
    loss = 0.5 * torch.sum((z.to(m.dtype) - m) ** 2)
    loss -= torch.sum(log_df_dz).to(m.dtype)
    loss /= torch.sum(length)
    loss += constant_term
    return loss
