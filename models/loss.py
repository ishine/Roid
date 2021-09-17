import math
import torch

constant_term = 0.5 * math.log(2 * math.pi)


def mle_loss(z, m, log_df_dz, length):
    print(z.dtype, m.dtype, log_df_dz.dtype)
    loss = 0.5 * torch.sum((z - m) ** 2)
    loss -= torch.sum(log_df_dz)
    loss /= torch.sum(length)
    loss += constant_term
    return loss
