import math
import torch

constant_term = 0.5 * math.log(2 * math.pi)


def mle_loss(z, m, log_df_dz, length):
    loss = 0.5 * torch.sum((z - m) ** 2)
    print(loss)
    loss -= torch.sum(log_df_dz)
    print(loss)
    loss /= torch.sum(length)
    print(loss)
    loss += constant_term
    return loss
