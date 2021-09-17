import math
import torch

constant_term = 0.5 * math.log(2 * math.pi)

#
# def mle_loss(z, m, log_df_dz, length):
#     print(z.dtype, m.dtype, log_df_dz.dtype)
#     loss = 0.5 * torch.sum((z - m) ** 2)
#     loss -= torch.sum(log_df_dz)
#     loss /= torch.sum(length)
#     loss += constant_term
#     return loss

def mle_loss(z, m, logdet, mask):
    print(z.size(), m.size(), logdet.size(), mask.size())
    l = 0.5 * torch.sum((z - m) ** 2)  # neg normal likelihood w/o the constant term
    l = l - torch.sum(logdet)  # log jacobian determinant
    l = l / torch.sum(torch.ones_like(z) * mask)  # averaging across batch, channel and time axes
    l = l + 0.5 * math.log(2 * math.pi)  # add the remaining constant term
    return l
