import torch
import torch.nn.functional as F


def vae_tabular_loss(x, x_recon, mu, logvar, ncont=30, beta=0.03):
    x_cont = x[:, :ncont]
    x_recon_cont = x_recon[:, :ncont]
    recon_loss_cont = F.mse_loss(x_recon_cont, x_cont)

    y_true = x[:, ncont:].unsqueeze(1)
    y_pred = x_recon[:, ncont:].unsqueeze(1)
    recon_loss_cat = F.binary_cross_entropy_with_logits(y_pred, y_true)

    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    total = recon_loss_cont + recon_loss_cat + beta * kl
    return total, recon_loss_cont, recon_loss_cat, kl
