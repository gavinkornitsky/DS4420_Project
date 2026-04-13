import torch
import lightning as l

from losses import vae_tabular_loss


class VAEModule(l.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        

    def _compute_loss(self, x, x_recon, mu, logvar, mode='train'):
        loss, recon_cont, recon_cat, kl = vae_tabular_loss(x.float(), x_recon, mu, logvar)
        self.log(f"{mode}_loss", loss, prog_bar=True)
        self.log(f"{mode}_recon_cont", recon_cont)
        self.log(f"{mode}_recon_cat", recon_cat)
        self.log(f"{mode}_kl_div", kl)
        return loss
    
    def training_step(self, batch):
        x, y = batch
        x = torch.cat([x, y], dim=1)
        x_recon, mu, logvar = self.model(x.float())
        loss = self._compute_loss(x, x_recon, mu, logvar, mode='train')
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        return optimizer
