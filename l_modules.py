import torch
import lightning as l

from losses import vae_tabular_loss


class VAEModule(l.LightningModule):
    def __init__(self, model, optimizer):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.loss_history = []
        self.loss_per_epoch = []

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = torch.cat([x, y], dim=1)
        x_recon, mu, logvar = self.model(x.float())
        loss, recon_cont, recon_cat, kl = vae_tabular_loss(x.float(), x_recon, mu, logvar)
        global_reconstruction_loss = recon_cont + recon_cat
        self.log('VAELoss', loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log('ReconstructionLoss', global_reconstruction_loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log('ReconstructionLoss_Feats', recon_cont, on_epoch=True, on_step=False, prog_bar=True)
        self.log('ReconstructionLoss_Targets', recon_cat, on_epoch=True, on_step=False, prog_bar=True)
        self.log('KullbackLeiblerDiv', kl, on_epoch=True, on_step=False, prog_bar=True)
        self.loss_history.append(loss.item())
        return loss

    def on_train_epoch_end(self):
        epoch_loss = sum(self.loss_history) / len(self.loss_history)
        self.loss_per_epoch.append(epoch_loss)
        self.loss_history = []

    def configure_optimizers(self):
        return self.optimizer
