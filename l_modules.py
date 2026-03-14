import torch
import lightning as l

from losses import vae_tabular_loss


class VAEModule(l.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.loss_history = []
        self.reconstruction_feats_history = []
        self.reconstruction_labels_history = []
        self.kl_div_loss_history = []
        self.lr_history = []

        self.loss_history_epoch = []
        self.reconstruction_feats_history_epoch = []
        self.reconstruction_labels_epoch = []
        self.kl_div_loss_history_epoch = []
        

    def training_step(self, batch):
        x, y = batch
        x = torch.cat([x, y], dim=1)
        x_recon, mu, logvar = self.model(x.float())
        loss, recon_cont, recon_cat, kl = vae_tabular_loss(x.float(), x_recon, mu, logvar)
        self.loss_history.append(loss.item())
        self.reconstruction_feats_history.append(recon_cont.item())
        self.reconstruction_labels_history.append(recon_cat.item())
        self.kl_div_loss_history.append(kl.item())
        return loss

    def on_train_epoch_end(self):
        self.loss_history_epoch.append(sum(self.loss_history) / len(self.loss_history))
        self.reconstruction_feats_history_epoch.append(sum(self.reconstruction_feats_history) / len(self.reconstruction_feats_history))
        self.reconstruction_labels_epoch.append(sum(self.reconstruction_labels_history) / len(self.reconstruction_labels_history))
        self.kl_div_loss_history_epoch.append(sum(self.kl_div_loss_history) / len(self.kl_div_loss_history))
        self.lr_history.append(self.optimizers().param_groups[0]["lr"])

        self.loss_history.clear()
        self.reconstruction_feats_history.clear()
        self.reconstruction_labels_history.clear()
        self.kl_div_loss_history.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=1e-4, total_steps=self.trainer.estimated_stepping_batches)


        return {
            "optimizer": optimizer,
            # "lr_scheduler": {
            #     "scheduler": scheduler,
            #     "interval": "step"
            # }
        }
