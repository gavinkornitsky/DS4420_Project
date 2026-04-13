import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as l
import numpy as np

class VAEModule(l.LightningModule):
    def __init__(
            self, 
            input_dim=31, 
            latent_dim=16,
            lr = 1e-4,
            weight_decay = 1e-5,
            label_weight = 10.0,
            beta_warmup_epochs = 15,
            beta_anneal_epochs = 60,
            beta_max = 1.0,
            feature_mean = None,
            feature_std = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["feature_mean", "feature_std"])

        self.register_buffer("feature_mean", torch.tensor(feature_mean) if feature_mean is not None else torch.zeros(input_dim - 1))
        self.register_buffer("feature_std", torch.tensor(feature_std) if feature_std is not None else torch.ones(input_dim - 1))

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.mu_layer = nn.Linear(64, latent_dim)
        self.logvar_layer = nn.Linear(64, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )
    
    def encode(self, x):
        h = self.encoder(x)
        return self.mu_layer(h), self.logvar_layer(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    def _get_beta(self):
        epoch = self.current_epoch
        warmup = self.hparams.beta_warmup_epochs
        anneal = self.hparams.beta_anneal_epochs

        if epoch < warmup:
            return 0.0
        
        progress = min((epoch - warmup) / max(1, anneal), 1.0)

        return (1 - math.cos(progress * math.pi)) * self.hparams.beta_max * 0.5
    
    def vae_tabular_loss(self, x, x_recon, mu, logvar, ncont=30):
        beta = self._get_beta()
        self.log("beta", beta, prog_bar=True)
        x_cont = x[:, :ncont]
        x_recon_cont = x_recon[:, :ncont]
        recon_loss_cont = F.mse_loss(x_recon_cont, x_cont)

        y_true = x[:, ncont:].unsqueeze(1)
        y_pred = x_recon[:, ncont:].unsqueeze(1)
        recon_loss_cat = F.binary_cross_entropy_with_logits(y_pred, y_true)

        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        total = recon_loss_cont + recon_loss_cat + beta * kl
        return total, recon_loss_cont, recon_loss_cat, kl
        

    def _compute_loss(self, x, x_recon, mu, logvar, mode='train'):
        loss, recon_cont, recon_cat, kl = self.vae_tabular_loss(x.float(), x_recon, mu, logvar)
        self.log(f"{mode}_loss", loss, prog_bar=True)
        self.log(f"{mode}_recon_cont", recon_cont)
        self.log(f"{mode}_recon_cat", recon_cat)
        self.log(f"{mode}_kl_div", kl)
        return loss
    
    def training_step(self, batch):
        x, y = batch
        y = y.unsqueeze(1)
        x = torch.cat([x, y], dim=1)
        x_recon, mu, logvar = self(x.float())
        loss = self._compute_loss(x, x_recon, mu, logvar, mode='train')
        return loss

    def validation_step(self, batch):
        x, y = batch
        y = y.unsqueeze(1)
        x = torch.cat([x, y], dim=1)
        x_recon, mu, logvar = self(x.float())
        loss = self._compute_loss(x, x_recon, mu, logvar, mode='val')
        return loss
    
    @torch.no_grad()
    def generate(self, n_samples=10):
        z = torch.randn(n_samples, self.hparams.latent_dim).to(self.device)
        samples = self.decode(z)
        samples_cont = samples[:, :self.hparams.input_dim - 1]
        samples_cont = samples_cont * self.feature_std + self.feature_mean
        samples_cat = F.sigmoid(samples[:, self.hparams.input_dim - 1:])
        samples = torch.cat([samples_cont, samples_cat], dim=1)
        return samples

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        return optimizer

if __name__ == "__main__": 
    dummy_input = torch.randn(4, 31)
    model = VAEModule(input_dim=31, latent_dim=16)
    x_recon, mu, logvar = model(dummy_input)
    print("Output shape:", x_recon.shape)
    print("Mu shape:", mu.shape)
    print("Logvar shape:", logvar.shape)    