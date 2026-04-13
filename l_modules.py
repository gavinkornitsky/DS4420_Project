import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as l
import numpy as np

class VAEModule(l.LightningModule):
    def __init__(
            self, 
            feature_dim = 30, 
            label_dim =1,
            latent_dim=16,
            hidden_dims=[256, 128, 64],
            dropout = 0.1,
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

        self.register_buffer("feature_mean", torch.tensor(feature_mean) if feature_mean is not None else torch.zeros(feature_dim))
        self.register_buffer("feature_std", torch.tensor(feature_std) if feature_std is not None else torch.ones(feature_dim))

        enc_in = feature_dim + label_dim
        enc_layers = []
        for h in hidden_dims:
            enc_layers += [nn.Linear(enc_in, h), nn.ReLU(), nn.Dropout(dropout)]
            enc_in = h
        self.encoder = nn.Sequential(*enc_layers)
        self.mu_layer = nn.Linear(hidden_dims[-1], latent_dim)
        self.logvar_layer = nn.Linear(hidden_dims[-1], latent_dim)

        dec_in = latent_dim + label_dim
        dec_layers = []
        for h in reversed(hidden_dims):
            dec_layers += [nn.Linear(dec_in, h), nn.ReLU(), nn.Dropout(dropout)]
            dec_in = h
        dec_layers += [nn.Linear(dec_in, feature_dim)]
        self.decoder = nn.Sequential(*dec_layers)
    
    def encode(self, x, y):
        x = torch.cat([x, y], dim=1)
        h = self.encoder(x)
        return self.mu_layer(h), self.logvar_layer(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y):
        z = torch.cat([z, y], dim=1)
        return self.decoder(z)

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, y), mu, logvar
    
    def _get_beta(self):
        epoch = self.current_epoch
        warmup = self.hparams.beta_warmup_epochs
        anneal = self.hparams.beta_anneal_epochs

        if epoch < warmup:
            return 0.0
        
        progress = min((epoch - warmup) / max(1, anneal), 1.0)

        return (1 - math.cos(progress * math.pi)) * self.hparams.beta_max * 0.5
    
    def vae_tabular_loss(self, x, x_recon, mu, logvar):
        beta = self._get_beta()
        self.log("beta", beta, prog_bar=True)
        x_recon = x_recon
        recon_loss = F.mse_loss(x_recon, x)

        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        total = recon_loss + beta * kl
        return total, recon_loss, kl
        

    def _compute_loss(self, x, x_recon, mu, logvar, mode='train'):
        loss, recon_cont, kl = self.vae_tabular_loss(x.float(), x_recon, mu, logvar)
        self.log(f"{mode}_loss", loss, prog_bar=True)
        self.log(f"{mode}_recon_cont", recon_cont)
        self.log(f"{mode}_kl_div", kl)
        return loss
    
    def training_step(self, batch):
        x, y = batch
        y = y.unsqueeze(1)
        x_recon, mu, logvar = self(x.float(), y)
        loss = self._compute_loss(x, x_recon, mu, logvar, mode='train')
        return loss

    def validation_step(self, batch):
        x, y = batch
        y = y.unsqueeze(1)
        x_recon, mu, logvar = self(x.float(), y)
        loss = self._compute_loss(x, x_recon, mu, logvar, mode='val')
        return loss
    
    @torch.no_grad()
    def generate(self, n_samples=10, y=None):
        z = torch.randn(n_samples, self.hparams.latent_dim).to(self.device)
        if y is None:
            y = torch.randint(0, 2, (n_samples, self.hparams.label_dim), dtype=torch.float32, device=self.device)
        else:
            y = torch.as_tensor(y, dtype=torch.float32, device=self.device)
            if y.dim() == 1:
                y = y.unsqueeze(1)
        samples = self.decode(z, y)
        samples = samples * self.feature_std + self.feature_mean
        return samples, y

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        return optimizer

if __name__ == "__main__": 
    dummy_x = torch.randn(4, 30)
    dummy_y = torch.tensor([[0], [1], [0], [1]], dtype=torch.float32)
    model = VAEModule(feature_dim=30, label_dim=1, latent_dim=16)
    x_recon, mu, logvar = model(dummy_x, dummy_y)
    print("Output shape:", x_recon.shape)
    print("Mu shape:", mu.shape)
    print("Logvar shape:", logvar.shape)    