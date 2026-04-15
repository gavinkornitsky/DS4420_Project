import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as l

class VAEModule(l.LightningModule):
    def __init__(
        self,
        feature_dim = 30,
        label_dim = 2,
        latent_dim = 16,
        hidden_dims = [256, 128, 64],
        dropout = 0.1,
        lr = 1e-4,
        weight_decay = 1e-5,
        label_weight = 1.0,
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

        self.prior_mu = nn.Parameter(torch.randn(label_dim, latent_dim))
        self.prior_logvar = nn.Parameter(torch.zeros(label_dim, latent_dim))

        enc_in = feature_dim
        enc_layers = []
        for h in hidden_dims:
            enc_layers += [nn.Linear(enc_in, h), nn.ReLU(), nn.Dropout(dropout)]
            enc_in = h
        self.encoder = nn.Sequential(*enc_layers)
        self.mu_layer = nn.Linear(hidden_dims[-1], latent_dim)
        self.logvar_layer = nn.Linear(hidden_dims[-1], latent_dim)

        dec_in = latent_dim
        dec_layers = []
        for h in reversed(hidden_dims):
            dec_layers += [nn.Linear(dec_in, h), nn.ReLU(), nn.Dropout(dropout)]
            dec_in = h
        self.decoder = nn.Sequential(*dec_layers)
        # outputs per-feature (mu, logvar) -> 2 * feature_dim
        self.fc_features = nn.Linear(dec_in, 2 * feature_dim)

        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[-1], 1),
        )

    def get_prior(self, y):
        idx = y.long().view(-1)
        return self.prior_mu[idx], self.prior_logvar[idx]

    def encode(self, x):
        h = self.encoder(x)
        return self.mu_layer(h), self.logvar_layer(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder(z)
        out = self.fc_features(h)
        mu_x, logvar_x = out.chunk(2, dim=-1)
        return mu_x, logvar_x

    def classify(self, z):
        return self.classifier(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        mu_x, logvar_x = self.decode(z)
        y_logit = self.classify(z)
        return mu_x, logvar_x, y_logit, mu, logvar

    def _get_beta(self):
        epoch = self.current_epoch
        warmup = self.hparams.beta_warmup_epochs
        anneal = self.hparams.beta_anneal_epochs

        if epoch < warmup:
            return 0.0
        progress = min((epoch - warmup) / max(1, anneal), 1.0)

        return (1 - math.cos(progress * math.pi)) * self.hparams.beta_max * 0.5

    def vae_conditional_loss(self, x, mu_x, logvar_x, mu, logvar, y, y_hat):
        beta = self._get_beta()
        self.log("beta", beta, prog_bar=True)

        nll_elem = 0.5 * (
            logvar_x
            + (x - mu_x).pow(2) / logvar_x.exp()
            + math.log(2 * math.pi)
        )
        recon_feat_loss = nll_elem.mean()

        recon_label_loss = F.binary_cross_entropy_with_logits(y_hat, y, reduction='mean')

        prior_mu, prior_logvar = self.get_prior(y)

        kl = 0.5 * torch.sum(prior_logvar - logvar + (logvar.exp() + (mu - prior_mu).pow(2)) / prior_logvar.exp() - 1, dim=-1).mean()

        total = recon_feat_loss + self.hparams.label_weight * recon_label_loss + beta * kl
        return total, recon_feat_loss, recon_label_loss, kl

    def _compute_loss(self, x, mu_x, logvar_x, mu, logvar, y, y_hat, mode='train'):
        loss, recon_cont, recon_label, kl = self.vae_conditional_loss(x.float(), mu_x, logvar_x, mu, logvar, y, y_hat)
        self.log(f"{mode}_loss", loss, prog_bar=True)
        self.log(f"{mode}_recon_cont", recon_cont)
        self.log(f"{mode}_recon_label", recon_label)
        self.log(f"{mode}_kl_div", kl)
        self.log(f"{mode}_decoder_sigma", (0.5 * logvar_x).exp().mean())
        return loss

    def training_step(self, batch):
        x, y = batch
        y = y.unsqueeze(1)
        mu_x, logvar_x, y_hat, mu, logvar = self(x.float())
        loss = self._compute_loss(x, mu_x, logvar_x, mu, logvar, y, y_hat, mode='train')
        return loss

    def validation_step(self, batch):
        x, y = batch
        y = y.unsqueeze(1)
        mu_x, logvar_x, y_hat, mu, logvar = self(x.float())
        loss = self._compute_loss(x, mu_x, logvar_x, mu, logvar, y, y_hat, mode='val')
        return loss

    @torch.no_grad()
    def generate(self, n_samples=1000, class_ratio=0.5, temperature=1.0, sample_decoder=True, seed=None):
        self.eval()
        device = self.device

        if seed is not None:
            torch.manual_seed(seed)

        n_pos = int(n_samples * class_ratio)
        n_neg = n_samples - n_pos
        y = torch.cat([torch.zeros(n_neg), torch.ones(n_pos)]).to(device)
        prior_mu, prior_logvar = self.get_prior(y)
        prior_std = (0.5 * prior_logvar).exp() * temperature
        z = prior_mu + torch.randn_like(prior_std) * prior_std

        mu_x, logvar_x = self.decode(z)
        if sample_decoder:
            std_x = (0.5 * logvar_x).exp()
            x_hat = mu_x + torch.randn_like(mu_x) * std_x
        else:
            x_hat = mu_x
        samples = x_hat * self.feature_std + self.feature_mean
        return samples, y

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams.lr,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.3,
            anneal_strategy="cos",
            div_factor=25.0,
            final_div_factor=1e4,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }


if __name__ == "__main__":
    dummy_x = torch.randn(4, 30)
    model = VAEModule(feature_dim=30, label_dim=2, latent_dim=16)
    mu_x, logvar_x, y_hat, mu, logvar = model(dummy_x)
    print("mu_x shape:", mu_x.shape)
    print("logvar_x shape:", logvar_x.shape)
    print("y_hat shape:", y_hat.shape)
    print("mu shape:", mu.shape)
    print("logvar shape:", logvar.shape)
    samples, ys = model.generate(n_samples=8, class_ratio=0.5, seed=0)
    print("samples shape:", samples.shape, "| labels:", ys.tolist())
