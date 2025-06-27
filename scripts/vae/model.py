import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model ---
class Encoder(nn.Module):
    def __init__(self, in_channels: int = 6, latent_dim: int = 32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=1, padding=1), nn.PReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.PReLU(),
            nn.Conv2d(32, 64, 3, stride=1, padding=1), nn.PReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.PReLU(),
            nn.Conv2d(64, 128, 3, stride=1, padding=1), nn.PReLU(),
            nn.Conv2d(128, 128, 3, stride=2, padding=1), nn.PReLU(),
            nn.Conv2d(128, 256, 3, stride=1, padding=1), nn.PReLU(),
            nn.Conv2d(256, 256, 3, stride=2, padding=1), nn.PReLU(),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(256 * 3 * 3, latent_dim)
        self.fc_logvar = nn.Linear(256 * 3 * 3, latent_dim)

    def forward(self, x):
        x = self.conv(x)
        return self.fc_mu(x), self.fc_logvar(x)

class Decoder(nn.Module):
    def __init__(self, in_channels: int, latent_dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256 * 3 * 3), nn.PReLU()
        )
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1), nn.PReLU(),
            nn.ConvTranspose2d(128, 128, 3, stride=2, padding=1, output_padding=1), nn.PReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), nn.PReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), nn.PReLU(),
            nn.Conv2d(32, in_channels, 3, padding=1),
            nn.Sigmoid()  # Matches paper output in [0,1]
        )

    def forward(self, z):
        x = self.fc(z).view(-1, 256, 3, 3)
        x = self.conv(x)
        x = F.interpolate(x, size=(45, 45), mode='bilinear', align_corners=False)
        return x

class VAE(nn.Module):
    def __init__(self, in_channels: int = 6, latent_dim: int = 32, beta: float = 0.01):
        super().__init__()
        self.encoder = Encoder(in_channels, latent_dim)
        self.decoder = Decoder(in_channels, latent_dim)
        self.latent_dim = latent_dim
        self.beta = beta

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar

    def loss_function(self, x, x_hat, mu, logvar):
        recon_loss = F.l1_loss(x_hat, x, reduction='mean')

        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        loss = recon_loss + self.beta * kl
        return {
            'loss': loss,
            'reconstruction_loss': recon_loss,
            'kl_divergence': kl
        }

class LatentRegressor(nn.Module):
    def __init__(self, latent_dim, output_dim=3):  # output_dim = redshift, e1, e2
        super().__init__()
        self.back = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.PReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.PReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.PReLU()
            )
        self.redshift = nn.Sequential(
            nn.Linear(32, 1),
            nn.ReLU() # Output > 0
        )
        self.ellipticity = nn.Sequential(
            nn.Linear(32, 2),
            nn.Tanh() # Output in [-1,1]
        )
    def forward(self, z):
        h = self.back(z)
        z_pred = self.redshift(h)
        e_pred = self.ellipticity(h)
        return z_pred, e_pred

class LatentRegressorRedshift(nn.Module):
    def __init__(self, latent_dim, output_dim=3):  # output_dim = redshift, e1, e2
        super().__init__()
        self.back = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.PReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.PReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.PReLU()
            )
        self.redshift = nn.Sequential(
            nn.Linear(32, 1),
            nn.ReLU() # Output > 0
        )
    def forward(self, z):
        h = self.back(z)
        z_pred = self.redshift(h)
        return z_pred

class LatentRegressorEllipticity(nn.Module):
    def __init__(self, latent_dim, output_dim=3):  # output_dim = redshift, e1, e2
        super().__init__()
        self.back = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.PReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.PReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.PReLU()
            )
        self.ellipticity = nn.Sequential(
            nn.Linear(32, 2),
            nn.Tanh() # Output in [-1,1]
        )
    def forward(self, z):
        h = self.back(z)
        e_pred = self.ellipticity(h)
        return e_pred
