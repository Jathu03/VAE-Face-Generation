import torch
import torch.nn as nn

class LinearVariationalAutoEncoder(nn.Module):
    def __init__(self, latent_dim=20):
        super().__init__()
        # Encoder network
        self.encoder = nn.Sequential(
            nn.Linear(3*32*32, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        # Mean and log variance layers
        self.fn_mu = nn.Linear(32, latent_dim)
        self.fn_logvar = nn.Linear(32, latent_dim)
        # Decoder network
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 3*32*32),
            nn.Sigmoid()
        )

    def forward_enc(self, x):
        x = self.encoder(x)
        mu = self.fn_mu(x)
        logvar = self.fn_logvar(x)
        sigma = torch.exp(0.5 * logvar)
        noise = torch.rand_like(sigma, device=sigma.device)
        z = mu + sigma * noise
        return z, mu, logvar

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        x = x.flatten(1)  # Flatten to [batch_size, num_channels * height * width]
        z, mu, logvar = self.forward_enc(x)
        dec = self.decoder(z)
        dec = dec.reshape(batch_size, channels, height, width)
        return z, mu, logvar, dec

    def forward_dec(self, z):
        return self.decoder(z)
