# see https://github.com/AxelNathanson/pytorch-Variational-Autoencoder

import torch
import torch.nn as nn
import torch.utils.data


class BetaVAE(nn.Module):
    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 kl_weight: float,
                 image_dim: int = 28,
                 beta: int = 1):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.beta = beta
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight

        self.in_channels = in_channels
        self.image_dim = image_dim

        cnn_channels = [16, 32, 32, 32]
        self.channels_into_decoder = cnn_channels[2]

        # We need two Linear layers to convert encoder -> mu, sigma
        # But first we need to calculate how big the output from our network is.
        self.cnn_output_size = cnn_output_size(image_dim)
        encoder_output_size = cnn_channels[2] * self.cnn_output_size ** 2

        self.linear_mu = nn.Linear(encoder_output_size, self.latent_dim)
        self.linear_log_sigma = nn.Linear(encoder_output_size, self.latent_dim)

        self.upsample = nn.Linear(self.latent_dim, encoder_output_size)

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=cnn_channels[0], kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(cnn_channels[0], out_channels=cnn_channels[1], kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(cnn_channels[1], out_channels=cnn_channels[2], kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
        )

        self.decoder = nn.Sequential(
            nn.LeakyReLU(),
            nn.ConvTranspose2d(cnn_channels[2], out_channels=cnn_channels[1], kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(cnn_channels[1], out_channels=cnn_channels[0], kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(cnn_channels[0], out_channels=cnn_channels[3], kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(cnn_channels[3], out_channels=in_channels, kernel_size=4, padding=1)
        )

        self.activation = torch.sigmoid

    def encode(self, x):
        z = self.encoder(x)
        z = torch.flatten(z, start_dim=1)

        mu = self.linear_mu(z)
        log_sigma = self.linear_log_sigma(z)

        return mu, log_sigma

    def decode(self, x):
        z = self.upsample(x).view(-1, self.channels_into_decoder, self.cnn_output_size, self.cnn_output_size)
        z = self.decoder(z)

        return self.activation(z)

    @staticmethod
    def reparameterization(mu, log_sigma):
        # This is done to make sure we get a positive semi-definite cov-matrix.
        sigma = torch.exp(log_sigma * .5)

        # The reparameterization-trick
        z_tmp = torch.randn_like(sigma)
        z = mu + sigma * z_tmp
        return z

    def forward(self, x):
        mu, log_sigma = self.encode(x.to(self.device))
        z = self.reparameterization(mu=mu, log_sigma=log_sigma)
        output = self.decode(z).view(-1, self.in_channels, self.image_dim, self.image_dim)
        return output, (mu, log_sigma)

    def compute_loss(self, x, output, mu, log_sigma):
        # First we compare how well we have recreated the image
        mse_loss = nn.functional.binary_cross_entropy(output.to(self.device).view(x.shape[0], -1),
                                                      x.to(self.device).view(x.shape[0], -1))

        # Then the KL_divergence
        kl_div = torch.mean(-0.5 * torch.sum(1 + log_sigma - mu ** 2 - log_sigma.exp(), dim=1), dim=0)

        loss = mse_loss + self.beta * self.kl_weight * kl_div

        return loss, (mse_loss, kl_div)

    def sample(self, num_samples=1):
        sample = torch.randn(num_samples, self.latent_dim)

        return self.decode(sample)

    def sample_latent_space(self, x):
        mu, log_sigma = self.encode(x)
        z = self.reparameterization(mu, log_sigma)
        return z

    def test_encoder_decoder(self, x):
        print(x.shape)
        z = self.encoder(x)
        print(z.shape, self.cnn_output_size)

        d = self.decoder(z)
        print(d.shape)


def cnn_output_size(input_dim=28, num_channels=3):
    dim = input_dim
    for i in range(num_channels):
        dim = (dim - 3 + 2 * 1) / 2 + 1
    return int(dim)
