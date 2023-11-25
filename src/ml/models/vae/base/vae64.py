# https://github.com/ttchengab/VAE/tree/main
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.distributions import Normal


class VAE64(nn.Module):
    def __init__(self,
                 device,
                 img_channels=1,
                 feature_dim=64 * 58 * 58,
                 z_dim=20,
                 num_epochs=30,
                 l: int = 10,
                 learning_rate=1e-3,
                 beta=1,
                 kl_weight=1):
        super(VAE64, self).__init__()
        self.z_dim = z_dim
        self.num_epochs = num_epochs
        self.l = l
        self.learning_rate = learning_rate
        self.device = device
        self.beta = beta
        self.kl_weight = kl_weight

        # Initializing the 2 convolutional layers and 2 full-connected layers for the encoder
        self.encConv1 = nn.Conv2d(img_channels, 16, 3, stride=1)
        self.encConv2 = nn.Conv2d(16, 32, 3, stride=1)
        self.encConv3 = nn.Conv2d(32, 64, 3, stride=1)
        self.encFC1 = nn.Linear(feature_dim, z_dim)
        self.encFC2 = nn.Linear(feature_dim, z_dim)

        # Initializing the fully-connected layer and 2 convolutional layers for decoder
        self.decFC1 = nn.Linear(z_dim, feature_dim)
        self.decConv1 = nn.ConvTranspose2d(64, 32, 3, stride=1)
        self.decConv2 = nn.ConvTranspose2d(32, 16, 3, stride=1)
        self.decConv3 = nn.ConvTranspose2d(16, img_channels, 3, stride=1)

        self.to(self.device)

    def encoder(self, x):
        # Input is fed into 2 convolutional layers sequentially
        # The output feature map are fed into 2 fully-connected layers to predict mean (mu) and variance (log_var)
        # Mu and log_var are used for generating middle representation z and KL divergence loss
        x = self.encConv1(x)
        x = torch.relu(x)

        x = self.encConv2(x)
        x = torch.relu(x)

        x = self.encConv3(x)
        x = torch.relu(x)

        x = x.view(-1, 64 * 58 * 58)

        mu = self.encFC1(x)
        log_var = self.encFC2(x)

        return mu, log_var

    def decoder(self, z):
        # z is fed back into a fully-connected layers and then into two transpose convolutional layers
        # The generated output is the same size of the original input
        x = self.decFC1(z)
        x = torch.relu(x)

        x = x.view(-1, 64, 58, 58)

        x = self.decConv1(x)
        x = torch.relu(x)

        x = self.decConv2(x)
        x = torch.relu(x)

        img = torch.sigmoid(self.decConv3(x))
        return img

    def forward(self, x):
        # The entire pipeline of the VAE: encoder -> reparameterization -> decoder
        # output, mu, and log_var are returned for loss computation
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        out = self.decoder(z)
        return out, mu, log_var

    @staticmethod
    def reparameterize(mu, log_var):
        # Reparameterization takes in the input mu and log_var and sample the mu + std * eps
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def train_model(self, dataloader):
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        for epoch in range(self.num_epochs):
            for idx, data in enumerate(dataloader, 0):
                imgs, _ = data
                imgs = imgs.to(self.device)

                out, mu, log_var = self(imgs)
                loss, (_, _) = self.compute_loss(imgs, out, mu, log_var)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f'Epoch {epoch}: Loss {loss}')

    def compute_loss(self, x, output, mu, log_var):
        # First we compare how well we have recreated the image
        mse_loss = nn.functional.binary_cross_entropy(output.view(x.shape[0], -1),
                                                      x.to(self.device).view(x.shape[0], -1))

        # Then the KL_divergence
        kl_div = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        loss = mse_loss + self.beta * self.kl_weight * kl_div
        return loss, (mse_loss, kl_div)

    def predict(self, x):
        output, _, _ = self.forward(x)
        return nn.functional.binary_cross_entropy(output.view(x.shape[0], -1),
                                                  x.to(self.device).view(x.shape[0], -1))

    def compute_reconstruction_probability(self, img):
        batch_size = len(img)
        latent_mu, latent_sigma = self.encoder(img)
        latent_sigma = F.softplus(latent_sigma)
        dist = Normal(latent_mu, latent_sigma)
        z = dist.rsample([self.l])  # shape: [L, batch_size, latent_size]
        z = z.view(self.l * batch_size, self.z_dim)
        recon_mu, recon_sigma = self.decoder(z)
        recon_sigma = F.softplus(recon_sigma)
        recon_mu = recon_mu.view(self.l, *img.shape)
        recon_sigma = recon_sigma.view(self.l, *img.shape)

        recon_dist = Normal(recon_mu, recon_sigma)
        x = img.unsqueeze(0)
        p = recon_dist.log_prob(x).exp().mean(dim=0).mean(dim=-1)  # vector of shape [batch_size]
        return p

    def draw_samples(self, dataset, num_samples=1):
        self.eval()
        with (torch.no_grad()):
            for data in random.sample(list(dataset), num_samples):
                imgs, _ = data
                imgs = imgs.to(self.device)
                img = np.transpose(imgs[0].cpu().numpy(), [1, 2, 0])
                plt.subplot(121)
                plt.imshow(np.squeeze(img))
                out, mu, log_var = self(imgs)
                out_img = np.transpose(out[0].cpu().numpy(), [1, 2, 0])
                plt.subplot(122)
                plt.imshow(np.squeeze(out_img))
                plt.show()

    def predict_samples(self, dataset, num_samples=10):
        self.eval()
        for data in random.sample(list(dataset), num_samples):
            imgs, _ = data
            loss = self.predict(imgs.to(self.device))
            print(loss.item())
