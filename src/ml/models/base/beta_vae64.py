# loosely following https://github.com/ttchengab/VAE/tree/main, https://github.com/AntixK/PyTorch-VAE/tree/master
# and https://github.com/1Konny/Beta-VAE/tree/master
import random

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.autograd import Variable


class BetaVAE64(nn.Module):
    def __init__(self,
                 device,
                 num_color_channels=1,
                 z_dim=100,
                 num_epochs=30,
                 learning_rate=1e-3,
                 beta=4,
                 kl_weight=1):
        super(BetaVAE64, self).__init__()
        self.z_dim = z_dim
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.device = device
        self.beta = beta
        self.kl_weight = kl_weight
        self.dataloader = None

        self.encConv1 = nn.Conv2d(num_color_channels, 64, 4, 2, 1)
        self.encConv2 = nn.Conv2d(64, 32, 4, 2, 1)
        self.encConv3 = nn.Conv2d(32, 32, 4, 2, 1)
        self.encConv4 = nn.Conv2d(32, 64, 3, 2, 1)
        self.encConv5 = nn.Conv2d(64, 64, 4, 2, 1)
        self.encConv6 = nn.Conv2d(64, 256, 4, 2, 1)
        self.encFC = nn.Linear(256, z_dim * 2)

        self.decFC = nn.Linear(z_dim, 256)
        self.decConv1 = nn.ConvTranspose2d(256, 64, 4)
        self.decConv2 = nn.ConvTranspose2d(64, 64, 4, 2, 1)
        self.decConv3 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.decConv4 = nn.ConvTranspose2d(32, 32, 4, 2, 1)
        self.decConv5 = nn.ConvTranspose2d(32, num_color_channels, 4, 2, 1)

        self.to(self.device)

    def encoder(self, x):
        x = self.encConv1(x)
        x = torch.relu(x)

        x = self.encConv2(x)
        x = torch.relu(x)

        x = self.encConv3(x)
        x = torch.relu(x)

        x = self.encConv4(x)
        x = torch.relu(x)

        x = self.encConv5(x)
        x = torch.relu(x)

        x = self.encConv6(x)
        x = torch.relu(x)

        x = x.view(-1, 256 * 1 * 1)

        x = self.encFC(x)

        return x

    def decoder(self, z):
        x = self.decFC(z)
        x = torch.relu(x)

        x = x.view(-1, 256, 1, 1)

        x = self.decConv1(x)
        x = torch.relu(x)

        x = self.decConv2(x)
        x = torch.relu(x)

        x = self.decConv3(x)
        x = torch.relu(x)

        x = self.decConv4(x)
        x = torch.relu(x)

        x = self.decConv5(x)
        x = torch.sigmoid(x)

        return x

    def forward(self, x):
        distributions = self.encoder(x)
        mu = distributions[:, :self.z_dim]
        log_var = distributions[:, self.z_dim:]
        z = self.reparameterize(mu, log_var)
        out = self.decoder(z).view(x.size())
        return out, mu, log_var

    @staticmethod
    def reparameterize(mu, log_var):
        # Reparameterization takes in the input mu and log_var and sample the mu + std * eps
        std = log_var.div(2).exp()
        eps = Variable(std.data.new(std.size()).normal_())
        return mu + std * eps

    def train_model(self, dataloader):
        self.train()
        self.dataloader = dataloader
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        for epoch in range(self.num_epochs):
            for batch_idx, data in enumerate(dataloader, 0):
                imgs, _, _ = data
                imgs = imgs.to(self.device)

                out, mu, log_var = self(imgs)
                loss = self.compute_loss(imgs, out, mu, log_var)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if epoch % 10 == 0:
                self.draw_samples(dataloader, 1)
            print(f'Epoch {epoch}/{self.num_epochs}: Loss {loss}')

    def compute_loss(self, x, output, mu, log_var):
        reconstruction_loss = nn.functional.binary_cross_entropy(output, x, reduction='sum')
        kl_d = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        return reconstruction_loss + self.beta * kl_d

    def get_reconstruction_loss(self, x):
        output, _, _ = self.forward(x)
        return nn.functional.binary_cross_entropy(output, x, reduction='sum')

    def draw_samples(self, dataset, num_samples=1):
        self.eval()
        with (torch.no_grad()):
            for data in random.sample(list(dataset), num_samples):
                imgs, _, _ = data
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
            loss = self.get_reconstruction_loss(imgs.to(self.device))
            print(loss.item())
