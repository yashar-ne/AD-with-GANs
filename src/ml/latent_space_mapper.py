import torch

from src.ml.models.discriminator import Discriminator
from src.ml.models.generator import Generator


class LatentSpaceMapper:

    def __init__(self, generator: Generator, discriminator: Discriminator, device):
        self.generator: Generator = generator
        self.discriminator: Discriminator = discriminator
        self.device = device

    def map_image_to_point_in_latent_space(self, image: torch.Tensor, size_z=100, opt_iterations=10000):
        z = torch.randn(1, size_z, 1, 1, device=self.device, requires_grad=True)
        z_optimizer = torch.optim.Adam([z], lr=1e-4)
        losses = []

        for i in range(opt_iterations):
            loss = self.__get_anomaly_score(z, image.unsqueeze(0).to(self.device))
            loss.backward()
            z_optimizer.step()
            if i % 1000 == 0:
                # print(f"Iteration: {i} -- Loss: {loss.data.item()}")
                losses.append(loss.data.item())

        return z

    def __get_anomaly_score(self, z, x_query):
        lamda = 0.1
        g_z = self.generator(z.to(self.device))
        _, x_prop = self.discriminator(x_query)
        _, g_z_prop = self.discriminator(g_z)

        loss_r = torch.sum(torch.abs(x_query - g_z))
        loss_d = torch.sum(torch.abs(x_prop - g_z_prop))

        return (1 - lamda) * loss_r + lamda * loss_d
