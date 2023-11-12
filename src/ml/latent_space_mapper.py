import torch

from src.ml.models.base.discriminator import Discriminator
from src.ml.models.base.generator import Generator


class LatentSpaceMapper:
    def __init__(self, generator: Generator, discriminator: Discriminator, device):
        self.generator: Generator = generator
        self.generator.to(device)
        self.discriminator: Discriminator = discriminator
        self.discriminator.to(device)
        self.device = device

        self.criterion = torch.nn.MSELoss()

    def map_image_to_point_in_latent_space(self,
                                           image: torch.Tensor,
                                           size_z=100,
                                           n_iterations=30000,
                                           retry_check_after_iter=4000,
                                           learning_rate=0.001,
                                           print_every_n_iters=10000,
                                           retry_threshold=200):
        image.to(self.device)
        z = torch.randn(1, size_z, 1, 1, device=self.device, requires_grad=True)
        z_optimizer = torch.optim.Adam([z], lr=learning_rate)
        losses = []
        final_loss = 0

        scheduler = torch.optim.lr_scheduler.CyclicLR(z_optimizer, base_lr=0.001, max_lr=0.2, cycle_momentum=False)

        for i in range(n_iterations):
            retry = False
            loss = self.__get_anomaly_score(z, image)
            final_loss = loss.data.item()

            if (i % print_every_n_iters == 0 and i != 0) or (i == n_iterations - 1):
                print(
                    f"Iteration: {i} -- Current Loss: {loss.data.item()} -- Current Learning-Rate: {z_optimizer.param_groups[0]['lr']}")
                losses.append(loss.data.item())

            if i == retry_check_after_iter:
                if loss.data.item() > retry_threshold:
                    print(f"Loss at Iteration {i} too high -- Loss: {loss.data.item()}")
                    retry = True
                    break

            z_optimizer.zero_grad()
            loss.backward()
            z_optimizer.step()
            scheduler.step()

        return z, final_loss, retry

    def __get_anomaly_score(self, z, image):
        lamda = 0.1
        fake = self.generator(z.to(self.device))
        _, f_real = self.discriminator(image.to(self.device))
        _, f_fake = self.discriminator(fake)

        loss_r = self.criterion(image.to(self.device), fake)
        loss_d = self.criterion(f_real, f_fake)
        loss = (1 - lamda) * loss_r + lamda * loss_d

        return loss
