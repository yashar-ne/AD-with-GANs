import torch
import tqdm

from src.ml.models.discriminator import Discriminator
from src.ml.models.generator import Generator


class LatentSpaceMapper:
    def __init__(self, generator: Generator, discriminator: Discriminator, device):
        self.generator: Generator = generator
        self.generator.to(device)
        self.discriminator: Discriminator = discriminator
        self.discriminator.to(device)
        self.device = device

        self.criterion = torch.nn.MSELoss()

    def map_image_to_point_in_latent_space(self, image: torch.Tensor, size_z=100, max_opt_iterations=30000,
                                           opt_threshold=140.0, plateu_threshold=3.0, check_every_n_iter=4000,
                                           learning_rate=0.4, print_every_n_iters=10000,
                                           ignore_rules_below_threshold=50, retry_after_n_iters=10000,
                                           immediate_retry_threshold=200):
        image.to(self.device)
        z = torch.randn(1, size_z, 1, 1, device=self.device, requires_grad=True)
        z_optimizer = torch.optim.Adam([z], lr=learning_rate)
        losses = []
        final_loss = 0
        latest_checkpoint_loss = 0

        # scheduler = lr_scheduler.LinearLR(z_optimizer, start_factor=0.4, end_factor=0.001, total_iters=max_opt_iterations-(math.floor(max_opt_iterations*0.2)))
        # scheduler = lr_scheduler.StepLR(z_optimizer, step_size=max_opt_iterations, gamma=0.9)
        # scheduler = torch.optim.lr_scheduler.CyclicLR(z_optimizer, base_lr=0.01, max_lr=0.4, cycle_momentum=False)
        for i in range(max_opt_iterations):
            retry = False

            fake = self.generator(z.to(self.device))
            _, f_real = self.discriminator(image.to(self.device))
            _, f_fake = self.discriminator(fake)

            loss_r = self.criterion(image.to(self.device), fake)
            loss_d = self.criterion(f_real, f_fake)
            loss = (1 - 0.1) * loss_r + 0.1 * loss_d

            # loss = self.__get_anomaly_score(z, image.to(self.device))
            final_loss = loss.data.item()

            if i == 1:
                latest_checkpoint_loss = loss.data.item()

            if loss.data.item() < opt_threshold:
                print(f"Iteration: {i} -- Reached Defined Optimum -- Final Loss: {loss.data.item()}")
                break

            if (i % print_every_n_iters == 0 and i != 0) or (i == max_opt_iterations - 1):
                print(
                    f"Iteration: {i} -- Current Loss: {loss.data.item()} -- Current Learning-Rate: {z_optimizer.param_groups[0]['lr']}")
                losses.append(loss.data.item())

            if i % check_every_n_iter == 0 and i != 0:
                if abs(loss.data.item() - latest_checkpoint_loss) < plateu_threshold:
                    print(f"Reached Plateu at Iteration {i} -- Loss: {loss.data.item()}")
                    retry = True
                    break
                if loss.data.item() > immediate_retry_threshold:
                    print(f"Loss at Iteration {i} too high -- Loss: {loss.data.item()}")
                    retry = True
                    break
                latest_checkpoint_loss = loss.data.item()

            if i == retry_after_n_iters and loss.data.item() > ignore_rules_below_threshold:
                retry = True
                break

            z_optimizer.zero_grad()
            loss.backward()
            z_optimizer.step()
            # scheduler.step()

        return z, final_loss, retry

    def __get_anomaly_score(self, z, real):
        lamda = 0.1
        fake = self.generator(z.to(self.device))
        loss_r = torch.sum(torch.abs(real - fake))

        # return loss_r

        _, f_real = self.discriminator(real)
        _, f_fake = self.discriminator(fake)
        loss_d = torch.sum(torch.abs(f_real - f_fake))

        lossR = self.criterion(real, fake)
        lossD = self.criterion(f_real, f_fake)
        # loss = (1 - alpha) * lossR + alpha * lossD

        # return (1 - lamda) * loss_r + lamda * loss_d
        return (1 - lamda) * lossR + lamda * lossD
