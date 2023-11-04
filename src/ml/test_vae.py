from torchvision.transforms import transforms

from src.ml.datasets.generate_dataset import get_dataloader
from src.ml.models.vae.base.solver import Solver
from src.ml.models.vae.base.vae import BetaVAE

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,),
                         std=(0.5,),
                         inplace=True),
])

train_set = dataloader = get_dataloader(
    dataset_folder="/home/yashar/git/AD-with-GANs/data/DS8_fashion_mnist_shirt_sneaker/dataset_raw",
    batch_size=128,
    transform=transform)

vae = BetaVAE(in_channels=1, latent_dim=10, kl_weight=1)
solver = Solver(vae, train_set)

solver.train(10)
solver.plot_training_loss()
