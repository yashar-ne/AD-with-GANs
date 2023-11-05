from matplotlib import pyplot as plt
from torchvision.transforms import transforms

from src.ml.datasets.generate_dataset import get_dataloader
from src.ml.models.vae.base.solver import Solver
from src.ml.models.vae.base.vae import BetaVAE


def result_grid(n_col, n_rows, original_data, recreation_data):
    fig, axis = plt.subplots(n_rows, n_col * 2, figsize=(40, 20))
    for i in range(n_col * n_rows):
        original = original_data[i].reshape(28, 28)
        recreation = recreation_data.detach()[i].reshape(28, 28)

        i_original = i % n_col + int(i / n_col) * 2 * n_col
        i_recreation = i % n_col + n_col + int(i / n_col) * 2 * n_col

        ax = axis.flatten()[i_original]

        ax.imshow(original, aspect='auto', cmap='viridis')
        ax.axis('off')
        ax.grid(None)

        ax = axis.flatten()[i_recreation]

        ax.imshow(recreation.cpu(), aspect='auto', cmap='viridis')
        ax.axis('off')
        ax.grid(None)

    plt.tight_layout()
    plt.show()


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,),
                         std=(0.5,),
                         inplace=True),
])

train_set = get_dataloader(
    dataset_folder="/home/yashar/git/AD-with-GANs/data/DS8_fashion_mnist_shirt_sneaker/dataset_raw",
    batch_size=128,
    transform=transform)

model = BetaVAE(in_channels=1, latent_dim=10, kl_weight=1)
solver = Solver(model, train_set)

solver.train(50)
solver.plot_training_loss()

_, (X_test, _) = next(enumerate(train_set))
output, (_, _) = model.forward(X_test)

n_col = 6
n_rows = 6

result_grid(n_col, n_rows, X_test, output)
