import torch
from torchvision.transforms import transforms

from src.ml.datasets.generate_dataset import get_dataloader
from src.ml.models.vae.base.vae64 import VAE64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 512
learning_rate = 1e-3
num_epochs = 30
transform = transforms.Compose([transforms.ToTensor()])

full_dataloader = get_dataloader(
    dataset_folder="/home/yashar/git/AD-with-GANs/data/DS14_fashionmnist_shirt_sneaker/dataset_raw",
    batch_size=batch_size,
    transform=transform)

net = VAE64(device=device, img_channels=1, num_epochs=100)
net.train_model(full_dataloader)
net.draw_samples(full_dataloader, 10)
net.predict_samples(full_dataloader)

# d = next(iter(full_dataloader))
# loss = net.compute_reconstruction_probability(d[0][0].to(net.device))
#
# print(loss)
