import torch
from torchvision.transforms import transforms

from src.ml.datasets.generate_dataset import get_dataloader
from src.ml.models.base.beta_vae64 import BetaVAE64
from src.ml.validation.vae_validation import get_vae_roc_auc_for_image_data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 512
num_epochs = 40
num_color_channels = 3
transform = transforms.Compose([transforms.ToTensor()])

full_dataloader = get_dataloader(
    # dataset_folder="/home/yashar/git/AD-with-GANs/data/DS12_mnist_9_6/dataset_raw",
    dataset_folder="/home/yashar/git/AD-with-GANs/data/DS13_celeba_bald/dataset_raw",
    batch_size=batch_size,
    transform=transform)

net = BetaVAE64(device=device, num_color_channels=num_color_channels, num_epochs=num_epochs)
roc_auc = get_vae_roc_auc_for_image_data(root_dir='/data', dataset_name='DS13_celeba_bald')

print(roc_auc)

# net.train_model(full_dataloader)
# net.draw_samples(full_dataloader, 10)
# net.predict_samples(full_dataloader)
