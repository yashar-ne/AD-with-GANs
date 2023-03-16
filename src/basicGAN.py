import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor

import matplotlib.pyplot as plt

from models.discriminator import Discriminator
from models.generator import Generator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.1307,), std=(0.3081,))])

train_dataset = datasets.MNIST(
    root='data',
    train=True,
    transform=transform,
    download=False,
)
test_dataset = datasets.MNIST(
    root='data',
    train=False,
    transform=ToTensor()
)

# figure = plt.figure(figsize=(10, 8))
# cols, rows = 10, 10
# for i in range(1, cols * rows + 1):
#     sample_idx = torch.randint(len(test_dataset), size=(1,)).item()
#     img, label = train_dataset[sample_idx]
#     figure.add_subplot(rows, cols, i)
#     plt.axis("off")
#     plt.imshow(img.squeeze(), cmap="gray")
# plt.show()


batch_size = 128
num_classes = 10
learning_rate = 0.002
num_epochs = 5
num_color_channels = 1
num_feature_maps_g = 32
num_feature_maps_d = 32
size_z = 100
adam_beta1 = 0.2
num_gpu = 0

dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                         batch_size=batch_size,
                                         shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


generator = Generator(size_z=size_z,
                      num_feature_maps=num_feature_maps_g,
                      num_color_channels=num_color_channels).to(device)

# generator.apply(weights_init)

discriminator = Discriminator(num_feature_maps=num_feature_maps_d,
                              num_color_channels=num_color_channels).to(device)

# discriminator.apply(weights_init)

criterion = nn.BCELoss()

fixed_noise = torch.randn(16, size_z, 1, 1, device=device)

real_label = 1.
fake_label = 0.

optimizerG = optim.Adam(generator.parameters(), lr=learning_rate, betas=(adam_beta1, 0.999))
optimizerD = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(adam_beta1, 0.999))

# Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
for epoch in range(num_epochs):
    for i, (real_images, _) in enumerate(dataloader):
        bs = real_images.shape[0]
        ##############################
        #   Training discriminator   #
        ##############################

        discriminator.zero_grad()
        real_images = real_images.to(device)
        label = torch.full((bs,), real_label, device=device)

        output = discriminator(real_images)
        lossD_real = criterion(output, label)
        lossD_real.backward()
        D_x = output.mean().item()

        noise = torch.randn(bs, size_z, 1, 1, device=device)
        fake_images = generator(noise)
        label.fill_(fake_label)
        output = discriminator(fake_images.detach())
        lossD_fake = criterion(output, label)
        lossD_fake.backward()
        D_G_z1 = output.mean().item()
        lossD = lossD_real + lossD_fake
        optimizerD.step()

        ##########################
        #   Training generator   #
        ##########################

        generator.zero_grad()
        label.fill_(real_label)
        output = discriminator(fake_images)
        lossG = criterion(output, label)
        lossG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        if (i + 1) % 100 == 0:
            print(
                'Epoch [{}/{}], step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, Discriminator - D(G(x)): {:.2f}, Generator - D(G(x)): {:.2f}'
                .format(
                    epoch + 1,
                    num_epochs,
                    i + 1,
                    batch_size,
                    lossD.item(),
                    lossG.item(),
                    D_x,
                    D_G_z1,
                    D_G_z2
                )
            )
    generator.eval()
    generator.train()

plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses, label="G")
plt.plot(D_losses, label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()
