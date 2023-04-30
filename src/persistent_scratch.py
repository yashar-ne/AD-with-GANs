import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

noise_size = 10
feature_map_size = 4
out_channel_size = 1

x = torch.randn(1, noise_size, 1, 1)
print(x.shape)

deconv1 = nn.Sequential(
    nn.ConvTranspose2d(in_channels=noise_size,
                       out_channels=feature_map_size*4,
                       kernel_size=4,
                       stride=1,
                       padding=0,
                       bias=False),
)
x = deconv1(x)
print(x.shape)

deconv2 = nn.Sequential(
    nn.ConvTranspose2d(in_channels=feature_map_size*4,
                       out_channels=feature_map_size*2,
                       kernel_size=3,
                       stride=2,
                       padding=1,
                       bias=False),
)
x = deconv2(x)
print(x.shape)

deconv3 = nn.Sequential(
    nn.ConvTranspose2d(in_channels=feature_map_size*2,
                       out_channels=feature_map_size*2,
                       kernel_size=4,
                       stride=2,
                       padding=1,
                       bias=False),
)
x = deconv3(x)
print(x.shape)

deconv4 = nn.Sequential(
    nn.ConvTranspose2d(in_channels=feature_map_size*2,
                       out_channels=out_channel_size,
                       kernel_size=4,
                       stride=2,
                       padding=1,
                       bias=False),
)
x = deconv4(x)
print(x.shape)


###################################################





# print(x)

# img = x.detach().numpy().squeeze()
# print(img)
#
# new_im = Image.fromarray(img)
# new_im.show()
#
# print(x)
# print(x.shape)
#
# x = torch.transpose(x, 0, 2)
# print(x.shape)
#
# x = torch.reshape(x, (1, 4, 4))
# print(x.shape)
#
# trans = transforms.ToPILImage()
# img = trans(x)
# img.show()


# conv = nn.Sequential(nn.Conv2d(1, 5, 2, 1, 0, bias=False))
# print(conv(x))
