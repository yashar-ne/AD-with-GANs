import numpy as np
import torch
import torch.nn as nn


class StyleGANReconstructor(nn.Module):
    def __init__(self, directions_count, num_channels=3, width=2, shape=[14, 6, 256, 256]):
        super(StyleGANReconstructor, self).__init__()
        self.shape = shape

        self.conv = nn.Sequential(
            nn.Conv2d(num_channels * 2, 3 * width, kernel_size=2),
            # nn.BatchNorm2d(3 * width),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(3 * width, 8 * width, kernel_size=2),
            # nn.BatchNorm2d(8 * width),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8 * width, 60 * width, kernel_size=4),
            # nn.BatchNorm2d(60 * width),
            nn.ReLU()
        )

        self.fc_logits = nn.Sequential(
            nn.Linear(60 * width, 42 * width),
            # nn.BatchNorm1d(42 * width),
            nn.ReLU(),
            nn.Linear(42 * width, np.prod(directions_count))
        )
        self.fc_shift = nn.Sequential(
            nn.Linear(60 * width, 42 * width),
            # nn.BatchNorm1d(42 * width),
            nn.ReLU(),
            nn.Linear(42 * width, 1)
        )

    def forward(self, x1, x2):
        batch_size = x1.shape[0]
        combined = torch.reshape(torch.cat([x1, x2], dim=3), shape=self.shape)
        features = self.conv(combined)
        features = features.mean(dim=[-1, -2])
        features = features.view(batch_size, -1)

        logits = self.fc_logits(features)
        shift = self.fc_shift(features)

        return logits, shift.squeeze()
