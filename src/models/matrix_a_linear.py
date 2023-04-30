import torch
import torch.nn as nn


class MatrixALinear(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True):
        super(MatrixALinear, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.linear = nn.Linear(self.input_dim, self.output_dim, bias=bias)
        self.linear.weight.data = torch.zeros_like(self.linear.weight.data)

        min_dim = int(min(self.input_dim, self.output_dim))
        self.linear.weight.data[:min_dim, :min_dim] = torch.eye(min_dim)

    def forward(self, x):
        return self.linear(x)
