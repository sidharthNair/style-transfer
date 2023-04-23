import torch
import torch.nn as nn

from blocks import ConvBlock, ResidualBlock, UpsampleBlock

# Implementation of Style Transfer Architecture as described in
# https://arxiv.org/pdf/1603.08155.pdf and https://cs.stanford.edu/people/jcjohns/papers/fast-style/fast-style-supp.pdf

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        channels = [3, 32, 64, 128]
        kernel_sizes = [9, 3, 3]
        strides = [1, 2, 2]
        num_residual_filters = 5

        layers = []
        for i in range(len(channels) - 1):
            layers.append(ConvBlock(
                in_channels=channels[i],
                out_channels=channels[i+1],
                kernel_size=kernel_sizes[i],
                stride=strides[i],
                batch_norm=True,
                activation='relu'
            ))

        for i in range(num_residual_filters):
            layers.append(ResidualBlock(
                num_channels=channels[len(channels) - 1],
                kernel_size=3,
                stride=1
            ))

        channels.reverse()
        kernel_sizes.reverse()
        strides.reverse()

        for i in range(len(channels) - 1):
            layers.append(UpsampleBlock(
                in_channels=channels[i],
                out_channels=channels[i+1],
                kernel_size=kernel_sizes[i],
                upscale_factor=strides[i],
                batch_norm=True if (i != len(strides) - 1) else False,
                activation='relu' if (i != len(strides) - 1) else None
            ))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

if __name__ == '__main__':
    net = Net()
    x = torch.randn((5, 3, 128, 128))
    y = net(x)
    print(x.shape, y.shape)