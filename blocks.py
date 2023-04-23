import torch.nn as nn

# Implementation of Style Transfer Architecture as described in
# https://arxiv.org/pdf/1603.08155.pdf and https://cs.stanford.edu/people/jcjohns/papers/fast-style/fast-style-supp.pdf
# The following classes define the building blocks for the generator and discriminator networks

class ConvBlock(nn.Module):
    # Convolutional layer with optional batch normalization and activation function
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, batch_norm=False, activation=False):
        super().__init__()

        layers = []
        layers.append(nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size // 2),
            bias=(not batch_norm)
        ))

        if batch_norm:
            layers.append(nn.BatchNorm2d(num_features=out_channels))

        if activation:
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            else:
                raise RuntimeError(
                    'Unsupported activation function:', activation)

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    # Two convolutional layers with a skip connection
    def __init__(self, num_channels, kernel_size, stride):
        super().__init__()
        self.conv_block1 = ConvBlock(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=kernel_size,
            stride=stride,
            batch_norm=True,
            activation='relu'
        )
        self.conv_block2 = ConvBlock(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=kernel_size,
            stride=stride,
            batch_norm=True,
            activation=None
        )

    def forward(self, x):
        y = self.conv_block1(x)
        y = self.conv_block2(y)
        return x + y # Skip connection

class UpsampleBlock(nn.Module):
    # Upscaling block with convolutional layer
    def __init__(self, in_channels, out_channels, kernel_size, upscale_factor, batch_norm=False, activation=None):
        super().__init__()
        self.upscale_factor = upscale_factor
        self.conv_block = ConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            batch_norm=batch_norm,
            activation=activation
        )

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.upscale_factor, mode='nearest')
        return self.conv_block(x)