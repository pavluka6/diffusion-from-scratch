import torch.nn as nn
import torch
import math

class Block(nn.Module):
    def __init__(self, in_c, out_c, up = False):
        super().init__()
        self.bnorm = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()
        # TODO add Time Embeddings
        if up:
            self.conv1 = nn.Conv2d(2*in_c, out_c, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_c, out_c, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_c, out_c, 4, 2, 1)

        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)

    def forward(self, x, t):
        h = self.bnorm(self.relu(self.conv1(x)))
        # TODO add Time Embeddings
        h = self.bnorm(self.relu(self.conv2d(h)))
        return self.transform(h)


class Unet(nn.Module):
    def __init__(self, image_size = 480, te_dim = 32):
        super(Unet, self).__init__()
        self.image_size = image_size
        image_channels = 3
        down_channels = (64, 128, 256, 512, 1024)
        up_channels = (1024, 512, 256, 128, 64)
        self.te_dim = 32

        # Init self.layers and add first Conv2D
        self.layers = nn.ModuleList(nn.Conv2d(image_channels,down_channels[0]))

        # Init downsample channels
        for i in range(len(down_channels)-1):
            self.layers.append(Block(down_channels[i],down_channels[i+1]))

        # Init upsample channels
        for i in range(len(up_channels)-1):
            self.layers.append(Block(up_channels[i],up_channels[i+1], up=True))

        # Add output layer
        self.layers.append(nn.Conv2d(up_channels[-1], 1))

    def forward(self, x, t):
        # TODO add timestamp
        t = 0
        x = self.layers[0](x) # Conv0
        residuals = []
        for i in range(1, len(self.down_channels)):
            x = self.layers[i](x)
            residuals.append(x)

        for i in range(len(self.down_channels), len(self.down_channels) + len(self.up_channels)):
            residual = residuals.pop()
            x = torch.cat((x, residual), dim=1)
            x = self.layers[i](x)

        return self.layers[-1](x)
