import torch.nn as nn
import math

class Block(nn.Module):
    def __init__(self, in_c, out_c, up = False):
        pass
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