import torch
import torch.nn as nn


class CE(nn.Module):
    def __init__(self, norm_layer, image_size, input_nc, latent_dim=512):
        super(CE, self).__init__()

        self.norm_layer = norm_layer
        self.image_size = image_size
        self.input_nc = input_nc
        self.latent_dim = latent_dim

        latent_size = int(image_size/32)
        longsize = 512*latent_size**2
        self.longsize = longsize

        activation = nn.ReLU()
        padding_type = 'reflect'
        norm_layer = nn.BatchNorm

    def Conv2D_Block(in_channels, out_channels, kernel_size, padding, stride):
        return nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=kernel_size,
                                       stride=stride,
                                       padding=padding),
                             nn.BatchNorm2d(out_channels),
                             nn.LeakyReLU
                             )

    def ConvTrans2D_Block(in_channels, out_channels, kernel_size, padding, stride):
        return nn.Sequential(nn.ConvTranspose2d(in_channels=in_channels,
                                                out_channels=out_channels,
                                                kernel_size=kernel_size,
                                                stride=stride,
                                                padding=padding),
                             nn.BatchNorm2d(out_channels),
                             nn.LeakyReLU
                             )

    def forward(self, x):
