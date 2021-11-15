import torch
import torch.nn as nn
from Resnetblock import ResnetBlock


class DecoderGenerator_feature_Res(nn.Module):
    def __init__(self, norm_layer, image_size, output_nc, latent_dim=512):
        super(DecoderGenerator_feature_Res, self).__init__()

        latent_size = int(image_size/32)
        self.latent_size = latent_size
        longsize = 512*latent_size*latent_size

        activation = nn.ReLU()
        padding_type = 'reflect'
        norm_layer = nn.BatchNorm

        self.conv = nn.Sequential(*layers_list)

        for m in self.modules():
            weights_init_normal(m)

    def execute(self, ten):
        # print("in DecoderGenerator, print some shape ")
        # print(ten.size())
        ten = self.fc(ten)
        # print(ten.size())
        ten = jt.reshape(
            ten, (ten.size()[0], 512, self.latent_size, self.latent_size))
        # print(ten.size())
        ten = self.conv(ten)

        return ten
