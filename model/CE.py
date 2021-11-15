import torch
import torch.nn as nn


class EncoderGen_Res(nn.Module):
    def __init__(self, norm_layer, image_size, input_nc, latent_dim=512):
        super(EncoderGen_Res, self).__init__()

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

        # conv1
        self.conv1_1 = Conv2D_Block(self.input_nc, 32, 4, 1, 2)
        self.conv1_2 = ResnetBlock(
            32, padding_type=padding_type, activation=activation, norm_layer=norm_layer)

        # conv2
        self.conv2_1 = Conv2D_Block(32, 64, 4, 1, 2)
        self.conv2_2 = ResnetBlock(
            32, padding_type=padding_type, activation=activation, norm_layer=norm_layer)

        # conv3
        self.conv3_1 = Conv2D_Block(64, 128, 4, 1, 2)
        self.conv3_2 = ResnetBlock(
            32, padding_type=padding_type, activation=activation, norm_layer=norm_layer)

        # conv4
        self.conv4_1 = Conv2D_Block(128, 256, 4, 1, 2)
        self.conv4_2 = ResnetBlock(
            32, padding_type=padding_type, activation=activation, norm_layer=norm_layer)

        # conv5
        self.conv5_1 = Conv2D_Block(256, 512, 4, 1, 2)
        self.conv5_2 = ResnetBlock(
            32, padding_type=padding_type, activation=activation, norm_layer=norm_layer)

        # Fully connected layer
        self.fc = nn.Linear(in_features=longsize, out_features=latent_dim)

    def forward(self, x):
        h = self.conv1_1(x)
        h = self.conv1_2(h)

        h = self.conv2_1(h)
        h = self.conv2_2(h)

        h = self.conv3_1(h)
        h = self.conv3_2(h)

        h = self.conv4_1(h)
        h = self.conv4_2(h)

        h = self.conv5_1(h)
        h = self.conv5_2(h)

        return self.fc(h)


class DecoderGen_Res(nn.Moudule):
    def __init__(self, norm_layer, image_size, output_nc, latent_dim=512):
        super(DecoderGen_Res, self).__init__()
        self.norm_layer = norm_layer
        self.image_size = image_size
        self.input_nc = output_nc
        self.latent_dim = latent_dim

        latent_size = int(image_size/32)
        self.latent_size = latent_size
        longsize = 512*latent_size*latent_size

        activation = nn.ReLU()
        padding_type = 'reflect'
        norm_layer = nn.BatchNorm

        def ConvTrans2D_Block(in_channels, out_channels, kernel_size, padding, stride):
            return nn.Sequential(nn.ConvTranspose2d(in_channels=in_channels,
                                                    out_channels=out_channels,
                                                    kernel_size=kernel_size,
                                                    stride=stride,
                                                    padding=padding),
                                 nn.BatchNorm2d(out_channels),
                                 nn.LeakyReLU
                                 )

        def Conv2D_Block(in_channels, out_channels, kernel_size, padding, stride):
            return nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                           out_channels=out_channels,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=padding),
                                 nn.BatchNorm2d(out_channels),
                                 nn.LeakyReLU
                                 )
        # fc
        self.fc = nn.Linear(in_features=latent_dim, out_features=longsize)

        # convTrans1
        self.convtr1_1 = ResnetBlock(
            512, padding_type=padding_type, activation=activation, norm_layer=norm_layer)
        self.convtr1_2 = ConvTrans2D_Block(512, 256, 4, 1, 2)

        # convTrans2
        self.convtr2_1 = ResnetBlock(
            256, padding_type=padding_type, activation=activation, norm_layer=norm_layer)
        self.convtr2_2 = ConvTrans2D_Block(256, 128, 4, 1, 2)

        # convTrans3
        self.convtr3_1 = ResnetBlock(
            128, padding_type=padding_type, activation=activation, norm_layer=norm_layer)
        self.convtr3_2 = ConvTrans2D_Block(128, 64, 4, 1, 2)

        # convTrans4
        self.convtr4_1 = ResnetBlock(
            64, padding_type=padding_type, activation=activation, norm_layer=norm_layer)
        self.convtr4_2 = ConvTrans2D_Block(64, 32, 4, 1, 2)

        # convTrans5
        self.convtr5_1 = ResnetBlock(
            32, padding_type=padding_type, activation=activation, norm_layer=norm_layer)
        self.convtr5_2 = ConvTrans2D_Block(32, 32, 4, 1, 2)

        # convTrans6
        self.convtr6_1 = ResnetBlock(
            32, padding_type=padding_type, activation=activation, norm_layer=norm_layer)
        self.convtr6_2 = nn.ReflectionPad2d(2)
        self.convtr6_3 = Conv2D_Block(32, output_nc, kernel_size=5, padding=0)

    def forward(self, x):
        h = self.convtr1_1(x)
        h = self.convtr1_2(h)

        h = self.convtr2_1(h)
        h = self.convtr2_2(h)

        h = self.convtr3_1(h)
        h = self.convtr3_2(h)

        h = self.convtr4_1(h)
        h = self.convtr4_2(h)

        h = self.convtr5_1(h)
        h = self.convtr5_2(h)

        h = self.convtr6_1(h)
        h = self.convtr6_2(h)
        return self.convtr6_3(h)


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(
            dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if (padding_type == 'reflect'):
            conv_block += [nn.ReflectionPad2d(1)]
        elif (padding_type == 'replicate'):
            conv_block += [nn.ReplicationPad2d(1)]
        elif (padding_type == 'zero'):
            p = 1
        else:
            raise NotImplementedError(
                ('padding [%s] is not implemented' % padding_type))
        conv_block += [nn.Conv(dim, dim, 3, padding=p),
                       norm_layer(dim), activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if (padding_type == 'reflect'):
            conv_block += [nn.ReflectionPad2d(1)]
        elif (padding_type == 'replicate'):
            conv_block += [nn.ReplicationPad2d(1)]
        elif (padding_type == 'zero'):
            p = 1
        else:
            raise NotImplementedError(
                ('padding [%s] is not implemented' % padding_type))
        conv_block += [nn.Conv(dim, dim, 3, padding=p), norm_layer(dim)]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = (x + self.conv_block(x))
        return out
