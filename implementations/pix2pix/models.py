from typing import List, Any, Tuple

import torch.nn as nn
import torch.nn.functional as F
import torch


def weights_init_normal(m: 'nn.Module[torch.Tensor]') -> None:
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


##############################
#           U-NET
##############################


class UNetDown(nn.Module):  # type: ignore
    def __init__(self, in_size: int, out_size: int, normalize: bool=True, dropout: float=0.0) -> None:
        super(UNetDown, self).__init__()
        layers: List[Any] = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        return self.model(x)


class UNetUp(nn.Module):  # type: ignore
    def __init__(self, in_size: int, out_size: int, dropout: float=0.0) -> None:
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, skip_input: torch.Tensor) -> torch.Tensor:  # type: ignore
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x


class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(GeneratorUNet, self).__init__()

        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final(u7)


class UNet32x32(nn.Module):  # type: ignore
    def __init__(self, in_channels: int=3, out_channels: int=3, in_size: Tuple[int, int]=(32, 32), out_size: Tuple[int, int]=(32, 32), base_num_filters: int=8):
        super(UNet32x32, self).__init__()

        b = base_num_filters

        self.adjust_in = nn.Upsample(size=in_size)

        self.down1 = UNetDown(in_channels, b, normalize=False)
        self.down2 = UNetDown(b, 2*b)
        # self.down3 = UNetDown(2*b, 4*b, normalize=False)
        self.down3 = UNetDown(2*b, 4*b)
        self.down4 = UNetDown(4*b, 4*b, dropout=0.5, normalize=False)
        # self.down4 = UNetDown(4*b, 8*b, dropout=0.5)
        # self.down5 = UNetDown(8*b, 8*b, dropout=0.5)
        # self.down6 = UNetDown(8*b, 8*b, dropout=0.5)
        # self.down7 = UNetDown(8*b, 8*b, dropout=0.5)
        # self.down8 = UNetDown(8*b, 8*b, normalize=False, dropout=0.5)

        # self.up7 = UNetUp(8*b, 8*b, dropout=0.5)
        # self.up6 = UNetUp(2*8*b, 8*b, dropout=0.5)
        # self.up5 = UNetUp(2*8*b, 8*b, dropout=0.5)
        # self.up4 = UNetUp(2*8*b, 8*b, dropout=0.5)
        # self.up3 = UNetUp(2*8*b, 4*b)
        self.up3 = UNetUp(4*b, 4*b)
        self.up2 = UNetUp(2*4*b, 2*b)
        self.up1 = UNetUp(2*2*b, b)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            # nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Conv2d(2*base_num_filters, out_channels, 4, padding=1),
            nn.Tanh(),
        )

        self.adjust_out = nn.Upsample(size=out_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        # U-Net generator with skip connections from encoder to decoder
        inp = self.adjust_in(x)

        d1 = self.down1(inp)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        # d5 = self.down5(d4)
        # d6 = self.down6(d5)
        # d7 = self.down7(d6)
        # d8 = self.down8(d7)
        # u7 = self.up7(d8, d7)
        # u6 = self.up6(u7, d6)
        # u5 = self.up5(u6, d5)
        # u4 = self.up4(u5, d4)
        # u3 = self.up3(u4, d3)
        u3 = self.up3(d4, d3)
        u2 = self.up2(u3, d2)
        # u2 = self.up2(d3, d2)
        u1 = self.up1(u2, d1)
        u0 = self.final(u1)

        out = self.adjust_out(u0)
        return out


##############################
#        Discriminator
##############################


class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels * 2, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)
