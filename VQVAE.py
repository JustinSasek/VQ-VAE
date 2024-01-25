import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision.transforms import CenterCrop as crop
from typing import Tuple, List


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, n_layers: int):
        super().__init__()
        mods = []
        for i in range(n_layers):
            layer_in_ch = in_ch if i == 0 else max(in_ch, out_ch)
            layer_out_ch = out_ch if i == n_layers - 1 else max(in_ch, out_ch)
            mods.append(
                nn.Sequential(
                    nn.Conv2d(layer_in_ch, layer_out_ch, 5, stride=1, padding="same"),
                    nn.BatchNorm2d(layer_out_ch),
                    nn.Tanh(),
                )
            )

        self.mods = nn.Sequential(*mods)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mods(x)
        return x


class Encoder(nn.Module):
    def __init__(self, config: dict):
        super().__init__()

        self.config = config
        mods: List[nn.Module] = []

        for i in range(len(config["n_channels"]) - 1):
            mods.append(
                ConvBlock(
                    config["n_channels"][i],
                    config["n_channels"][i + 1],
                    config["num_layers_per_channel_size"],
                )
            )
            mods.append(nn.MaxPool2d(2, 2))

        mods.append(
            ConvBlock(
                config["n_channels"][-1],
                config["n_channels"][-1],
                config["num_layers_per_channel_size"],
            )
        )

        self.mods = nn.Sequential(*mods)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ideal_size = max(self.config["input_size"])
        ideal_size = 2 ** (ideal_size - 1).bit_length()
        left_pad = (ideal_size - self.config["input_size"][1]) // 2
        right_pad = ideal_size - self.config["input_size"][1] - left_pad
        top_pad = (ideal_size - self.config["input_size"][0]) // 2
        bottom_pad = ideal_size - self.config["input_size"][0] - top_pad

        x = F.pad(
            x, pad=(left_pad, right_pad, top_pad, bottom_pad), mode="constant", value=0
        )
        x = self.mods(x)
        return x


class Decoder(nn.Module):
    def __init__(self, config: dict):
        super().__init__()

        mods: List[nn.Module] = []

        n_channels_r = config["n_channels"][::-1]

        for i in range(len(n_channels_r) - 1):
            mods.append(
                ConvBlock(
                    n_channels_r[i],
                    n_channels_r[i + 1],
                    config["num_layers_per_channel_size"],
                )
            )
            mods.append(nn.Upsample(scale_factor=2))

        mods.append(
            ConvBlock(
                n_channels_r[-1],
                n_channels_r[-1],
                config["num_layers_per_channel_size"],
            )
        )

        self.mods = nn.Sequential(*mods)

        self.crop = crop(size=config["input_size"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mods(x)
        x = self.crop(x)
        return x


class VQVAE(nn.Module):
    def __init__(self, config: dict):
        super().__init__()

        self.config = config
        self.latent_size = config["n_channels"][-1]

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.codebook = nn.Embedding(config["codebook_size"], self.latent_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.encoder(x)

        # lookup
        x_flat = x.view(size=(-1, self.latent_size))
        dists = torch.cdist(x_flat, self.codebook.weight, 2)
        x_vq = self.codebook(torch.argmin(dists, dim=1))

        x_vq = x_vq.view(size=(-1, self.latent_size, *self.config["hidden_size"]))

        x_vq = x + (x_vq - x).detach()

        # find loss
        codebook_loss = torch.mean((x_vq - x.detach()) ** 2)
        commitment_loss = torch.mean((x_vq.detach() - x) ** 2)
        beta = 1
        loss = codebook_loss + beta * commitment_loss

        output = self.decoder(x_vq)
        return output, loss
