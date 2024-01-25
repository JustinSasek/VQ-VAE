import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision.transforms import CenterCrop as crop
from typing import Tuple

SIZES = ((28, 28, 1), (16, 16, 4), (8, 8, 8), (4, 4, 16))
NUM_LAYERS_PER_CHANNEL_SIZE = 4

# class Conv_Block(nn.module):
#     def __init__(self, in_ch:int, out_ch:int, n_layers:int):
#         super().__init__()
#         self.mods = []
#         for i in range(n_layers):
            
#             self.mods.append(nn.Sequential(
#                 nn.Conv2d(in_ch, out_ch, 5, stride=1, padding="same"),
#                 nn.BatchNorm2d(out_ch),
#                 nn.Tanh(),
#             ))
#             in_ch = out_ch


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        sizes = SIZES
        num_layers_per_channel_size = NUM_LAYERS_PER_CHANNEL_SIZE
        
        nn_modules = []
        
        for i in range(len(sizes)):
            for j in range(num_layers_per_channel_size):
                start_n_channels = sizes[i][2]
                end_n_channels = sizes[i][2] if i == len(sizes) - 1 else sizes[i + 1][2]
                
                if j != 0:
                    start_n_channels = end_n_channels
                    
                nn_modules.append(nn.Sequential(
                    nn.Conv2d(start_n_channels, end_n_channels, 5, stride=1, padding="same"),
                    nn.BatchNorm2d(end_n_channels),
                    nn.Tanh(),
                ))
            if i != len(sizes) - 1:
                nn_modules.append(nn.MaxPool2d(2, 2))
            
        self.nn_modules = nn.Sequential(*nn_modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, pad=(2, 2, 2, 2), mode="constant", value=0)
        x = self.nn_modules(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        sizes = SIZES[::-1]
        num_layers_per_channel_size = NUM_LAYERS_PER_CHANNEL_SIZE
        
        nn_modules = []
        
        for i in range(len(sizes) - 1):
            for j in range(num_layers_per_channel_size):
                start_n_channels = sizes[i][2]
                end_n_channels = sizes[i][2] if i == len(sizes) - 1 else sizes[i + 1][2]
                
                if j != len(sizes) - 1:
                    end_n_channels = start_n_channels
                
                nn_modules.append(nn.Sequential(
                    nn.Conv2d(start_n_channels, end_n_channels, 5, stride=1, padding="same"),
                    nn.BatchNorm2d(end_n_channels),
                    nn.Tanh(),
                ))
            if i != len(sizes) - 1:
                nn_modules.append(nn.Upsample(scale_factor=2))
        
        self.nn_modules = nn.Sequential(*nn_modules)
        
        self.crop = crop(size=(sizes[-1][0], sizes[-1][1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.nn_modules(x)
        x = self.crop(x)
        return x


class VQVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.sizes = SIZES

        self.code_book_size = 5

        self.encoder = Encoder()
        self.decoder = Decoder()
        self.codebook = nn.Embedding(self.code_book_size, self.sizes[-1][2])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.encoder(x)

        # lookup
        x_flat = x.view(size=(-1, self.sizes[-1][2]))
        dists = torch.cdist(x_flat, self.codebook.weight, 2)
        x_vq = self.codebook(torch.argmin(dists, dim=1))
        
        x_vq = x_vq.view(size=(-1, self.sizes[-1][2], self.sizes[-1][0], self.sizes[-1][1]))
        
        x_vq = x + (x_vq - x).detach()

        # find loss
        codebook_loss = torch.mean((x_vq - x.detach()) ** 2)
        commitment_loss = torch.mean((x_vq.detach() - x) ** 2)
        beta = 1
        loss = codebook_loss + beta * commitment_loss

        output = self.decoder(x_vq)
        return output, loss
