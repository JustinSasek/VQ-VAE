import os
import numpy as np
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Qt5Agg")

from VQVAE import VQVAE

CKPT_PATH = "vqvae.ckpt"

# Prep model
config = {
    "input_size": (28, 28),
    "hidden_size": (4, 4),
    "n_channels": [1, 4, 8, 16],
    "num_layers_per_channel_size": 4,
    "codebook_size": 5,
    "commitment_loss_scaler": 1,
    "vqvae_loss_scaler": 0.25,
}

vqvae = VQVAE(config)
optimizer = optim.Adam(vqvae.parameters(), lr=5e-4)

# Load model
if os.path.isfile(CKPT_PATH):
    vqvae.load_state_dict(torch.load(CKPT_PATH))

if torch.cuda.is_available():
    print("Using Cuda")
    vqvae = vqvae.cuda()

# Load data
data = MNIST(root="Data", train=True, download=True).data

# Normalize data and make dataloader
mean = torch.mean(data.float())
std = torch.std(data.float())
data_norm = data.float() - mean
data_norm = data_norm / std
data_norm = data_norm.unsqueeze(1)
data_loader = DataLoader(data_norm, batch_size=128, shuffle=True)

# Create visualization
_, axes = plt.subplots(10, 21, figsize=(10, 28))
axes = np.array(axes)

for ax in axes.flatten():
    ax.axis("off")

for ax, img in zip(axes[:, :10].flatten(), data[:100]):
    ax.imshow(img, cmap="gray")

plt.pause(0.01)

i = 0
while 1:
    n_batches = data_loader.dataset.shape[0] // data_loader.batch_size
    with tqdm(
        total=n_batches,
        desc="Training",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}",
        colour="green",
    ) as pbar:
        for batch in data_loader:
            output, vqvae_loss = vqvae(batch)
            loss = F.mse_loss(output, batch) + config["vqvae_loss_scaler"] * vqvae_loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if i % 100 == 0:
                for ax, img in zip(axes[:, :10].flatten(), batch[:100]):
                    img = img * std + mean
                    img = img.detach().cpu().squeeze().numpy()
                    img = img.clip(0, 255)
                    ax.imshow(img, cmap="gray")

                for ax, img in zip(axes[:, 11:].flatten(), output[:100]):
                    img = img * std + mean
                    img = img.detach().cpu().squeeze().numpy()
                    img = img.clip(0, 255)
                    ax.imshow(img, cmap="gray")

                plt.pause(0.01)

                # save model
                torch.save(vqvae.state_dict(), CKPT_PATH)
            i += 1
            pbar.update(1)
