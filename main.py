import torch
from torch import nn, optim
from torch.nn import functional as F
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
from VQVAE import VQVAE
from torchvision.datasets import MNIST

matplotlib.use("Qt5Agg")

CKPT_PATH = "vqvae.ckpt"

data = MNIST(root="Data", train=True, download=True).data

config = {
    "input_size": (28, 28),
    "hidden_size": (4, 4),
    "n_channels": [1, 4, 8, 16],
    "num_layers_per_channel_size": 4,
    "codebook_size": 5,
    "commitment_loss_scaler": 0.25,
    "vqvae_loss_scaler": 0.25,
}

vqvae = VQVAE(config)
optimizer = optim.Adam(vqvae.parameters(), lr=1e-3)

# load model
if os.path.isfile(CKPT_PATH):
    vqvae.load_state_dict(torch.load(CKPT_PATH))

if torch.cuda.is_available():
    print("Using Cuda")
    vqvae = vqvae.cuda()

mean = torch.mean(data.float())
std = torch.std(data.float())
data_norm = data.float() - mean
data_norm = data_norm / std
data_norm = data_norm[:100, :, :].unsqueeze(1)

_, axes = plt.subplots(10, 21, figsize=(10, 28))
axes = np.array(axes)

for ax in axes.flatten():
    ax.axis("off")

for ax, img in zip(axes[:, :10].flatten(), data[:100]):
    ax.imshow(img, cmap="gray")

plt.pause(0.01)

for i in range(100000):
    print(i, end="\r")
    output, vqvae_loss = vqvae(data_norm)
    loss = F.mse_loss(output, data_norm) + config["vqvae_loss_scaler"] * vqvae_loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if i % 100 == 0:
        for ax, img in zip(axes[:, 11:].flatten(), output[:100]):
            img = img * std + mean
            img = img.detach().cpu().squeeze().numpy()
            img = img.clip(0, 255)
            ax.imshow(img, cmap="gray")

        plt.pause(0.01)

        # save model
        torch.save(vqvae.state_dict(), CKPT_PATH)
