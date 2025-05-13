import os
import sys
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import btk

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import random_split, Dataset, TensorDataset, DataLoader
from torchvision import transforms

name = "delta_1e-6_num_channel_6"

class SimpleBlendDataset(Dataset):
    def __init__(self, base_path, num_files):
        self.file_paths = []
        
        # Find all valid files
        for i in range(num_files):
            path = os.path.join(base_path, f'blend_{i}.hdf5')
            if os.path.exists(path):
                self.file_paths.append(path)
        
        if not self.file_paths:
            raise ValueError(f"No valid files found in {base_path}")
        
        # Pre-calculate total samples (1000 per file)
        self.total_samples = len(self.file_paths) * 1000

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        file_idx = idx // 1000  # Which file to use
        sample_idx = idx % 1000  # Which sample in the file
        
        with h5py.File(self.file_paths[file_idx], 'r') as f:
            # Get image (6, 45, 45)
            image = f['blend_images'][sample_idx].astype(np.float32)

            # Normalization
            min_ = image.min(axis=(1,2), keepdims=True)
            max_ = image.max(axis=(1,2), keepdims=True)

            denom = max_ - min_

            image = (image - min_)/denom
            
            # Get attributes from catalog_list
            catalog_data = f[f'catalog_list/{sample_idx}'][0] 
            redshift = catalog_data['redshift']
            g1 = catalog_data['g1']
            g2 = catalog_data['g2']
            r_ab = catalog_data['r_ab']
        
        # Convert to PyTorch tensors
        image_tensor = torch.from_numpy(image).float()
        image_tensor = image_tensor.unsqueeze(0)
        attributes = torch.tensor([redshift, g1, g2, r_ab], dtype=torch.float32)
        
        return image_tensor, attributes

# Create dataset
SCRATCH = os.getenv("ALL_CCFRSCRATCH")  #
train_path = os.path.join(SCRATCH, "deblending/isolated/isolated_training_9_arcs")
dataset = SimpleBlendDataset(base_path=train_path, num_files=200)

# Data loader
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=8,  # Parallel loading
    pin_memory=True if torch.cuda.is_available() else False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=128,
    shuffle=False,
    num_workers=8,
    pin_memory=True if torch.cuda.is_available() else False
)

# Device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Model
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        filters = [32, 64, 128, 256]
        kernel_size = [3, 3, 3, 3]
        num_channels = 6
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(num_channels, filters[0], kernel_size=kernel_size[0], stride=2, padding=1)
        self.conv2 = nn.Conv2d(filters[0], filters[1], kernel_size=kernel_size[1], stride=2, padding=1)
        self.conv3 = nn.Conv2d(filters[1], filters[2], kernel_size=kernel_size[2], stride=2, padding=1)
        self.conv4 = nn.Conv2d(filters[2], filters[3], kernel_size=kernel_size[3], stride=2, padding=1)

        # Batch normalization layers
        self.bn1 = nn.BatchNorm2d(filters[0])
        self.bn2 = nn.BatchNorm2d(filters[1])
        self.bn3 = nn.BatchNorm2d(filters[2])
        self.bn4 = nn.BatchNorm2d(filters[3])

        # PReLU activation layers
        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()
        self.prelu3 = nn.PReLU()
        self.prelu4 = nn.PReLU()
        self.prelu_fc1 = nn.PReLU()

        # Fully connected layers
        self.fc1 = nn.Linear(3 * 3 * filters[3], 512) # 2304 -> 512
        self.fc2 = nn.Linear(512, 128) # 512 -> 128
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

    def forward(self, x):
        x = x.to(device)

        # Apply Conv -> PReLU -> BatchNorm 
        x = self.bn1(self.prelu1(self.conv1(x)))  # 45x45 → 23x23
        x = self.bn2(self.prelu2(self.conv2(x)))  # 23x23 → 12x12
        x = self.bn3(self.prelu3(self.conv3(x)))  # 12x12 → 6x6
        x = self.bn4(self.prelu4(self.conv4(x)))  # 6x6 → 3x3
        
        x = torch.flatten(x, start_dim=1)  # Flatten to [batch_size, 3x3x256]
        x = self.prelu_fc1(self.fc1(x))    
        x = self.fc2(x)                    # Fully connected layer
    
        mu = self.fc_mu(x)                 # Mean of latent distribution
        logvar = self.fc_logvar(x)         # Log variance of latent distribution

        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        filters = [32, 64, 128, 256]
        kernel_size = [3, 3, 3, 3]
        num_channels = 6

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.PReLU(),
            nn.Linear(128, 512),
            nn.PReLU(),
            nn.Linear(512, 3 * 3 * filters[3]),  # Output size: 2304
            nn.PReLU(),
        )

        # Unflatten 
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(filters[3], 3, 3))

        # Transposed convolutional layers
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(filters[3], filters[2], kernel_size=kernel_size[0], stride=2, padding=1, output_padding=1),
            nn.PReLU(),
            nn.ConvTranspose2d(filters[2], filters[1], kernel_size=kernel_size[1], stride=2, padding=1, output_padding=1),
            nn.PReLU(),
            nn.ConvTranspose2d(filters[1], filters[0], kernel_size=kernel_size[2], stride=2, padding=1, output_padding=1),
            nn.PReLU(),
            nn.ConvTranspose2d(filters[0], num_channels, kernel_size=kernel_size[3], stride=2, padding=1, output_padding=1),
            nn.ReLU()  # Output in range [0, +\infty]
        )

    def forward(self, x):
        x = x.to(device)
        x = self.fc(x)               # Fully connected layers
        x = self.unflatten(x)        # Reshape to [batch_size, 256, 3, 3]
        x = self.deconv(x)           # Upsample to [batch_size, 1, 45, 45]
        x = x + 1e-6
        
        x = cropping(x, output_size=45)
        return x

def cropping(x, output_size=45):
    _, _, h, w = x.shape
    top = (h - output_size) // 2
    left = (w - output_size) // 2
    return x[:, :, top:top+output_size, left:left+output_size]

def reparameterization(mu, logvar):
        sigma = torch.exp(0.5 * logvar)    # Standard deviation
        N = torch.distributions.Normal(0, 1)
        # Reparameterization trick
        z = mu + sigma * N.sample(mu.shape).to(device)

        # KL divergence
        kl = 0.5 * (sigma**2 + mu**2 - logvar - 1).mean()
        
        return z, kl

class VAE(nn.Module):
    def __init__(self, latent_dims):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z, kl = reparameterization(mu, logvar)

        return self.decoder(z), kl

## training function
def loss_function(x, x_hat, kl):
    mse = ((x-x_hat)**2).mean()
    return mse + kl

def train_epoch(vae, device, dataloader, optimizer):
    vae.train()
    train_loss = 0.0

    for x, _ in dataloader:
        x = x.to(device)
        
        x_hat, kl = vae(x)
        loss = loss_function(x, x_hat, kl)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print('\t Partial train loss (single batch): %f' % (loss.item()))
        train_loss += loss.item()

    return train_loss/len(dataloader.dataset)

def test_epoch(vae, device, dataloader):
    vae.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            x_hat, kl = vae(x)
            loss = loss_function(x, x_hat, kl)
            val_loss += loss.item()

    return val_loss / len(dataloader.dataset)

def save_checkpoint(epoch, vae, optimizer, train_loss, val_loss, checkpoint_path=f"checkpoints/vae_checkpoint_{name}.pth"):
    torch.save({
        'epoch': epoch,
        'model_state_dict': vae.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
    }, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch + 1}")

def save_model(epoch, vae, optimizer, train_loss, val_loss, model_path=f"models/vae_model_{name}.pth"):
    torch.save({
        'epoch': epoch,
        'model_state_dict': vae.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
    }, model_path)
    print(f"Model saved at epoch {epoch + 1}")

def plot_loss_function(num_epochs, train_loss, val_loss):
    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    epochs = np.arange(1, num_epochs + 1)

    # Train Loss Plot
    ax[0].plot(epochs, train_loss, label="Train Loss", color="royalblue", linestyle="-", marker="o", markersize=4)
    ax[0].set_title("Training Loss", fontsize=12, fontweight="bold")
    ax[0].set_ylabel("Loss", fontsize=11)
    ax[0].set_yscale("log")  # Set log scale
    ax[0].grid(True, linestyle="--", alpha=0.6)
    ax[0].legend()

    # Validation Loss Plot
    ax[1].plot(epochs, val_loss, label="Validation Loss", color="crimson", linestyle="-", marker="s", markersize=4)
    ax[1].set_title("Validation Loss", fontsize=12, fontweight="bold")
    ax[1].set_xlabel("Epochs", fontsize=11)
    ax[1].set_ylabel("Loss", fontsize=11)
    ax[1].set_yscale("log")  # Set log scale
    ax[1].grid(True, linestyle="--", alpha=0.6)
    ax[1].legend()

    plt.tight_layout()
    plt.savefig(f"figures/loss_plots_{name}.png")
    print("Saved loss function plots")

def generated_images():
    num_samples = 100

    latent_samples = torch.randn(num_samples, 32)  # Sample random points from N(0,1)

    # Get the device of the model
    device = next(vae.parameters()).device

    # Move latent samples to the same device as the model
    latent_samples = latent_samples.to(device)

    # Decode to get images
    with torch.no_grad():
        generated_images = vae.decoder(latent_samples).cpu()    

    # Plot the generated images
    fig, axes = plt.subplots(10, 10, figsize=(15, 15))
    for i in range(10):
        for j in range(10):
            k = 10*i + j
            axes[i,j].imshow(generated_images[k].squeeze(), cmap="gray")
            axes[i,j].axis("off")
    plt.savefig(f"figures/generated_images_{name}.png")
    print("Saved generated images")

# --- Initialize Model ---
torch.manual_seed(12)
latent_dims = 32
vae = VAE(latent_dims=latent_dims)

optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3, weight_decay=1e-5)

vae.to(device)

# --- Training Loop ---
num_epochs = 500
train_loss = np.empty(num_epochs)
val_loss = np.empty(num_epochs)

for epoch in range(num_epochs):
    train_loss[epoch] = train_epoch(vae, device, train_loader, optimizer)
    val_loss[epoch] = test_epoch(vae, device, val_loader)
    # Save checkpoint every 50 epochs
    if (epoch + 1) % 50 == 0:
        save_checkpoint(epoch, vae, optimizer, train_loss[:epoch+1], val_loss[:epoch+1])

save_model(epoch, vae, optimizer, train_loss, val_loss)
plot_loss_function(num_epochs, train_loss, val_loss)
generated_images()
