from vae import VAE, SimpleBlendDataset, LatentRegressor
from vae import generated_images, save_checkpoint, train_epoch, val_epoch, train_epoch_reg, val_epoch_reg
from torch.utils.data import Subset
import torch
from torch.utils.data import DataLoader, random_split
import numpy as np
import os

# Setup
latent_dims = 32
name = "v1"
num_epochs_vae = 50
num_epochs_regressor = 200

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Models
vae = VAE(latent_dim=latent_dims, beta=1e-3).to(device)
regressor = LatentRegressor(latent_dim=latent_dims).to(device)

# Optimizers
optimizer_vae = torch.optim.Adam(vae.parameters(), lr=1e-4, weight_decay=1e-5)
optimizer_reg = torch.optim.Adam(regressor.parameters(), lr=1e-3)

# Load data
SCRATCH = os.getenv("ALL_CCFRSCRATCH")
train_path = os.path.join(SCRATCH, "deblending/isolated/isolated_training_9_arcs")
dataset = SimpleBlendDataset(base_path=train_path, num_files=200) # Number of images in dataset = num_files * 1000

# Choosen e > 0.1
if not os.path.exists("e_values.npy"):
    # If the file doesn't exist, calculate e, e1, e2
    e = np.empty(len(dataset))
    e1 = np.empty(len(dataset))
    e2 = np.empty(len(dataset))
    
    for idx in range(len(dataset)):
        _, attrs = dataset[idx]
        e1[idx] = attrs[1]
        e2[idx] = attrs[2]
        e[idx] = np.sqrt(e1[idx]**2 + e2[idx]**2)
    
    # Save the arrays to .npy files
    np.save("e_values.npy", e)
    np.save("e1_values.npy", e1)
    np.save("e2_values.npy", e2)
else:
    # If the file exists, load e_values.npy
    e = np.load("e_values.npy")

# Filter the dataset based on e > 0.1
index = np.where(e > 0.1)[0]
dataset = Subset(dataset, index)

# Dividing 90% for training and 10% for valuation
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

generator = torch.Generator().manual_seed(42)
train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=8, # Parallel loading
    pin_memory=True if torch.cuda.is_available() else False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=128,
    shuffle=False,
    num_workers=8,
    pin_memory=True if torch.cuda.is_available() else False
)

# Training loop
train_loss = np.empty(num_epochs_vae)
val_loss = np.empty(num_epochs_vae)
train_mse = np.empty(num_epochs_vae)
val_mse = np.empty(num_epochs_vae)
train_kl = np.empty(num_epochs_vae)
val_kl = np.empty(num_epochs_vae)

train_loss_reg = np.empty(num_epochs_regressor)
val_loss_reg = np.empty(num_epochs_regressor)

for epoch in range(num_epochs_vae):
    train_loss[epoch], train_mse[epoch], train_kl[epoch] = train_epoch(vae, device, train_loader, optimizer_vae)
    val_loss[epoch], val_mse[epoch], val_kl[epoch] = val_epoch(vae, device, val_loader)
    if (epoch + 1) % 10 == 0 or (epoch + 1) == num_epochs_vae: # Save checkpoint every 10 epoch
        save_checkpoint(epoch, vae, optimizer_vae, train_loss[:epoch+1], train_mse[:epoch+1], train_kl[:epoch+1], val_loss[:epoch+1], val_mse[:epoch+1], val_kl[:epoch+1], f"checkpoints/vae_checkpoint_{name}.pth")

for epoch in range(num_epochs_regressor):
    train_loss_reg[epoch] = train_epoch_reg(vae, regressor, device, train_loader, optimizer_reg)
    val_loss_reg[epoch] = val_epoch_reg(vae, regressor, device, val_loader)
    if (epoch + 1) % 10 == 0 or (epoch + 1) == num_epochs_regressor:
        save_checkpoint(epoch, vae, optimizer_reg, train_loss[:epoch+1], None, None, val_loss[:epoch+1], None, None, f"checkpoints/reg_checkpoint_{name}.pth")

# # Save and plot
# save_model(epoch, vae, optimizer, train_loss, train_mse, train_kl, val_loss, val_mse, val_kl, f"models/vae_model_{name}.pth")
# plot_loss_function(num_epochs, train_loss, val_loss, name)
# generated_images(vae, latent_dims, name)
