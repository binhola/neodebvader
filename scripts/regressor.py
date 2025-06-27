from vae import VAE, LatentRegressorEllipticity
from vae import train_epoch_reg, val_epoch_reg, save_checkpoint, load_checkpoint
from vae import SimpleBlendDataset
import numpy as np
import os
import torch
from torch.utils.data import random_split, DataLoader

blend_path = "data_generation/sim_lsst_norm_blended.npy"
iso_path = "data_generation/sim_lsst_norm_iso.npy"
iso_noisy_path = "data_generation/sim_lsst_norm_iso_noisy.npy"
e1_path = "data_generation/sim_lsst_e1.npy"
e2_path = "data_generation/sim_lsst_e2.npy"
redshift_path = "data_generation/sim_lsst_redshift.npy"
rab_path = "data_generation/sim_lsst_rab.npy"
avg_max_vals_path = "data_generation/sim_lsst_avg_max_vals.npy"

avg_max_vals = np.load(avg_max_vals_path)
sample_size = 200000

dataset = SimpleBlendDataset(sample_size, blend_path, iso_path, iso_noisy_path, e1_path, e2_path, redshift_path, rab_path, avg_max_vals_path)

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
    num_workers=8, # Parallel loading
    pin_memory=True if torch.cuda.is_available() else False
)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

vae = VAE(latent_dim=32).to(device)
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-4, weight_decay=1e-5)
checkpoint_path = "../notebooks/checkpoints/vae_checkpoint_debvader_1e-3.pth"

epoch, train_loss, train_mse, train_kl, val_loss, val_mse, val_kl = load_checkpoint(vae, optimizer, checkpoint_path)

regressor = LatentRegressorEllipticity(latent_dim=32).to(device)
optimizer = torch.optim.Adam(regressor.parameters(), lr=1e-3)

# --- Training Loop ---
num_epochs = 100
train_loss = np.empty(num_epochs)
val_loss = np.empty(num_epochs)
train_mse = np.empty(num_epochs)
val_mse = np.empty(num_epochs)
train_kl = np.empty(num_epochs)
val_kl = np.empty(num_epochs)

# Reset checkpoint file
checkpoint_path = "../notebooks/checkpoints/regressor_checkpoint_no_filtered_ellip_1.pth"

for epoch in range(num_epochs):
    print("Training ...")
    train_loss[epoch] = train_epoch_reg(vae, regressor, device, train_loader, optimizer)
    print("Evaluating ...")
    val_loss[epoch] = val_epoch_reg(vae, regressor, device, val_loader)

    print(f"\nEPOCH {epoch + 1}/{num_epochs}")
    print(f"Train Loss: {train_loss[epoch]:.3e}")
    print(f"Val   Loss: {val_loss[epoch]:.3e}")

    if (epoch + 1) % 10 == 0 or (epoch + 1) == num_epochs:
        save_checkpoint(epoch, regressor, optimizer,
                        train_loss[:epoch + 1], train_mse[:epoch + 1], train_kl[:epoch + 1],
                        val_loss[:epoch + 1], val_mse[:epoch + 1], val_kl[:epoch + 1], checkpoint_path=checkpoint_path)
