import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import os

import torch.nn.functional as F

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def train_epoch(vae, device, dataloader, optimizer):
    vae.train()
    total_loss, total_mse, total_kl = 0.0, 0.0, 0.0

    for x, _ in dataloader:
        x = x.to(device)

        x_hat, mu, logvar = vae(x)
        loss_dict = vae.loss_function(x, x_hat, mu, logvar)

        optimizer.zero_grad()
        loss_dict['loss'].backward()
        optimizer.step()

        total_loss += loss_dict['loss'].item()
        total_mse += loss_dict['reconstruction_loss'].item()
        total_kl += loss_dict['kl_divergence'].item()

    n = len(dataloader.dataset)
    return total_loss / n, total_mse / n, total_kl / n


def val_epoch(vae, device, dataloader, plot=False):
    vae.eval()
    total_loss, total_mse, total_kl = 0.0, 0.0, 0.0

    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)

            x_hat, mu, logvar = vae(x)
            loss_dict = vae.loss_function(x, x_hat, mu, logvar)

            total_loss += loss_dict['loss'].item()
            total_mse += loss_dict['reconstruction_loss'].item()
            total_kl += loss_dict['kl_divergence'].item()

            # Plot 3 random reconstructions from this batch
            if plot:
                plotted = 0
                if plotted < 3:
                    fig, axs = plt.subplots(1, 2, figsize=(6, 3))
                    i = np.random.randint(0, x.shape[0])
                    orig = denormalize_non_linear(x[i].cpu().numpy() , dataset.avg_max_vals)
                    recon = denormalize_non_linear(x_hat[i].cpu().numpy() , dataset.avg_max_vals)
                    mse = np.mean((recon - orig) ** 2)
                    mae = np.mean(np.abs(recon - orig))

                    axs[0].imshow(orig[4], cmap="viridis")
                    axs[0].set_title("Original")
                    axs[0].axis("off")

                    axs[1].imshow(recon[4], cmap="viridis")
                    axs[1].set_title(f"Recon\nMSE:{mse:.2e}\nMAE:{mae:.2e}")
                    axs[1].axis("off")

                    plt.tight_layout()
                    plt.show()

                    plotted += 1

    n = len(dataloader.dataset)
    return total_loss / n, total_mse / n, total_kl / n

def train_epoch_reg(vae, regressor, device, dataloader, optimizer):
    vae.encoder.eval() # Freeze VAE encoder
    for param in vae.encoder.parameters():
        param.requires_grad = False

    regressor.train()

    train_loss = 0.0
    for blended, iso_noisy, iso_clean, target in dataloader:
        x = blended.to(device)
        target = target.to(device)

        e1_true = target[:, 1]
        e2_true = target[:, 2]

        with torch.no_grad():
            mu, _ = vae.encoder(x)

        e_pred = regressor(mu)
        e1_pred = e_pred[:,0]
        e2_pred = e_pred[:,1]

        # Loss function
        loss_e1 = F.mse_loss(e1_pred, e1_true,reduction="mean")
        loss_e2 = F.mse_loss(e2_pred, e2_true,reduction="mean")

        loss = loss_e1 + loss_e2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    n = len(dataloader.dataset)
    return train_loss / n

def val_epoch_reg(vae, regressor, device, dataloader):
    vae.encoder.eval()
    for param in vae.encoder.parameters():
        param.requires_grad = False

    regressor.eval()
    
    val_loss = 0.0
    with torch.no_grad():
        for blended, iso_noisy, iso_clean, target in dataloader:
            x = blended.to(device)
            target = target.to(device)
            
            e1_true = target[:, 1]
            e2_true = target[:, 2]

            mu, _ = vae.encoder(x)

            e_pred = regressor(mu)
            e1_pred = e_pred[:,0]
            e2_pred = e_pred[:,1]
            
            # Loss function
            loss_e1 = F.mse_loss(e1_pred, e1_true,reduction="mean")
            loss_e2 = F.mse_loss(e2_pred, e2_true,reduction="mean")

            loss = loss_e1 + loss_e2

            val_loss += loss.item()

    n = len(dataloader.dataset)
    return val_loss / n
    
def save_checkpoint(epoch, vae, optimizer, train_loss, train_mse, train_kl, val_loss, val_mse, val_kl, checkpoint_path="vae_checkpoint.pth"):
    torch.save({
        'epoch': epoch,
        'model_state_dict': vae.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'train_mse': train_mse,
        'train_kl': train_kl, 
        'val_loss': val_loss,
        'val_mse': val_mse,
        'val_kl': val_kl
    }, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch + 1}")

def load_checkpoint(vae, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    vae.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    train_loss = checkpoint['train_loss']
    train_mse = checkpoint['train_mse']
    train_kl = checkpoint['train_kl']
    val_loss = checkpoint['val_loss']
    val_mse = checkpoint['val_mse']
    val_kl = checkpoint['val_kl']

    print(f"Checkpoint loaded from epoch {epoch + 1}")
    return epoch, train_loss, train_mse, train_kl, val_loss, val_mse, val_kl
    
def save_model(epoch, vae, optimizer, train_loss, train_mse, train_kl, val_loss, val_mse, val_kl, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': vae.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'train_mse': train_mse,
        'train_kl': train_kl, 
        'val_loss': val_loss,
        'val_mse': val_mse,
        'val_kl': val_kl
    }, path)
    print(f"Model saved at epoch {epoch + 1}")

def plot_loss_function(num_epochs, train_loss, train_mse, train_kl, val_loss, val_mse, val_kl, name):
        fig, ax = plt.subplots(figsize=(8, 6), sharex=True)

        epochs = np.arange(1, num_epochs + 1)

        # Train Loss Plot
        ax.plot(epochs, train_loss, label="Train Loss", color="royalblue", linestyle="-")
        ax.plot(epochs, val_loss, label="Validation Loss", color="crimson", linestyle="-")

        ax.plot(epochs, train_mse, label="Train MSE", color="cyan", linestyle="--")
        ax.plot(epochs, val_mse, label="Val MSE", color="orange", linestyle="--")

        # ax.plot(epochs, train_kl, label="Train KL", color="limegreen", linestyle="--")
        # ax.plot(epochs, val_kl, label="Val KL", color="gold", linestyle="--")

        ax.set_xlabel("Epochs", fontsize=11)
        ax.set_ylabel("Loss", fontsize=11)
        ax.set_yscale("log")  # Set log scale
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()

        # Validation Loss Plot
        plt.tight_layout()
        plt.savefig(f"figures/loss_plots_{name}.png")
        plt.close()

def generated_images(vae, latent_dims, name):
    vae.eval()
    z = torch.randn(100, latent_dims).to(next(vae.parameters()).device)
    with torch.no_grad():
        imgs = vae.decoder(z).cpu()

    fig, axes = plt.subplots(10, 10, figsize=(15, 15))
    for i in range(10):
        for j in range(10):
            axes[i, j].imshow(imgs[10*i + j].squeeze(), cmap="gray")
            axes[i, j].axis("off")
    plt.savefig(f"figures/generated_images_{name}.png")
    plt.close()

def reconstructed_image(vae, name, latent_dims, img):
    vae.eval()
    z = vae.encoder(img)

    with torch.no_grad():
        mu, logvar = vae.encoder(img)
        z, kl = reparameterization(mu, logvar)
        recon = vae.decoder(z).cpu()
    
    return recon

def show_random_reconstructions(vae, dataset, device, avg_max_vals, n_samples=5):

    vae.eval()
    fig, axs = plt.subplots(n_samples, 3, figsize=(6, 3 * n_samples))
    
    for i in range(n_samples):
        idx = np.random.randint(0, len(dataset))
        img = dataset[idx][0].unsqueeze(0).to(device)  # shape [1, 1, 45, 45]
        with torch.no_grad():
            recon, mu, log_var = vae(img)


        #recon_ = log_denorm(recon.squeeze().cpu().numpy(), mu=1.0, nu=2.5)
        img_ = denormalize_non_linear(img.squeeze().cpu().numpy(), avg_max_vals)
        recon_ = denormalize_non_linear(recon.squeeze().cpu().numpy(), avg_max_vals)
        #img_ = (img.squeeze().cpu().numpy())
        band = 3
        mse = ((recon_[band] - img_[band]) ** 2).mean().item()
        mae = (np.abs(recon_[band] - img_[band])).mean().item()

        axs[i, 0].imshow(img_[band], cmap='viridis')
        axs[i, 0].set_title(f"Original (idx {idx})")
        axs[i, 1].imshow(recon_[band], cmap='viridis')
        axs[i, 1].set_title(f"Reconstruction")
        axs[i, 0].axis('off')
        axs[i, 1].axis('off')

        im2 = axs[i, 2].imshow(img_[band] - recon_[band])
        axs[i, 2].set_title(f"Residual")
        axs[i, 2].axis('off')

        # fig.colorbar(im2, ax=axs[i, 2], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()

def draw_ellipticity(ax, centroid, e1, e2, scale=20, color='red'):
    # Convert (e1, e2) to ellipticity magnitude and orientation
    e = np.sqrt(e1**2 + e2**2)
    if e > 1:
        e = 1  # limit for physical interpretation
    
    theta = 0.5 * np.arctan2(e2, e1)  # orientation in radians
    width = scale * (1 + e)
    height = scale * (1 - e)

    ellipse = Ellipse(xy=centroid, width=width, height=height,
                      angle=np.degrees(theta), edgecolor=color,
                      facecolor='none', lw=2)
    ax.add_patch(ellipse)

def plot_figures(normalized=True):
    fig, ax = plt.subplots(5,5, figsize=(15,15))
    centroid = [22, 22]
    for i in range(5):
        for j in range(5):
            image, attrs = fdataset[5*i + j]
            if normalized:
                ax[i,j].imshow(image[2])
            else:
                denorm = denormalize_non_linear(image , dataset.avg_max_vals)
                ax[i,j].imshow(denorm[2], origin="lower")  # Updated to use denorm instead of image[0, 5]
            ax[i,j].axis("off")
            e_ = np.sqrt(attrs[1]**2 + attrs[2]**2)
            # ax[i,j].set_title(f"e1 = {attrs[1]:.2f} | e2 = {attrs[2]:.2f} | e = {e_:.2f}")
            draw_ellipticity(ax[i,j], centroid, attrs[1], attrs[2])

    plt.tight_layout()
    plt.show()
    plt.close()

def plot_blended_galaxies(type, denormalized=True):
    # plot the first blend in the batch, with the r-band
    fig, ax = plt.subplots(5, 5, figsize=(20, 20))

    for i in range(5):
        for j in range(5):
            k = 5 * i + j
            img, attrs = fdataset[k]
            if type == "blended":
                img = img[0]
            elif type == "center":
                img = img[1]
            elif type == "shifted":
                img = img[2]
            else:
                raise ValueError(f"Invalid type '{type}'. Choose from 'blended', 'center', 'shifted'.")
            if denormalized:
                img = denormalize_non_linear(img, dataset.avg_max_vals)
            img = img[2, :, :]  # r-band

            ax[i, j].imshow(img, origin="lower")
            ax[i, j].axis("off")
            
            # plot centers of truth
            ax[i, j].scatter([attrs[6], attrs[8]], [attrs[7], attrs[9]],
                            c="red", marker="x")

    plt.tight_layout()
    plt.show()
