import os
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import random
import galsim

def normalize_non_linear(images, avg_max_vals, beta=2.5):
    """Vectorized normalization from Arcelin et al. 2020 (Eq. 4)"""
    scaled = beta * images / avg_max_vals[:, None, None]
    return np.tanh(np.arcsinh(scaled))

def denormalize_non_linear(images_normed, avg_max_vals, beta=2.5):
    """Vectorized inverse normalization."""
    unscaled = np.sinh(np.arctanh(images_normed))
    return unscaled * avg_max_vals[:, None, None] / beta

def log_norm(images, vmin=None, vmax=None):
    normed = []
    for b in range(images.shape[0]):
        img = images[b]
        # Set vmin and vmax if not given
        min_val = vmin[b] if vmin is not None else np.percentile(img, 1)
        max_val = vmax[b] if vmax is not None else np.percentile(img, 99)
        
        # Clip to positive range to avoid log(0)
        img_clip = np.clip(img, a_min=max(min_val, 1e-10), a_max=max_val)
        
        # Apply log10 normalization to [0,1]
        norm_img = (np.log10(img_clip) - np.log10(min_val)) / (np.log10(max_val) - np.log10(min_val))
        normed.append(norm_img)
    return np.stack(normed)

def de_log_norm(normed_images, vmin, vmax):
    denormed = []
    for b in range(normed_images.shape[0]):
        norm_img = normed_images[b]
        val = 10**(norm_img * (np.log10(vmax[b]) - np.log10(vmin[b])) + np.log10(vmin[b]))
        denormed.append(val)
    return np.stack(denormed)

# def log_norm(images, mu=1.0, nu=2.5):
#     """Applies per-band logarithmic normalization."""
#     return np.stack([np.log1p(mu * band) / np.log1p(mu * nu) for band in images])

# def log_denorm(images_normed, mu=1.0, nu=2.5):
#     """Applies inverse of per-band logarithmic normalization."""
#     return np.stack([np.expm1(band * np.log1p(mu * nu)) / mu for band in images_normed])

class SimpleBlendDataset(Dataset):
    def __init__(self, sample_size, blend_path, iso_path, iso_noisy_path, e1_path, e2_path, redshift_path, rab_path, avg_max_vals_path, augment=False):
        self.total_samples = sample_size  
        self.shape = (self.total_samples, 6, 45, 45)

        self.blend_imgs = np.memmap(blend_path, dtype=np.float32, mode="r", shape=self.shape)
        self.iso_imgs = np.memmap(iso_path, dtype=np.float32, mode="r", shape=self.shape)
        self.iso_imgs_noisy = np.memmap(iso_noisy_path, dtype=np.float32, mode="r", shape=self.shape)

        self.e1 = np.memmap(e1_path, dtype=np.float32, mode="r", shape=(self.total_samples,))
        self.e2 = np.memmap(e2_path, dtype=np.float32, mode="r", shape=(self.total_samples,))
        self.redshift = np.memmap(redshift_path, dtype=np.float32, mode="r", shape=(self.total_samples,))
        self.rab = np.memmap(rab_path, dtype=np.float32, mode="r", shape=(self.total_samples,))

        self.avg_max_vals = np.load(avg_max_vals_path)
        self.augment = augment

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        blend_tensor = torch.from_numpy(self.blend_imgs[idx].copy()).float()
        iso_tensor = torch.from_numpy(self.iso_imgs[idx].copy()).float()
        iso_tensor_noisy = torch.from_numpy(self.iso_imgs_noisy[idx].copy()).float()

        if self.augment:
            if random.random() > 0.5:
                blend_tensor = TF.hflip(blend_tensor)
                iso_tensor_noisy = TF.hflip(iso_tensor_noisy)
            if random.random() > 0.5:
                blend_tensor = TF.vflip(blend_tensor)
                iso_tensor_noisy = TF.vflip(iso_tensor_noisy)
            angle = random.choice([0, 90, 180, 270])
            blend_tensor = TF.rotate(blend_tensor, angle)
            iso_tensor_noisy = TF.rotate(iso_tensor_noisy, angle)

        attributes = torch.tensor([self.redshift[idx], self.e1[idx], self.e2[idx]], dtype=torch.float32)
        return blend_tensor, iso_tensor_noisy, iso_tensor, attributes

def calculate_ellipticity(image, centroids):
    y, x = np.indices(image.shape)
    x_center = centroids[0]
    y_center = centroids[1]
    Qxx = np.sum((x - x_center)**2 * image)
    Qyy = np.sum((y - y_center)**2 * image)
    Qxy = np.sum((x - x_center) * (y - y_center) * image)
    e1 = (Qxx - Qyy) / (Qxx + Qyy)
    e2 = (2 * Qxy) / (Qxx + Qyy)
    return e1, e2