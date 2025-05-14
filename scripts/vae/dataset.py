import os
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import random
import galsim

def normalize_non_linear(images, avg_max_vals, beta=2.5):
    """Applies the normalization from Eq. (4) of Arcelin et al. 2020 per band."""
    normed = []
    for b in range(images.shape[0]):
        normed_band = np.tanh(np.arcsinh(beta * images[b] / avg_max_vals[b]))
        normed.append(normed_band)
    return np.stack(normed)

def denormalize_non_linear(images_normed, avg_max_vals, beta=2.5):
    """Inverse of Arcelin normalization."""
    denormed = []
    for b in range(images_normed.shape[0]):
        denormed_band = (np.sinh(np.arctanh(images_normed[b])) * avg_max_vals[b]) / beta
        denormed.append(denormed_band)
    return np.stack(denormed)

def normalize_non_linear_old(images):
    return np.tanh(np.arcsinh(images))

def denormalize_non_linear_old(images_normed):
    return np.sinh(np.arctanh(images_normed))


def log_norm(images, mu=1.0, nu=2.5):
    """Applies per-band logarithmic normalization."""
    return np.stack([np.log1p(mu * band) / np.log1p(mu * nu) for band in images])

def log_denorm(images_normed, mu=1.0, nu=2.5):
    """Applies inverse of per-band logarithmic normalization."""
    return np.stack([np.expm1(band * np.log1p(mu * nu)) / mu for band in images_normed])


class SimpleBlendDataset(Dataset):
    def __init__(self, base_path, num_files, beta=2.5, augment=True, isolated=True):
        self.file_paths = []
        self.beta = beta
        self.image_shape = None
        self.augment = augment
        all_band_maxima = []

        for i in range(num_files):
            path = os.path.join(base_path, f'blend_{i}.hdf5')
            if os.path.exists(path):
                self.file_paths.append(path)

        if not self.file_paths:
            raise ValueError(f"No valid files found in {base_path}")

        self.total_samples = len(self.file_paths) * 1000

        # Compute per-band average of max pixel values
        for path in self.file_paths:
            with h5py.File(path, 'r') as f:
                imgs = f['blend_images'][:]  # shape: (1000, 6, H, W)
                self.image_shape = imgs.shape[2:]
                max_vals = imgs.max(axis=(2, 3))  # shape: (1000, 6)
                all_band_maxima.append(max_vals)

        all_band_maxima = np.concatenate(all_band_maxima, axis=0)  # shape: (N, 6)
        self.avg_max_vals = all_band_maxima.mean(axis=0)  # shape: (6,)
        print("Average max per band (used for normalization):", self.avg_max_vals)

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        file_idx = idx // 1000
        sample_idx = idx % 1000

        with h5py.File(self.file_paths[file_idx], 'r') as f:
            image_ = f['blend_images'][sample_idx]
            image = normalize_non_linear(image_, self.avg_max_vals, beta=self.beta)

            if isolated:
                catalog_data = f[f'catalog_list/{sample_idx}'][0] 
                redshift = catalog_data['redshift']
                
                # g1 = catalog_data['g1']
                # g2 = catalog_data['g2']
                r_ab = catalog_data['r_ab']
    
                ### Ellipticity
                centroids = np.array([catalog_data["x_peak"], catalog_data["y_peak"]])
                
                e1, e2 = calculate_ellipticity(image_[2], centroids)
            else:
                blend_image = f['blend_images'][sample_idx]
                isolated_images = f['isolated_images'][sample_idx]
                
                centered_gal, shifted_gal = isolate_images
                
                catalog_cg, catalog_sg = f[f'catalog_list/{sample_idx}']
                
                centroid_cg = np.array([catalog_cg['x_peak'], catalog_cg['y_peak']])
                centroid_sg = np.array([catalog_sg['x_peak'], catalog_sg['y_peak']])

                e1_cen, e2_cen = calculate_ellipticity(centered_gal[2], centroids_cg)
                e1_shif, e2_shif = calculate_ellipticity(shifted_gal[2], centroid_sg)

                self.isolated_galaxies = isolate_images

        image_tensor = torch.from_numpy(image).float()

        if self.augment:
            # Apply random horizontal flip
            if random.random() > 0.5:
                image_tensor = TF.hflip(image_tensor)
            # Apply random vertical flip
            if random.random() > 0.5:
                image_tensor = TF.vflip(image_tensor)
            # Apply random rotation
            angles = [0, 90, 180, 270]
            angle = random.choice(angles)
            image_tensor = TF.rotate(image_tensor, angle)
    
        attributes = torch.tensor([redshift, e1, e2, r_ab], dtype=torch.float32)
        return image_tensor, attributes

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