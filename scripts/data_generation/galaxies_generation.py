import os
import numpy as np
from tqdm import tqdm

import btk
import btk.survey
import btk.draw_blends
import btk.catalog
import btk.sampling_functions

from btk.sampling_functions import DefaultSampling, SamplingFunction
from btk.utils import DEFAULT_SEED
from typing import Optional, Tuple
from astropy.table import Table

class SamplingShear(DefaultSampling):
    def __init__(
        self,
        stamp_size: float = 24.0,
        max_number: int = 2,
        min_number: int = 1,
        max_shift: Optional[float] = None,
        seed=DEFAULT_SEED,
        max_mag: float = 25.3,
        min_mag: float = -np.inf,
        mag_name: str = "i_ab",
        sigma: float = 0.02,
    ):
        """Initializes default sampling function with shear.

        Args:
            stamp_size: Defined in parent class.
            max_number: Defined in parent class.
            min_number: Defined in parent class.
            stamp_size: Defined in parent class.
            max_shift: Defined in parent class.
            seed: Defined in parent class.
            max_mag: Defined in parent class.
            min_mag: Defined in parent class.
            mag_name: Defined in parent class.
            shear: Constant (g1,g2) shear to apply to every galaxy.
        """
        super().__init__(
            stamp_size, max_number, min_number, max_shift, seed, max_mag, min_mag, mag_name
        )
        
        self.sigma = sigma
        
    def __call__(self, table: Table, **kwargs) -> Table:
        """Same as corresponding function for `DefaultSampling` but adds shear to output tables."""
        blend_table = super().__call__(table)
        theta = np.random.uniform(0, np.pi)
        shear = np.random.rayleigh(scale=self.sigma)
        blend_table["g1"] = shear * np.cos(2 * theta)
        blend_table["g2"] = shear * np.sin(2 * theta)
        return blend_table

class PairSamplingShear(SamplingFunction):
    """Sampling function for pairs of galaxies. Picks one centered bright galaxy and second dim.

    The bright galaxy is centered at the center of the stamp and the dim galaxy is shifted.
    The bright galaxy is chosen with magnitude less than `bright_cut` and the dim galaxy
    is chosen with magnitude cut larger than `bright_cut` and less than `dim_cut`. The cuts
    can be customized by the user at initialization.

    """

    def __init__(
        self,
        stamp_size: float = 24.0,
        max_shift: Optional[float] = None,
        mag_name: str = "i_ab",
        seed: int = DEFAULT_SEED,
        bright_cut: float = 25.3,
        dim_cut: float = 28.0,
        sigma: float = 0.02,
    ):
        """Initializes the PairSampling function.

        Args:
            stamp_size: Size of the desired stamp (in arcseconds).
            max_shift: Maximum value of shift from center. If None then its set as one-tenth the
                stamp size (in arcseconds).
            mag_name: Name of the magnitude column in the catalog to be used.
            seed: See parent class.
            bright_cut: Magnitude cut for bright galaxy. (Default: 25.3)
            dim_cut: Magnitude cut for dim galaxy. (Default: 28.0)
        """
        super().__init__(stamp_size, 2, 1, seed)
        self.stamp_size = stamp_size
        self.max_shift = max_shift if max_shift is not None else self.stamp_size / 10.0
        self.mag_name = mag_name
        self.bright_cut = bright_cut
        self.dim_cut = dim_cut
        
        shear = np.random.rayleigh(scale=sigma)
        self.shear = shear
    
    def __call__(self, table: Table):
        """Samples galaxies from input catalog to make blend scene."""
        if self.mag_name not in table.colnames:
            raise ValueError(f"Catalog must have '{self.mag_name}' column.")

        (q_bright,) = np.where(table[self.mag_name] <= self.bright_cut)
        (q_dim,) = np.where(
            (table[self.mag_name] > self.bright_cut) & (table[self.mag_name] <= self.dim_cut)
        )

        indexes = [np.random.choice(q_bright), np.random.choice(q_dim)]
        blend_table = table[indexes]

        blend_table["ra"] = 0.0
        blend_table["dec"] = 0.0

        x_peak, y_peak = btk.sampling_functions._get_random_center_shift(1, self.max_shift, self.rng)

        blend_table["ra"][1] += x_peak
        blend_table["dec"][1] += y_peak

        btk.sampling_functions._raise_error_if_out_of_bounds(blend_table["ra"], blend_table["dec"], self.stamp_size)
        
        theta = np.random.uniform(0, np.pi)
        blend_table["g1"] = self.shear * np.cos(2 * theta)
        blend_table["g2"] = self.shear * np.sin(2 * theta)

        return blend_table

def generate_data(store_path, num_images, batch_size, dataset_type, seed=0):
    """
    Generate isolated galaxy images and save them in the given directory.
    
    Args:
        store_path (str): Path to store the dataset.
        num_images (int): Total number of images to generate.
        batch_size (int): Number of images per batch.
        dataset_type (str): "training" or "validation".
        seed (int): Random seed for reproducibility.
    """
    catalog_name = "../../data/input_catalog.fits"  # Adjust path if needed
    catalog = btk.catalog.CatsimCatalog.from_file(catalog_name)
    
    stamp_size = 9  # in arcseconds
    # max_number = 1  # Isolated galaxies
    max_shift = 3  # No shift

    #sampling_function = SamplingShear(
    #    max_number=max_number, min_number=max_number,
    #    stamp_size=stamp_size, max_shift=max_shift,
    #    sigma=0.1
    #)
    # sampling_function = btk.sampling_functions.DefaultSampling(
    #     max_number=max_number, min_number=max_number,
    #     stamp_size=stamp_size, max_shift=max_shift,
    #     seed = 12
    # )
    sampling_function = PairSamplingShear(
        stamp_size = stamp_size,
        max_shift = max_shift,
        seed = 24,
        bright_cut = 24.3,
        dim_cut = 27.0
    )
    
    LSST = btk.survey.get_surveys("LSST")
    
    output_dir = os.path.join(store_path, "blended", dataset_type)
    os.makedirs(output_dir, exist_ok=True)
   
    num_batches = num_images // batch_size 
    
    for i in tqdm(range(num_batches), desc=f"Generating {dataset_type} data"):
        draw_generator = btk.draw_blends.CatsimGenerator(
            catalog, sampling_function, LSST,
            batch_size=batch_size, njobs=1,
            add_noise="background"
        )
        blend_batch = next(draw_generator)
        blend_batch.save(output_dir, i)
    
SCRATCH = os.getenv("ALL_CCFRSCRATCH")
SCRATCH = os.path.join(SCRATCH, "deblending")

generate_data(SCRATCH, 200000, 1000, "blended_training_pair_noise_background")
