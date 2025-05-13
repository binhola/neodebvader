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
    sampling_function = btk.sampling_functions.PairSampling(
        stamp_size = stamp_size,
        max_shift = max_shift,
        seed = 24,
        bright_cut = 25.3,
        dim_cut = 28.0
    )
    
    LSST = btk.survey.get_surveys("LSST")
    
    output_dir = os.path.join(store_path, "blended", dataset_type)
    os.makedirs(output_dir, exist_ok=True)
   
    num_batches = num_images // batch_size 
    
    for i in tqdm(range(num_batches), desc=f"Generating {dataset_type} data"):
        draw_generator = btk.draw_blends.CatsimGenerator(
            catalog, sampling_function, LSST,
            batch_size=batch_size, njobs=1,
            add_noise="all"
        )
        blend_batch = next(draw_generator)
        blend_batch.save(output_dir, i)
    
SCRATCH = os.getenv("ALL_CCFRSCRATCH")
SCRATCH = os.path.join(SCRATCH, "deblending")

generate_data(SCRATCH, 200000, 1000, "blended_training_pair_noise")
