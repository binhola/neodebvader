#!/bin/bash

# =============================================================================
# Run Benchmarks
# =============================================================================
sbatch --account=tkc@a100 --nodes=1 --gres=gpu:1 --tasks-per-node=1 -C a100 --job-name=gen_data script.slurm galaxies_generation.py
