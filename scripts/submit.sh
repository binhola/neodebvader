#!/bin/bash

# =============================================================================
# Run Benchmarks
# =============================================================================
sbatch --account=tkc@a100 --nodes=1 --gres=gpu:1 --tasks-per-node=1 -C a100 --job-name=v1 script.slurm regressor.py
