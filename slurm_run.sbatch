#!/usr/bin/env bash
#SBATCH --job-name=hproj
#SBATCH --partition=gpu.L40S
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=4-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

mkdir -p logs

# load modules if your cluster uses them
module load cuda || true

# activate setup uv
pip3 install uv

# good practice for CPU-heavy native libs inside Dask workers
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

bash scripts/run_full.sh