#!/bin/bash
#SBATCH --job-name=wave_optics_propagation
#SBATCH --time=120:00:00
#SBATCH --partition=long,standard,gpu,gpu-test
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --output=.result/slurm-%j.out

# ──────────────────────────────────────────────────────────────
# 0. How to use this script:
#    Submit this script to SLURM with: sbatch run.slurm
# ──────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────
# 1. Go to project directory (where .venv and uv.lock are)
# ──────────────────────────────────────────────────────────────
cd $SLURM_SUBMIT_DIR
cd ..

# ──────────────────────────────────────────────────────────────
# 2. Activate the local .venv manually
#    (uv does NOT activate environments automatically)
# ──────────────────────────────────────────────────────────────
source .venv/bin/activate

# ──────────────────────────────────────────────────────────────
# 3. (Optional) Ensure packages are installed within local .venv
#    uv respects the active venv, so this installs locally
# ──────────────────────────────────────────────────────────────
uv sync --all-packages

# ──────────────────────────────────────────────────────────────
# 4. Run your Python script inside the venv using uv
# ──────────────────────────────────────────────────────────────
#    The run is stored in apps/wave_optics_propagation/.result/<uuid>
uv run python apps/wave_optics_propagation/scripts/simulate.py --max-delta=0.01 --reverse-order --fresnel-approximation
