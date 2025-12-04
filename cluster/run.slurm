#!/bin/bash
#SBATCH --job-name=wave_optics_propagation
#SBATCH --time=120:00:00
#SBATCH --partition=standard,long
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --output=.result/slurm-%j.out

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
uv sync

# ──────────────────────────────────────────────────────────────
# 4. Run your Python script inside the venv using uv
# ──────────────────────────────────────────────────────────────
uv run python apps/wave_optics_propagation/src/wave_optics_propagation/simulation_script.py
