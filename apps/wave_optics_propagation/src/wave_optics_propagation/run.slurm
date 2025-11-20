#!/bin/bash
#SBATCH --job-name=wave_optics_propagation
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --output=slurm-%j.out

# ──────────────────────────────────────────────────────────────
# 1. Go to project directory (where .venv and uv.lock are)
# ──────────────────────────────────────────────────────────────
cd $SLURM_SUBMIT_DIR

# ──────────────────────────────────────────────────────────────
# 2. Activate the local .venv manually
#    (uv does NOT activate environments automatically)
# ──────────────────────────────────────────────────────────────
source .venv/bin/activate

# ──────────────────────────────────────────────────────────────
# 3. (Optional) Ensure packages are installed within local .venv
#    uv respects the active venv, so this installs locally
# ──────────────────────────────────────────────────────────────
uv sync        # or: uv pip install -r requirements.txt

# ──────────────────────────────────────────────────────────────
# 4. Run your Python script inside the venv using uv
# ──────────────────────────────────────────────────────────────
uv run python simulation_script.py
