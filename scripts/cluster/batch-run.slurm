#!/bin/bash
#SBATCH --job-name=wave_optics_propagation
#SBATCH --array=0-29
#SBATCH --time=120:00:00
#SBATCH --partition=long,standard,gpu,gpu-test
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --output=.result/slurm-%j.out

# ──────────────────────────────────────────────────────────────
# 0. How to use this script:
#    Submit it with: bash scripts/cluster/submit-batch.sh
#    which syncs the environment once, creates scripts/cluster/.result/ for the logs, and
#    submits this array job. Its tasks run concurrently, so they do not sync the
#    environment themselves, which would race on the same .venv.
# ──────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────
# 1. Go to project directory (where .venv and uv.lock are)
# ──────────────────────────────────────────────────────────────
cd $SLURM_SUBMIT_DIR
cd ../..

# ──────────────────────────────────────────────────────────────
# 2. Activate the local .venv manually
#    (uv does NOT activate environments automatically)
# ──────────────────────────────────────────────────────────────
source .venv/bin/activate

# ──────────────────────────────────────────────────────────────
# 3. Run the simulation of this task's max_delta in the synced environment;
#    --no-sync, as `uv run` would sync the environment otherwise
# ──────────────────────────────────────────────────────────────

DELTAS=($(python - << 'EOF'
import numpy as np
for x in np.linspace(1e-3, 3e-1, 30):
    print(x)
EOF
))

#    The runs are stored in apps/wave_optics_propagation/.result/batch/<uuid>, where
#    apps/wave_optics_propagation/scripts/batch_analysis.py reads them
uv run --no-sync python apps/wave_optics_propagation/scripts/simulate.py --max-delta=${DELTAS[$SLURM_ARRAY_TASK_ID]} --reverse-order --results-dir=apps/wave_optics_propagation/.result/batch
