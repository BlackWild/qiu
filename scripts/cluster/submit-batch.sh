#!/bin/bash
# ──────────────────────────────────────────────────────────────
# Submit the batch of runs over max_delta (batch-run.slurm), from anywhere:
#
#    bash scripts/cluster/submit-batch.sh
#
# The environment is synced once here, before the array job, whose tasks run
# concurrently: syncing the same .venv in each of them would race.
# ──────────────────────────────────────────────────────────────
set -euo pipefail

cd "$(dirname "$0")/../.."            # the repository root, where .venv and uv.lock are
uv sync --all-packages --no-default-groups --inexact --locked  # the packages all tasks share
mkdir -p scripts/cluster/.result      # SLURM does not create the directory of the logs

cd scripts/cluster                    # the jobs expect to be submitted from scripts/cluster/
sbatch batch-run.slurm
