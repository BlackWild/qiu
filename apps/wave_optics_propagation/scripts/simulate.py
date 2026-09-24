"""Simulate the lens experiment of the paper with Qiskit and store the run.

Run from anywhere, e.g. on the cluster (see `cluster/*.slurm`):

    uv run python apps/wave_optics_propagation/scripts/simulate.py --max-delta=0.01 \
        --reverse-order

The run is stored in `<results-dir>/<uuid>`, by default in this app's `.result`.
"""

# %%
from pathlib import Path

from python_wave_optics.cli import parse_parameters
from python_wave_optics.simulation import simulate
from python_wave_optics.storage import save_experiment
from tqdm import tqdm
from wave_optics_propagation.backend import QiskitBackend

APP_DIR = Path(__file__).resolve().parents[1]

# %% Parameters

parameters, results_dir = parse_parameters(
    "Simulate the lens experiment with Qiskit.", default_results_dir=APP_DIR / ".result"
)
print(parameters)
for problem in parameters.validity_problems():
    print(f"Warning: {problem}")

# %% Simulation

result = simulate(
    parameters, QiskitBackend(), progress=lambda loop, name: tqdm(loop, desc=name)
)
folder = save_experiment(results_dir, parameters, result)
print(f"Stored run {parameters.uuid} in {folder}")
print(f"Total probability of success: {result.total_probability_of_success}")
