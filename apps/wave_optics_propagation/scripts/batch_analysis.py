"""The figures of the paper of the batch of runs over max_delta (`scripts/cluster/batch-run.slurm`).

For each run, the field two thirds of the way behind the lens is compared with the
classical numerics: the fidelity to it, fitted by `1 + a max_delta**2`, and the success
probability of the phase protocol up to it, fitted by `exp(a max_delta)`. The figures
are saved into this app's `.output`.

The figures are typeset with LaTeX, which needs a LaTeX installation.
"""

# %%
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from qiu_classical_simulation.wave_optics.classical_numerics import (
    classical_numerics_simulation,
)
from qiu_classical_simulation.wave_optics.result import free_space_snapshot_name
from qiu_classical_simulation.wave_optics.storage import (
    load_experiment,
    load_parameters,
    run_folders,
)
from scipy.optimize import curve_fit

APP_DIR = Path(__file__).resolve().parents[1]
BATCH_DIR = APP_DIR / ".result" / "batch"
OUTPUT_DIR = APP_DIR / ".output"

plt.rcParams["text.usetex"] = True
plt.style.use("seaborn-v0_8-paper")
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 16,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 12,
        "figure.titlesize": 18,
        "figure.dpi": 600,
        "savefig.dpi": 600,
        "axes.linewidth": 1.2,
    }
)

# %% The runs, without repetitions of the same parameters

runs = pd.DataFrame(
    [
        {**asdict(load_parameters(folder)), "path_to_folder": folder}
        for folder in run_folders(BATCH_DIR)
    ]
)
runs = runs.drop_duplicates(
    subset=list(
        runs.columns.difference(["experiment_datetime", "path_to_folder", "uuid"])
    )
).reset_index(drop=True)
runs = runs.sort_values(by="max_delta")

# %% The fidelity to the classical numerics and the success probability of each run

for index, row in runs.iterrows():
    params, results = load_experiment(Path(str(row["path_to_folder"])))

    reference_step = params.num_of_steps_after_lens * 2 // 3 + 1
    reference_snapshot = free_space_snapshot_name(reference_step)
    state = results.snapshots[reference_snapshot]
    classical_state = classical_numerics_simulation(
        params, propagation_after_lens=params.step_size_after_lens * reference_step
    )
    state = state / np.linalg.norm(state)
    classical_state = classical_state / np.linalg.norm(classical_state)

    runs.at[index, "overlap"] = np.abs(np.vdot(state, classical_state))
    runs.at[index, "success_probability"] = results.success_probability(
        reference_snapshot
    )

# %% The fidelity over max_delta, for the sample-based propagator

NUMS_TO_CUT = 9
NUMS_TO_KEEP_FOR_FITTING = 8

sample_based_runs = runs[~runs["direct_propagator"]]


def column(name: str) -> np.ndarray:
    """Return a column of the sample-based runs, without the last NUMS_TO_CUT."""
    return np.asarray(sample_based_runs[name], dtype=float)[:-NUMS_TO_CUT]


max_deltas = column("max_delta")
fidelities = column("overlap") ** 2


def quadratic_fit(x, a):
    """The fidelity model `1 + a x**2`."""
    return a * x**2 + 1


# the range of max_delta of the points of the fits
fit_range = (
    rf"$\Delta_{{max}}\in[{max_deltas[0]:.3g}, "
    rf"{max_deltas[NUMS_TO_KEEP_FOR_FITTING - 1]:.3g}]$"
)

(a_fidelity,), _ = curve_fit(
    quadratic_fit,
    max_deltas[:NUMS_TO_KEEP_FOR_FITTING],
    fidelities[:NUMS_TO_KEEP_FOR_FITTING],
)
x_fit = np.linspace(min(max_deltas), max(max_deltas), 100)

fig, ax = plt.subplots(1, 1)
ax.scatter(max_deltas, fidelities, c="C0", label="Numerical results")
ax.plot(
    x_fit,
    quadratic_fit(x_fit, a_fidelity),
    c="C0",
    linestyle="--",
    label=f"Quadratic fit, through {fit_range}",
)
ax.set_xlabel(r"Maximum Delta $\Delta_{max}$")
ax.set_ylabel(r"Fidelity to Classical Numerics $F$")
ax.legend()
fig.tight_layout()
OUTPUT_DIR.mkdir(exist_ok=True)
fig.savefig(OUTPUT_DIR / "overlap_with_classical_numerics.pdf")
print(f"Quadratic coefficient of the fidelity: {a_fidelity}")

# %% The success probability over max_delta


def exponential_fit(x, a):
    """The success probability model `exp(a x)`."""
    return np.exp(a * x)


success_probabilities = column("success_probability")
(a_success,), _ = curve_fit(
    exponential_fit,
    max_deltas[:NUMS_TO_KEEP_FOR_FITTING],
    success_probabilities[:NUMS_TO_KEEP_FOR_FITTING],
)

fig, ax = plt.subplots(1, 1)
ax.scatter(max_deltas, success_probabilities, c="C0", label="Numerical results")
ax.plot(
    x_fit,
    exponential_fit(x_fit, a_success),
    c="C0",
    linestyle="--",
    label=f"Linear fit, through {fit_range}",
)
ax.set_yscale("log")
ax.set_xlabel(r"Maximum Delta $\Delta_{max}$")
ax.set_ylabel(r"Success Probability $P_{success}$")
ax.legend(loc="lower left")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "success_probability_vs_max_delta.pdf")
print(f"Largest success probability: {np.max(success_probabilities)}")

# %% The slopes of the success probability in log and log-log scale

slope, _ = np.polyfit(
    max_deltas[:NUMS_TO_KEEP_FOR_FITTING],
    np.log(success_probabilities[:NUMS_TO_KEEP_FOR_FITTING]),
    1,
)
print(f"Slope of the success probability in log scale: {slope}")

slope_log_log, intercept_log_log = np.polyfit(
    np.log(max_deltas[:5]), np.log(success_probabilities[:5]), 1
)
print(f"Slope of the success probability in log-log scale: {slope_log_log}")
print(f"Intercept of the success probability in log-log scale: {intercept_log_log}")
