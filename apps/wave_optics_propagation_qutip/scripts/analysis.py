"""Analysis of a single QuTiP run against the thin lens and the classical numerics.

Plots the beam waist along the propagation, and snapshots behind the lens together with
the thin lens profile and the exact split-step field at the same distance.
"""

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from qiu_classical_simulation.wave_optics.analysis import (
    beam_waist,
    propagation_distances,
    thin_lens_reference_states,
)
from qiu_classical_simulation.wave_optics.classical_numerics import (
    classical_numerics_simulation,
)
from qiu_classical_simulation.wave_optics.storage import load_experiment
from qiu_classical_simulation.wave_optics.visualization import plot_wavefunction

APP_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = APP_DIR / ".result"

# %% The run to analyze, see the runs in README.md
# The runs of December 2025 did not store the lens model; their simulation used the
# Fresnel approximation and phases reduced modulo 2 pi.

EXPERIMENT_FOLDER = (
    RESULTS_DIR / "13-lens-simulation-after-classical-numerics" / "2025-12-10_15-12-36"
)
LEGACY_DEFAULTS = {"fresnel_approximation": True, "scale_down_phases": True}

params, results = load_experiment(EXPERIMENT_FOLDER, defaults=LEGACY_DEFAULTS)
print(params)

# %% The beam waist along the propagation

states = results.lens_states(params.lens_slices) + results.free_space_states(
    params.num_of_steps_after_lens
)
x_values = params.x_axis.values
plt.plot(
    propagation_distances(params),
    [beam_waist(state, x_values) for state in states],
    label="Simulated beam waist",
)
plt.plot(
    propagation_distances(params),
    [beam_waist(state, x_values) for state in thin_lens_reference_states(params)],
    "r-",
    label="Thin lens analytical beam waist",
)
plt.xlabel("Propagation distance (m)")
plt.ylabel("Beam waist (m)")
plt.legend()
plt.show()

# %% Snapshots behind the lens

STEPS_TO_PLOT = 5

thin_lens_states = thin_lens_reference_states(params)[params.lens_slices :]
free_space_states = results.free_space_states(params.num_of_steps_after_lens)
for step in np.linspace(
    0, params.num_of_steps_after_lens - 1, STEPS_TO_PLOT, dtype=int
):
    fig, (ax_magnitude, _) = plot_wavefunction(free_space_states[step])
    distance = params.step_size_after_lens * (step + 1)
    ax_magnitude.plot(
        np.abs(thin_lens_states[step]), "r--", label="thin lens analytical"
    )
    ax_magnitude.plot(
        np.abs(classical_numerics_simulation(params, distance)),
        "g--",
        label="classical numerics",
    )
    ax_magnitude.legend()
    fig.suptitle(f"After free space step {step + 1}/{params.num_of_steps_after_lens}")
    plt.show()
