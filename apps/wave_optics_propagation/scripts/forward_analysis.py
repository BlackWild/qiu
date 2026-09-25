"""The figure of the paper of a single run: the intensity through the lens and behind it.

Compares the beam waist along the propagation with the one behind an ideal thin lens,
and draws the intensity map with the lens surface, the focal point and the principal
plane. The figure is saved into the run folder and into this app's `.output`.

The figures are typeset with LaTeX, which needs a LaTeX installation.
"""

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from qiu_classical_simulation.wave_optics.analysis import (
    beam_waist,
    lens_surface,
    principal_plane_position,
    propagation_distances,
    thin_lens_reference_states,
)
from qiu_classical_simulation.wave_optics.storage import load_experiment

APP_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = APP_DIR / ".result"
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

# %% The run to analyze
# Local runs:
#   2026-01-08_07-54-21  reverse, various parameters
#   2026-01-08_08-41-00  forward, various parameters
#   2026-01-13_12-46-04  reverse, max_delta=0.05
#   2026-01-13_13-20-27  reverse, max_delta=0.01
#   2026-01-13_14-02-31  reverse, max_delta=0.01, longer z
#   2026-01-16_09-55-12  non-Fresnel, max_delta=0.1, check waist
# Cluster runs (copy them into RESULTS_DIR first):
#   2026-01-27_10-14-10  non-Fresnel, max_delta=0.1, waist 23, 7 qubits
#   2026-01-27_11-47-28  non-Fresnel, max_delta=0.1, waist 23, 6 qubits
#   2026-01-27_12-14-57  non-Fresnel, max_delta=0.1, waist 20, 6 qubits
#   2026-01-27_12-26-13  non-Fresnel, max_delta=0.1, waist 25, 6 qubits
#   2026-01-27_12-11-53  non-Fresnel, max_delta=0.01, waist 20, 6 qubits
#   2026-01-27_16-06-30  non-Fresnel, max_delta=0.1, waist 23, 6 qubits, sample-based
#   2026-01-27_16-23-42  non-Fresnel, max_delta=0.1, waist 25, 6 qubits, sample-based
#   2026-01-27_16-36-38  non-Fresnel, max_delta=0.01, waist 23, 6 qubits, direct
#   2026-01-27_16-50-50  non-Fresnel, max_delta=0.01, waist 23, 6 qubits, sample-based
#   2026-01-29_10-33-38  non-Fresnel, max_delta=0.1, waist 25, 6 qubits, direct
#   5d25e6d4672541b88adec9c49e7cee22, 9c37f675cc31430fa50cd6397166547e,
#   2385107e059c4e7d97494ee394c54dac, c137fa26c71f4d64a05047379aa8c74f (the figure)

EXPERIMENT_ID = "c137fa26c71f4d64a05047379aa8c74f"
RUN_FOLDER = RESULTS_DIR / EXPERIMENT_ID

params, results = load_experiment(RUN_FOLDER)
print(params)

# %% The snapshots and the thin lens references along the propagation

lens_states = results.lens_states(params.lens_slices)
after_lens_states = results.free_space_states(params.num_of_steps_after_lens)
simulated_states = lens_states + after_lens_states
z_axis = propagation_distances(params)
principal_plane = principal_plane_position(params)
thin_lens_states = thin_lens_reference_states(params)

x_values = params.x_axis.values
beam_waists = [beam_waist(state, x_values) for state in simulated_states]
thin_lens_beam_waists = [beam_waist(state, x_values) for state in thin_lens_states]

# %% The beam waist along the propagation

plt.plot(z_axis, beam_waists, "-", label="Simulated beam waist")
plt.plot(z_axis, thin_lens_beam_waists, "r-", label="Thin lens analytical beam waist")
plt.xlabel("Distance after lens (m)")
plt.ylabel("Beam waist (m)")
plt.legend()
plt.show()

# %% The intensity map

fig, ax = plt.subplots(1, 1, figsize=(1.2 * 6.4, 1.2 * 4.8))

vmin = 0
vmax = np.max([np.abs(state) ** 2 for state in after_lens_states]) + 0.01
image = ax.imshow(
    [np.abs(state) ** 2 for state in lens_states],
    extent=(0.0, params.transverse_length, 0.0, params.lens_thickness),
    aspect="auto",
    vmin=vmin,
    vmax=vmax,
    origin="lower",
)
ax.imshow(
    [np.abs(state) ** 2 for state in after_lens_states],
    extent=(
        0.0,
        params.transverse_length,
        params.lens_thickness,
        params.lens_thickness + params.propagation_after_lens,
    ),
    aspect="auto",
    vmin=vmin,
    vmax=vmax,
    origin="lower",
)
ax.set_ylim(0, params.lens_thickness + params.propagation_after_lens)
ax.set_xlim(0.0, params.transverse_length)

# the beam waist behind the lens
middle_point = params.transverse_length / 2 + params.delta_x / 2
behind_lens = slice(params.lens_slices, None)
ax.plot(
    (middle_point + np.array(beam_waists))[behind_lens],
    z_axis[behind_lens],
    "r--",
    lw=2,
    label="Beam waist",
)
ax.plot(
    (middle_point - np.array(beam_waists))[behind_lens],
    z_axis[behind_lens],
    "r--",
    lw=2,
)

# the lens: its convex surface, its plane surface and its sides
surface = lens_surface(params)
left_edge, right_edge = x_values[0], x_values[-1] + params.delta_x / 2
ax.plot(x_values, surface, color="lightgray", lw=3, ls="-", label="Lens surface")
if params.lens_reverse_order:
    ax.axhline(z_axis[0], color="lightgray", lw=8, ls="-")
    ax.plot([left_edge, left_edge], [0, surface[0]], color="lightgray", lw=6, ls="-")
    ax.plot([right_edge, right_edge], [0, surface[-1]], color="lightgray", lw=3, ls="-")
else:
    ax.axhline(params.lens_thickness, color="lightgray", lw=3, ls="-")
    two_steps = 2 * params.step_size_after_lens
    ax.plot(
        [left_edge, left_edge],
        [params.lens_thickness - two_steps, surface[0] + two_steps],
        color="lightgray",
        lw=6,
        ls="-",
    )
    ax.plot(
        [right_edge, right_edge],
        [params.lens_thickness, surface[-1]],
        color="lightgray",
        lw=3,
        ls="-",
    )

focal_point = z_axis[int(np.argmin(beam_waists))]
ax.axhline(focal_point, color="green", lw=1.5, ls="--", label="Focal point")
ax.axhline(principal_plane, color="orange", lw=1.5, ls="--", label="Principal plane")

ax.set_xlabel(r"Transverse position ($\mu m$)")
ax.set_ylabel(r"Propagation distance ($\mu m$)")
fig.colorbar(image, ax=ax, label="Field intensity")
ax.legend(loc="upper right", bbox_to_anchor=(1, 0.9), facecolor="0.7", edgecolor="0.5")
micrometers = ticker.FuncFormatter(lambda x, _: f"{x * 1e6:g}")
ax.xaxis.set_major_formatter(micrometers)
ax.yaxis.set_major_formatter(micrometers)
fig.tight_layout()

fig.savefig(RUN_FOLDER / "wave_propagation.pdf")
OUTPUT_DIR.mkdir(exist_ok=True)
order = "r" if params.lens_reverse_order else "nr"
fresnel = "f" if params.fresnel_approximation else "nf"
fig.savefig(OUTPUT_DIR / f"wave_propagation-{order}-{fresnel}.pdf")

# %%
print(f"Focal point: {focal_point}, radius of curvature: {params.radius_of_curvature}")
