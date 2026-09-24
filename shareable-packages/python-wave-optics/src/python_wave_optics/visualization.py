"""Plots of transverse fields."""

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

_PHASE_TICKS = np.pi * np.arange(-4, 5) / 4
_PHASE_LABELS = [
    r"$-\pi$",
    r"$-\frac{3\pi}{4}$",
    r"$-\frac{\pi}{2}$",
    r"$-\frac{\pi}{4}$",
    r"$0$",
    r"$\frac{\pi}{4}$",
    r"$\frac{\pi}{2}$",
    r"$\frac{3\pi}{4}$",
    r"$\pi$",
]


def plot_wavefunction(
    psi: npt.ArrayLike,
    plot_size_scale: float = 1,
    normalize: bool = True,
    ylim: float | None = None,
) -> tuple[Figure, tuple[Axes, Axes]]:
    """Plot the magnitude and the phase of a field over its samples, side by side.

    Args:
        psi: The amplitudes of the field.
        plot_size_scale: The scale of the figure size.
        normalize: If True, plot the magnitude normalized to unit norm.
        ylim: The upper limit of the magnitude axis, if given.

    Returns:
        The figure and its axes of the magnitude and the phase.
    """
    amplitudes = np.asarray(psi)
    magnitude = np.abs(amplitudes)
    if normalize:
        magnitude = magnitude / np.linalg.norm(magnitude)

    fig, (ax_magnitude, ax_phase) = plt.subplots(
        1, 2, figsize=(2 * 6.4 * plot_size_scale, 4.8 * plot_size_scale)
    )
    ax_magnitude.plot(magnitude)
    ax_phase.plot(np.angle(amplitudes))

    if ylim is not None:
        ax_magnitude.set_ylim(0, ylim)
    ax_phase.set_ylim(-np.pi - 0.3, np.pi + 0.3)
    ax_phase.set_yticks(_PHASE_TICKS, _PHASE_LABELS)

    ax_magnitude.set_xlabel(r"$|x\rangle$")
    ax_phase.set_xlabel(r"$|x\rangle$")
    ax_magnitude.set_ylabel(r"$|\psi(x)|$")
    ax_phase.set_ylabel(r"$\angle\psi(x)$")
    return fig, (ax_magnitude, ax_phase)
