"""Free space diffraction of a double slit, with the direct quantum propagator.

The paraxial propagation of a transverse field over a distance `dz` multiplies its
angular spectrum by `e^(-i k**2 dz / (2 k0))`, a quadratic phase in the angular
wavenumber `k`, which `MomentumDomainEvolutionQuadratic` applies between an inverse QFT
and a QFT. The field of two slits, 1 m wide on 10 qubits, diffracts over 1 km of green
light into its far-field interference pattern.

    uv run python packages/qiu-hamiltonian-simulation/examples/free_space_double_slit.py
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
from qiskit.quantum_info import Statevector
from qiu_hamiltonian_simulation.time_independent.direct import (
    MomentumDomainEvolutionQuadratic,
)
from qiu_signals.algebraic_signal import QuadraticSignal
from qiu_signals.integer_axis import IndexOrdering
from qiu_signals.physical_axis import AngularWavenumberAxis, PositionAxis

NUM_QUBITS = 10
WAVELENGTH = 550e-9
DISTANCE = 1000.0
STEPS = 100

dimension = 2**NUM_QUBITS
x_axis = PositionAxis(dimension, delta_x=1 / dimension, ordering=IndexOrdering.NATURAL)
k_axis = AngularWavenumberAxis.from_position_axis(x_axis)  # the FFT ordering
k0 = 2 * np.pi / WAVELENGTH

# %% Two slits, 5 samples wide and 35 samples apart

field = np.zeros(dimension)
middle = dimension // 2
field[middle - 20 : middle - 15] = 1
field[middle + 15 : middle + 20] = 1
psi = Statevector(field / np.linalg.norm(field))

# %% Propagation in equal steps

step = QuadraticSignal(k_axis, alpha=-DISTANCE / STEPS / (2 * k0))
propagator = MomentumDomainEvolutionQuadratic(step)

snapshots = [psi]
for _ in range(STEPS):
    snapshots.append(snapshots[-1].evolve(propagator))

# the same propagation with NumPy's FFT, at once
expected = np.fft.ifft(
    np.exp(1j * STEPS * step.data) * np.fft.fft(psi.data, norm="ortho"), norm="ortho"
)
print(
    f"Fidelity to the FFT propagation: {abs(np.vdot(snapshots[-1].data, expected)) ** 2}"
)

# %% The intensity along the way

fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
for ax, index in zip(axes.flat, np.linspace(0, STEPS, 6, dtype=int), strict=True):
    ax.plot(x_axis.values, np.abs(snapshots[index].data) ** 2)
    ax.set_title(f"z = {DISTANCE * index / STEPS:g} m")
for ax in axes[1]:
    ax.set_xlabel("Transverse position (m)")
fig.tight_layout()
plt.show()
