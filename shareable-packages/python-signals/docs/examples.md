# Examples

## The Fourier pair of a Gaussian

The Fourier transform of a Gaussian `exp(-x**2 / (2 sigma**2))` is the Gaussian `sqrt(2 pi) sigma exp(-sigma**2 k**2 / 2)` of the angular wavenumber `k`. Sampling the Gaussian on a position axis and its transform on the conjugate axis checks that the DFT, the conjugate spacing and the `FFT` ordering fit together.

```python
import numpy as np
from python_signals.algebraic_signal import AlgebraicSignal
from python_signals.integer_axis import IndexOrdering
from python_signals.physical_axis import (
    AngularWavenumberAxis,
    MomentumAxis,
    PositionAxis,
)
from python_signals.signal import Signal

sigma = 1.0
x_axis = PositionAxis(size=128, delta_x=0.25, ordering=IndexOrdering.FFT)
k_axis = AngularWavenumberAxis.from_position_axis(x_axis)

gaussian = AlgebraicSignal(x_axis, lambda x: np.exp(-(x**2) / (2 * sigma**2)))
# the DFT times the spacing approximates the continuous transform
spectrum = Signal(k_axis, np.fft.fft(gaussian.data) * x_axis.period)
expected = AlgebraicSignal(
    k_axis, lambda k: np.sqrt(2 * np.pi) * sigma * np.exp(-(sigma**2) * k**2 / 2)
)
assert np.allclose(spectrum.data, expected.data, atol=1e-12)

# in natural units, the momentum axis has the same values, but another domain
p_axis = MomentumAxis.from_position_axis(x_axis, hbar=1.0)
assert np.allclose(p_axis.values, k_axis.values)
assert p_axis != k_axis
```

The sampled spectrum matches the analytic transform to machine precision, since the Gaussian decays well within the window of `128 * 0.25 = 32` and the conjugate axis extends to `pi / 0.25`, far beyond its width. The momentum axis of `hbar = 1` has the same values but is not equal to the angular wavenumber axis, so signals on the two cannot be combined by mistake.

## The free spreading of a wave packet

A free particle of mass `m` evolves by the phase `exp(-i p**2 t / (2 m hbar))` of its momentum. As a quadratic signal on the momentum axis, this is a split-step propagation in a single step; the width of a Gaussian packet of initial width `sigma` then grows as `sigma sqrt(1 + (hbar t / (2 m sigma**2))**2)`.

```python
import numpy as np
from python_signals.algebraic_signal import QuadraticSignal
from python_signals.integer_axis import IndexOrdering
from python_signals.physical_axis import MomentumAxis, PositionAxis
from python_signals.signal import Signal

hbar, mass, sigma, time = 1.0, 1.0, 1.0, 3.0
x_axis = PositionAxis(size=512, delta_x=0.1, ordering=IndexOrdering.FFT)
p_axis = MomentumAxis.from_position_axis(x_axis, hbar=hbar)

# a packet whose probability density has the standard deviation sigma
packet = Signal(x_axis, np.exp(-(x_axis.values**2) / (4 * sigma**2)))
psi = packet.normalized_data

kinetic_phase = QuadraticSignal(p_axis, alpha=-time / (2 * mass * hbar))
evolved = np.fft.ifft(np.exp(1j * kinetic_phase.data) * np.fft.fft(psi))

density = np.abs(evolved) ** 2
width = np.sqrt(np.sum(x_axis.values**2 * density) / np.sum(density))
expected = sigma * np.sqrt(1 + (hbar * time / (2 * mass * sigma**2)) ** 2)
assert np.isclose(width, expected)

# the phase in terms of the integer indices, as a qubit encoding applies it
index_phase = kinetic_phase.effective_alpha * p_axis.index**2
assert np.allclose(kinetic_phase.data, index_phase)
```

The numerical width agrees with the analytic one. `effective_alpha` gives the same phase as a quadratic function of the integer indices of the momentum axis, which is the form a circuit applying it to the basis states of a qubit register needs.

## A potential and its force from SymPy

A signal built from a SymPy expression keeps the expression, so derived quantities can be computed symbolically and sampled on the same axis. Here, the force `-dV/dx` of a double-well potential is derived symbolically and compared with the finite differences of the sampled potential.

```python
import numpy as np
import sympy
from python_signals.algebraic_signal import AlgebraicSignal
from python_signals.integer_axis import IndexOrdering
from python_signals.physical_axis import PositionAxis

# a CENTERED axis has ascending values, as numpy.gradient needs
x_axis = PositionAxis(size=401, delta_x=0.01, ordering=IndexOrdering.CENTERED)

potential = AlgebraicSignal.from_sympy(x_axis, "x**4 / 4 - x**2")
force = AlgebraicSignal.from_sympy(
    x_axis, -sympy.diff(potential.expression, potential.symbol)
)
x = potential.symbol
assert force.expression == -(x**3) + 2 * x

finite_differences = -np.gradient(potential.data, x_axis.values)
assert np.allclose(force.data[1:-1], finite_differences[1:-1], atol=1e-3)

# the minima of the double well, where the force vanishes
minima = [float(root) for root in sympy.solve(force.expression, x) if root != 0]
assert np.allclose(sorted(minima), [-np.sqrt(2), np.sqrt(2)])
assert np.allclose(force(np.array(minima)), 0.0)
```

The symbolic force agrees with the central differences of the samples up to their `O(delta_x**2)` error, and it can be evaluated anywhere, e.g. at the minima SymPy finds, not only on the axis.

## Comparing a measurement with a model

Measured data is a sampled `Signal`, a model an `AlgebraicSignal`. Their difference is a sampled signal on the same axis, and the arithmetic guards against combining samples of different grids or raw arrays of unknown ordering.

```python
import numpy as np
from python_signals.algebraic_signal import AlgebraicSignal
from python_signals.integer_axis import IndexOrdering
from python_signals.physical_axis import PositionAxis
from python_signals.signal import Signal

x_axis = PositionAxis(size=64, delta_x=0.5, ordering=IndexOrdering.CENTERED)
rng = np.random.default_rng(seed=1)
noise = 0.01 * rng.standard_normal(x_axis.size)

model = AlgebraicSignal(x_axis, lambda x: 2.0 * np.exp(-(x**2) / 8))
measurement = Signal(x_axis, model.data + noise)

residual = measurement - model
assert type(residual) is Signal
assert np.allclose(residual.data, noise)
assert np.sqrt(np.mean(residual.data**2)) < 0.02

# a raw array must be wrapped in a signal on the intended axis
background = Signal(x_axis, np.full(x_axis.size, 0.1))
assert np.allclose((measurement - background).data, measurement.data - 0.1)
try:
    measurement - np.full(x_axis.size, 0.1)
except TypeError:
    pass
else:
    raise AssertionError("raw arrays are rejected")

# samples of another grid cannot be combined
coarse = PositionAxis(size=64, delta_x=1.0, ordering=IndexOrdering.CENTERED)
try:
    measurement - Signal(coarse, np.zeros(64))
except ValueError:
    pass
else:
    raise AssertionError("signals on different axes are rejected")
```

The residual is the added noise, up to rounding. Subtracting a raw array raises a `TypeError`, and subtracting a signal on a coarser grid a `ValueError`, instead of silently combining samples at different positions.

## A spectrum for plotting

Plots and tables of a spectrum need ascending frequencies, while `numpy.fft` returns the `FFT` ordering. A conjugate axis with `keep_ordering=True` takes the `CENTERED` ordering of the position axis, and the spectrum is reordered to it with `numpy.fft.fftshift`.

```python
import numpy as np
from python_signals.algebraic_signal import AlgebraicSignal
from python_signals.integer_axis import IndexOrdering
from python_signals.physical_axis import PositionAxis, SpatialFrequencyAxis
from python_signals.signal import Signal

x_axis = PositionAxis(size=256, delta_x=0.0625, ordering=IndexOrdering.CENTERED)
f_axis = SpatialFrequencyAxis.from_position_axis(x_axis, keep_ordering=True)
assert np.all(np.diff(f_axis.values) > 0)  # ascending

# a wave packet of the spatial frequency 2 under a Gaussian envelope
packet = AlgebraicSignal(
    x_axis, lambda x: np.exp(-(x**2)) * np.cos(2 * np.pi * 2.0 * x)
)
# the CENTERED positions are shifted to the positions numpy.fft assumes, and back
spectrum = Signal(
    f_axis,
    np.fft.fftshift(np.fft.fft(np.fft.ifftshift(packet.data))) * x_axis.period,
)

peaks = f_axis.values[np.argsort(np.abs(spectrum.data))[-2:]]
assert np.allclose(sorted(peaks), [-2.0, 2.0])
assert np.allclose(spectrum.data.imag, 0.0, atol=1e-12)  # an even packet
```

The samples `f_axis.values` and `spectrum.data` are ready to be plotted against each other. The spectrum peaks at the spatial frequencies `-2` and `2` of the carrier, and it is real, as the transform of an even function must be; without the `ifftshift` of the `CENTERED` samples, it would alternate in sign.
