# Python Signals

Uniformly sampled axes and signals on them, for any kind of numerical application, classical or quantum. Only depends on NumPy.

`qiskit-signals` builds on this package to represent signals encoded in the computational basis states of qubit registers.

## Concepts

### Axes

Axes come in two layers, each in its own module together with the enum it uses:

- `integer_axis.IntegerAxis(size, ordering)` is a set of `size` uniformly spaced samples labeled by integer indices. The sample at array position `k` has the index `index[k]`, fixed by the axis `ordering` (`integer_axis.IndexOrdering`).
- `physical_axis.PhysicalAxis(size, period, ordering, domain)` is an `IntegerAxis` placed in a physical domain, with a spacing `period` between neighboring samples. The sample at array position `k` lies at

  ```
  values[k] = index[k] * period
  ```

The orderings of the integer indices are:

| ordering   | indices for `size = 4` | indices for `size = 5` | equivalent to                          |
| ---------- | ---------------------- | ---------------------- | -------------------------------------- |
| `NATURAL`  | `0, 1, 2, 3`           | `0, 1, 2, 3, 4`        | `numpy.arange(size)`                   |
| `FFT`      | `0, 1, -2, -1`         | `0, 1, 2, -2, -1`      | `numpy.fft.fftfreq(size) * size`       |
| `CENTERED` | `-2, -1, 0, 1`         | `-2, -1, 0, 1, 2`      | `numpy.fft.fftshift` of the `FFT` ones |

The `FFT` ordering is the one in which `numpy.fft` returns the transform of a signal, so a Fourier conjugate axis defaults to it.

The `domain` (`physical_axis.AxisDomain`) of a physical axis is `POSITION`, or one of its Fourier conjugates `MOMENTUM`, `ANGULAR_WAVENUMBER` and `SPATIAL_FREQUENCY`. The concrete physical axis classes in `physical_axis` fix the domain:

- `PositionAxis(size, delta_x, ordering)` samples positions spaced by `delta_x`.
- `MomentumAxis(size, delta_p, ordering)`, `AngularWavenumberAxis(size, delta_k, ordering)` and `SpatialFrequencyAxis(size, delta_f, ordering)` sample the Fourier conjugate domains with the given spacing.
- Their `from_position_axis` constructors create the axis conjugate to a position axis of `size` samples spaced by `delta_x`, with the spacing of the discrete Fourier transform: `2 pi hbar / (size delta_x)`, `2 pi / (size delta_x)` and `1 / (size delta_x)` respectively (see `reciprocal_period`). `MomentumAxis.from_position_axis(position_axis, hbar)` has no default for `hbar`, so the units are always explicit, e.g. `scipy.constants.hbar` for SI or `1.0` for natural units.

### Signals

The signals live in `signal`, together with the `SignalFunctionType` aliases for the functions they sample.

- `Signal(axis, signal_function)` is a function sampled on the values of a `PhysicalAxis`. `data` holds the sampled values, and `normalized_data` the values scaled to unit Euclidean norm.
- `PolynomialSignal(axis, alpha, power)` is the monomial `alpha * x**power`. Its `effective_alpha` is the coefficient in terms of the integer indices, so that `data == effective_alpha * axis.index**power`.
- `QuadraticSignal(axis, alpha)` is the monomial of power 2, also called an intensity signal.

## Usage

```python
import numpy as np
from python_signals.integer_axis import IndexOrdering
from python_signals.physical_axis import PositionAxis, SpatialFrequencyAxis
from python_signals.signal import QuadraticSignal, Signal

x_axis = PositionAxis(size=256, delta_x=0.1, ordering=IndexOrdering.FFT)
gaussian = Signal(x_axis, lambda x: np.exp(-(x**2)))

# the spectrum of the signal lives on the conjugate axis, in the same ordering
f_axis = SpatialFrequencyAxis.from_position_axis(x_axis)
spectrum = np.fft.fft(gaussian.data)
assert np.allclose(f_axis.values, np.fft.fftfreq(256, d=0.1))

# a quadratic phase profile, and its coefficient in terms of the integer indices
lens = QuadraticSignal(x_axis, alpha=-0.5)
assert np.allclose(lens.data, lens.effective_alpha * x_axis.index**2)
```

## Tests

From the repository root:

```sh
uv run pytest shareable-packages/python-signals
```
