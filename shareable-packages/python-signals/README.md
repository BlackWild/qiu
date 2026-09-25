# Python Signals

[![PyPI](https://img.shields.io/pypi/v/python-signals)](https://pypi.org/project/python-signals/) [![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)](https://www.python.org/) [![License: MIT](https://img.shields.io/badge/license-MIT-green)](https://github.com/BlackWild/qiu/blob/master/LICENSE) [![CI](https://github.com/BlackWild/qiu/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/BlackWild/qiu/actions/workflows/ci.yml) [![Docs](https://img.shields.io/badge/docs-mkdocs-blue)](https://blackwild.github.io/qiu/python-signals/)

Uniformly sampled axes and signals on them, for any kind of numerical application, classical or quantum. It depends on NumPy and `python-encore`, and optionally on SymPy.

The quantum packages of this monorepo, e.g. `qiskit-phase-propagator`, encode these signals in the basis states of qubit registers, and `python-wave-optics` builds its optical elements on them.

## Installation

```sh
pip install python-signals
```

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

A signal lives on a `PhysicalAxis`, and comes in two forms:

- `signal.Signal(axis, data)` is given by its sampled values: `data[k]` is the value at `axis.values[k]`, real or complex. `normalized_data` holds the values scaled to unit Euclidean norm.
- `algebraic_signal.AlgebraicSignal(axis, function)` is given by an algebraic expression of the axis values, held as a vectorized function such as a lambda or a NumPy function. `data` evaluates it on the axis values, calling the signal evaluates it at arbitrary values, and `to_signal()` returns the sampled `Signal`.
  - `AlgebraicSignal.from_sympy(axis, expression, symbol=None)` creates it from a SymPy expression (or a string SymPy parses), keeping the symbolic `expression` and its `symbol` for inspection. It needs the optional `sympy` extra, `python-signals[sympy]`.
  - `PolynomialSignal(axis, alpha, power)` is the algebraic signal `alpha * x**power`. Its `effective_alpha` is the coefficient in terms of the integer indices, so that `data == effective_alpha * axis.index**power`.
  - `QuadraticSignal(axis, alpha)` is the polynomial signal of power 2, also called an intensity signal.
- `algebraic_signal.SampledSignal` is the type `Signal | AlgebraicSignal`, for code that only needs the `axis` and the sampled values `data` of either kind.

The `SignalFunctionType` aliases for the functions of algebraic signals live in `algebraic_signal` as well.

### Arithmetic

Signals support `+`, `-`, `*`, `/`, `**` and negation, elementwise, with scalars (from either side) and with signals on an equal axis (axes compare by value). Raw NumPy arrays are rejected rather than silently broadcast.

| operands                                   | result                                                                 |
| ------------------------------------------ | ---------------------------------------------------------------------- |
| `Signal` and scalar or `Signal`            | a `Signal` of the combined samples                                     |
| `AlgebraicSignal` and scalar or `AlgebraicSignal` | an `AlgebraicSignal` of the composed functions, and of the composed SymPy expressions if both operands have one |
| `AlgebraicSignal` and `Signal`             | a `Signal`, sampling the algebraic operand first                       |
| `PolynomialSignal` `*` or `/` scalar       | a `PolynomialSignal` (or `QuadraticSignal`) with the scaled `alpha`, so `effective_alpha` stays available |

## Usage

```python
import numpy as np
import sympy
from python_signals.algebraic_signal import AlgebraicSignal, QuadraticSignal
from python_signals.integer_axis import IndexOrdering
from python_signals.physical_axis import PositionAxis, SpatialFrequencyAxis
from python_signals.signal import Signal

x_axis = PositionAxis(size=256, delta_x=0.1, ordering=IndexOrdering.FFT)

# an algebraic signal, from a lambda or from a symbolic expression
gaussian = AlgebraicSignal(x_axis, lambda x: np.exp(-(x**2)))
x = sympy.Symbol("x")
symbolic = AlgebraicSignal.from_sympy(x_axis, sympy.exp(-(x**2)))
assert np.allclose(gaussian.data, symbolic.data)
assert sympy.diff(symbolic.expression, x) == -2 * x * sympy.exp(-(x**2))

# the spectrum is a sampled signal on the conjugate axis, in the same ordering
f_axis = SpatialFrequencyAxis.from_position_axis(x_axis)
spectrum = Signal(f_axis, np.fft.fft(gaussian.data))
assert np.allclose(f_axis.values, np.fft.fftfreq(256, d=0.1))

# a quadratic phase profile, and its coefficient in terms of the integer indices
lens = QuadraticSignal(x_axis, alpha=-0.5)
assert np.allclose(lens.data, lens.effective_alpha * x_axis.index**2)

# arithmetic: scaled monomials stay monomials, other combinations are algebraic
phase = -0.1 * lens
assert isinstance(phase, QuadraticSignal)
beam = 2 * gaussian + symbolic
assert beam.expression is None and np.allclose(beam.data, 3 * gaussian.data)
assert (2 * symbolic + 1).expression == 2 * sympy.exp(-(x**2)) + 1
```

## Documentation

The documentation, with the API reference from the docstrings, is built from `docs/` with MkDocs and published at <https://blackwild.github.io/qiu/python-signals/>. To serve it locally, from the repository root:

```sh
uv run mkdocs serve -f shareable-packages/python-signals/mkdocs.yml
```

## Tests

From the repository root:

```sh
uv run pytest shareable-packages/python-signals
```
