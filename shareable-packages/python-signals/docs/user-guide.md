# User Guide

`python-signals` has two layers: axes, which fix where a function is sampled, and signals, which are functions on an axis. Each lives in its own modules:

| module                                                     | contents                                                                 |
| ---------------------------------------------------------- | ------------------------------------------------------------------------ |
| [`integer_axis`][python_signals.integer_axis]              | `IndexOrdering`, `IntegerAxis`: samples labeled by integer indices        |
| [`physical_axis`][python_signals.physical_axis]            | `AxisDomain`, `PhysicalAxis` and its subclasses, `reciprocal_period`     |
| [`signal`][python_signals.signal]                          | `Signal`: a signal given by its sampled values                           |
| [`algebraic_signal`][python_signals.algebraic_signal]      | `AlgebraicSignal`, `PolynomialSignal`, `QuadraticSignal`, `SampledSignal` |
| [`arithmetic`][python_signals.arithmetic]                  | the arithmetic operators shared by the signal classes                    |

The package has no top-level exports; import from the modules, e.g. `from python_signals.signal import Signal`.

## Integer axes and index orderings

An [`IntegerAxis`][python_signals.integer_axis.IntegerAxis]`(size, ordering)` is a set of `size` samples, at least one, labeled by integer indices. The sample at array position `k` has the index `index[k]`, and the [`IndexOrdering`][python_signals.integer_axis.IndexOrdering] fixes which:

| ordering   | indices for `size = 4` | indices for `size = 5` | equivalent to                          |
| ---------- | ---------------------- | ---------------------- | -------------------------------------- |
| `NATURAL`  | `0, 1, 2, 3`           | `0, 1, 2, 3, 4`        | `numpy.arange(size)`                   |
| `FFT`      | `0, 1, -2, -1`         | `0, 1, 2, -2, -1`      | `numpy.fft.fftfreq(size) * size`       |
| `CENTERED` | `-2, -1, 0, 1`         | `-2, -1, 0, 1, 2`      | `numpy.fft.fftshift` of the `FFT` ones |

In general, for `N = size`:

- `NATURAL` gives `0, 1, ..., N-1`.
- `FFT` gives the non-negative indices `0, 1, ..., ceil(N/2)-1` followed by the negative ones `-floor(N/2), ..., -1`. This is the order in which `numpy.fft.fft` returns the frequencies of a transform.
- `CENTERED` gives the ascending indices `-floor(N/2), ..., ceil(N/2)-1`. For an even `N`, the most negative index `-N/2` has no positive counterpart, as in `numpy.fft`.

`FFT` and `CENTERED` contain the same indices, in a different order; `NATURAL` contains different ones, except for `N = 1`, where all orderings give `0`.

```python
import numpy as np
from python_signals.integer_axis import IndexOrdering, IntegerAxis

for size in [4, 5]:
    fft = IntegerAxis(size, IndexOrdering.FFT).index
    centered = IntegerAxis(size, IndexOrdering.CENTERED).index
    assert np.array_equal(fft, np.fft.fftfreq(size) * size)
    assert np.array_equal(centered, np.fft.fftshift(fft))
    assert np.array_equal(IntegerAxis(size, "natural").index, np.arange(size))
```

The ordering, like every enum argument of the package, may be given as its raw value, e.g. `"fft"`, since the enums of the package are `ExtendedEnum`s of [python-encore](../../python-encore/); it is normalized to the member. `index` is computed on every access, as a new array.

Two integer axes are equal if they have the same `size` and `ordering`, and they hash accordingly, so axes can be used as dict keys or cache keys.

## Physical axes

A [`PhysicalAxis`][python_signals.physical_axis.PhysicalAxis]`(size, period, ordering, domain)` places an integer axis in a physical domain, with the spacing `period` between neighboring samples, positive and finite. The sample at array position `k` lies at

```text
values[k] = index[k] * period
```

so the ordering decides both the order of the samples and their range: with `NATURAL`, a position axis covers `[0, (N-1) period]`; with `FFT` and `CENTERED`, it covers `[-floor(N/2) period, (ceil(N/2)-1) period]`, with `0` at array position `0` for `FFT` and at `N // 2` for `CENTERED`. `sampling_window_length` is `size * period`, the length of the periodic window the samples tile, one spacing longer than the distance between the first and the last sample.

The [`AxisDomain`][python_signals.physical_axis.AxisDomain] is `POSITION` or one of its Fourier conjugates `MOMENTUM`, `ANGULAR_WAVENUMBER` and `SPATIAL_FREQUENCY`; `is_fourier_domain` tells them apart. The subclasses fix the domain and name the spacing after it:

| class                                                                         | domain               | spacing argument |
| ----------------------------------------------------------------------------- | -------------------- | ---------------- |
| [`PositionAxis`][python_signals.physical_axis.PositionAxis]                   | `POSITION`           | `delta_x`        |
| [`MomentumAxis`][python_signals.physical_axis.MomentumAxis]                   | `MOMENTUM`           | `delta_p`        |
| [`AngularWavenumberAxis`][python_signals.physical_axis.AngularWavenumberAxis] | `ANGULAR_WAVENUMBER` | `delta_k`        |
| [`SpatialFrequencyAxis`][python_signals.physical_axis.SpatialFrequencyAxis]   | `SPATIAL_FREQUENCY`  | `delta_f`        |

The package does not fix units: the values are in the units of the spacing, e.g. meters for a position axis spaced in meters.

Physical axes are equal if their `size`, `ordering`, `period` and `domain` are equal, whatever their class: a `PositionAxis` equals the `PhysicalAxis` of the same attributes with the domain `POSITION`. An integer axis never equals a physical axis.

## Fourier conjugate axes

The discrete Fourier transform of `N` samples spaced by `delta_x` samples the conjugate domain with the spacing given by [`reciprocal_period`][python_signals.physical_axis.reciprocal_period]:

| domain               | spacing                      |
| -------------------- | ---------------------------- |
| `SPATIAL_FREQUENCY`  | `1 / (N delta_x)`            |
| `ANGULAR_WAVENUMBER` | `2 pi / (N delta_x)`         |
| `MOMENTUM`           | `2 pi hbar / (N delta_x)`    |

The classmethods `from_position_axis` of the conjugate axis classes create the axis conjugate to a position axis, of the same size with this spacing. `hbar` is a required argument of [`MomentumAxis.from_position_axis`][python_signals.physical_axis.MomentumAxis.from_position_axis], and must be given for the momentum domain and only for it in `reciprocal_period`, so that the units are always explicit, e.g. `scipy.constants.hbar` for SI units or `1.0` for natural units.

The conjugate axis is in the `FFT` ordering, whatever the ordering of the position axis, since that is the order of `numpy.fft.fft`: `spectrum = numpy.fft.fft(signal.data)` holds the component of the frequency `f_axis.values[m]` at `spectrum[m]`. With `keep_ordering=True`, the conjugate axis takes the ordering of the position axis instead, e.g. `CENTERED` for plotting, and the spectrum must be reordered accordingly, e.g. with `numpy.fft.fftshift`.

```python
import numpy as np
from python_signals.integer_axis import IndexOrdering
from python_signals.physical_axis import (
    AngularWavenumberAxis,
    MomentumAxis,
    PositionAxis,
    SpatialFrequencyAxis,
)

x_axis = PositionAxis(8, 0.5, IndexOrdering.CENTERED)
f_axis = SpatialFrequencyAxis.from_position_axis(x_axis)
k_axis = AngularWavenumberAxis.from_position_axis(x_axis)
p_axis = MomentumAxis.from_position_axis(x_axis, hbar=2.0)

assert f_axis.ordering is IndexOrdering.FFT
assert np.allclose(f_axis.values, np.fft.fftfreq(8, d=0.5))
assert np.allclose(k_axis.values, 2 * np.pi * f_axis.values)
assert np.allclose(p_axis.values, 2.0 * k_axis.values)  # p = hbar k

centered = SpatialFrequencyAxis.from_position_axis(x_axis, keep_ordering=True)
assert np.allclose(centered.values, np.fft.fftshift(f_axis.values))
```

!!! note "The positions the DFT assumes"
    `numpy.fft.fft` takes the sample at array position `k` to lie at `k * delta_x`. For the `NATURAL` and `FFT` orderings, `values[k]` equals `k * delta_x` modulo the window length, so the DFT of `signal.data`, times `delta_x`, approximates the continuous Fourier transform `integral f(x) e^(-2 pi i f x) dx` of a signal that decays within the window. For the `CENTERED` ordering, the positions are shifted by half the window, which multiplies the spectrum by `(-1)**m` for even sizes; transform `numpy.fft.ifftshift(signal.data)` instead.

## Sampled signals

A [`Signal`][python_signals.signal.Signal]`(axis, data)` is given by its sampled values: `data[k]` is the value at `axis.values[k]`. The data must be a one-dimensional numeric array, real or complex, with one value per sample; any array-like is accepted and converted with `numpy.asarray`, and a wrong shape or a non-numeric dtype raises a `ValueError`. Booleans are not numeric in this sense.

[`normalized_data`][python_signals.signal.Signal.normalized_data] returns the values scaled to unit Euclidean norm, e.g. the amplitudes of a quantum state. It is computed without underflow or overflow, also for values as tiny as `1e-200` or as huge as `1e300`, by first scaling the values by a power of two, which is exact. An all-zero signal has no direction and is returned unchanged.

```python
import numpy as np
from python_signals.integer_axis import IndexOrdering
from python_signals.physical_axis import PositionAxis
from python_signals.signal import Signal

axis = PositionAxis(3, 1.0, IndexOrdering.NATURAL)
signal = Signal(axis, [3e-200, 0.0, 4e-200])

assert signal.size == 3
assert np.allclose(signal.normalized_data, [0.6, 0.0, 0.8])
```

!!! warning "The data is not copied"
    `Signal` keeps the array it is given, if it already is a NumPy array: modifying that array afterwards modifies the signal. Pass a copy where the array is reused. The arithmetic operators, in contrast, always return signals with new arrays.

## Algebraic signals

An [`AlgebraicSignal`][python_signals.algebraic_signal.AlgebraicSignal]`(axis, function)` is given by an algebraic expression of the axis values, held as a vectorized function, e.g. a lambda or a NumPy function (see [`SignalFunctionType`][python_signals.algebraic_signal.SignalFunctionType]). The function is called with a whole array of values at once, so it must act elementwise, e.g. with NumPy functions rather than `math` ones. It may return a scalar for a constant signal, which is broadcast to the shape of the values.

- `data` evaluates the function on `axis.values`. It is computed on first access and cached.
- Calling the signal, `signal(values)`, evaluates it at arbitrary values, e.g. off the axis on a finer grid for plotting.
- [`to_signal()`][python_signals.algebraic_signal.AlgebraicSignal.to_signal] returns the sampled [`Signal`][python_signals.signal.Signal] on the same axis, e.g. for its `normalized_data`.

```python
import numpy as np
from python_signals.algebraic_signal import AlgebraicSignal
from python_signals.integer_axis import IndexOrdering
from python_signals.physical_axis import PositionAxis

axis = PositionAxis(16, 0.25, IndexOrdering.CENTERED)
wave = AlgebraicSignal(axis, lambda x: np.exp(1j * x))

assert np.allclose(wave.data, np.exp(1j * axis.values))
assert np.allclose(wave(np.array([0.0, np.pi])), [1.0, -1.0])
assert np.allclose(np.linalg.norm(wave.to_signal().normalized_data), 1.0)
assert np.array_equal(AlgebraicSignal(axis, lambda x: 2.0).data, np.full(16, 2.0))
```

Nothing is checked when the signal is created: a function that fails, or returns values of an incompatible shape, raises when `data` is first evaluated.

## Symbolic signals with SymPy

[`AlgebraicSignal.from_sympy(axis, expression, symbol=None)`][python_signals.algebraic_signal.AlgebraicSignal.from_sympy] creates an algebraic signal from a SymPy expression, or from a string SymPy parses, compiled to a NumPy function by `sympy.lambdify`. It keeps the `expression` and the `symbol` standing for the axis values, so that the signal can be inspected and manipulated symbolically, e.g. differentiated. Signals created otherwise have `expression` and `symbol` set to `None`. It needs SymPy, e.g. via the extra `python-signals[sympy]`; without it, `from_sympy` raises an `ImportError` naming the extra.

The symbol of the axis values is:

- the given `symbol`, if any; the expression must then have no other free symbols;
- otherwise the single free symbol of the expression, e.g. `t` for `"2*t**3"`;
- `x` for a constant expression, which gives a constant signal.

An expression of several free symbols without `symbol` raises a `ValueError`: substitute the parameters first, e.g. with `expression.subs(alpha, 3)`. A relation such as `"x > 1"` is not an algebraic expression and raises a `TypeError`.

```python
import numpy as np
import sympy
from python_signals.algebraic_signal import AlgebraicSignal
from python_signals.integer_axis import IndexOrdering
from python_signals.physical_axis import PositionAxis

axis = PositionAxis(8, 0.5, IndexOrdering.FFT)
x, alpha = sympy.symbols("x alpha")

signal = AlgebraicSignal.from_sympy(axis, (alpha * x**2).subs(alpha, 3), x)
assert np.allclose(signal.data, 3 * axis.values**2)
assert sympy.diff(signal.expression, signal.symbol) == 6 * x

from_string = AlgebraicSignal.from_sympy(axis, "2*t**3")
assert from_string.symbol == sympy.Symbol("t")
assert np.allclose(from_string.data, 2 * axis.values**3)
```

## Polynomial and quadratic signals

A [`PolynomialSignal`][python_signals.algebraic_signal.PolynomialSignal]`(axis, alpha, power)` is the algebraic signal `alpha * x**power` of a single monomial, and a [`QuadraticSignal`][python_signals.algebraic_signal.QuadraticSignal]`(axis, alpha)` the one of power 2, also called an intensity signal. They keep `alpha` and `power`, so that code can recognize them and treat them specially, e.g. [qiskit-phase-propagator](../../qiskit-phase-propagator/) applies the phase of a monomial of power up to 3 exactly with controlled phase gates.

[`effective_alpha`][python_signals.algebraic_signal.PolynomialSignal.effective_alpha] is the coefficient in terms of the integer indices rather than the values: since `values = index * period`,

```text
alpha * values**power == (alpha * period**power) * index**power == effective_alpha * index**power
```

which is what an encoding of the integer indices in qubits needs.

```python
import numpy as np
from python_signals.algebraic_signal import QuadraticSignal
from python_signals.integer_axis import IndexOrdering
from python_signals.physical_axis import PositionAxis

axis = PositionAxis(16, 0.1, IndexOrdering.FFT)
lens = QuadraticSignal(axis, alpha=-0.5)

assert lens.power == 2
assert np.isclose(lens.effective_alpha, -0.5 * 0.1**2)
assert np.allclose(lens.data, lens.effective_alpha * axis.index**2)
```

Polynomial signals have no SymPy `expression`; create them with `from_sympy` where the expression is needed.

## Functions of either kind

[`SampledSignal`][python_signals.algebraic_signal.SampledSignal] is the type `Signal | AlgebraicSignal`, for code that only needs the `axis` and the sampled values `data`, which both kinds provide. `isinstance(value, SampledSignal)` works as well, since it is a union of classes.

```python
import numpy as np
from python_signals.algebraic_signal import AlgebraicSignal, SampledSignal
from python_signals.integer_axis import IndexOrdering
from python_signals.physical_axis import PositionAxis
from python_signals.signal import Signal


def mean_value(signal: SampledSignal) -> float:
    return float(np.mean(signal.data))


axis = PositionAxis(4, 1.0, IndexOrdering.NATURAL)
assert mean_value(Signal(axis, [1.0, 2.0, 3.0, 4.0])) == 2.5
assert mean_value(AlgebraicSignal(axis, lambda x: x + 1)) == 2.5
assert isinstance(Signal(axis, np.zeros(4)), SampledSignal)
```

## Arithmetic

Signals support `+`, `-`, `*`, `/`, `**` and negation, elementwise, with scalars from either side and with signals on an equal axis. The result is always a new object:

| operands                                          | result                                                                                                                |
| ------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `Signal` and scalar or `Signal`                   | a `Signal` of the combined samples                                                                                    |
| `AlgebraicSignal` and scalar or `AlgebraicSignal` | an `AlgebraicSignal` of the composed functions, and of the composed SymPy expressions if both operands have one        |
| `AlgebraicSignal` and `Signal`                    | a `Signal`, sampling the algebraic operand first                                                                      |
| `PolynomialSignal` `*` or `/` scalar              | a `PolynomialSignal` (or `QuadraticSignal`) with the scaled `alpha`, so `effective_alpha` stays available              |

The rules in detail:

- **Scalars** are instances of `numbers.Number`: Python and NumPy numbers, real or complex, including `bool`. A zero-dimensional array is not a scalar; convert it with `.item()`.
- **Arrays are rejected.** A raw NumPy array, or a list, raises a `TypeError` from either side, rather than being broadcast silently against samples of possibly another ordering. Wrap it in a `Signal` on the intended axis instead.
- **Axes must be equal**, by value (see above), otherwise a `ValueError` is raised. Axes created separately with the same attributes are equal, but the periods are compared exactly: a spacing computed as `2 * np.pi / (N * dx)` by hand may differ in the last bit from the one of `from_position_axis`. Derive axes from each other, or share one axis object.
- **Algebraic results stay algebraic**: they can still be evaluated off the axis, by calling them. A combination with a sampled `Signal` cannot, so it is sampled.
- **Expressions** are combined only if both operands have one, a scalar counting as one: `2 * symbolic + 1` keeps the expression `2 * f(x) + 1`, but combining with a function-based signal drops it. If the operands use different symbols, the other operand's symbol is replaced by the one of the left operand.
- **Monomials stay monomials** under `*` by a scalar from either side, `/` by a scalar and negation. Every other operation on a `PolynomialSignal`, e.g. `+ 1`, `scalar / signal` or the product of two monomials, gives a general `AlgebraicSignal`.
- `+signal` returns the signal itself.

```python
import numpy as np
import sympy
from python_signals.algebraic_signal import AlgebraicSignal, QuadraticSignal
from python_signals.integer_axis import IndexOrdering
from python_signals.physical_axis import PositionAxis
from python_signals.signal import Signal

axis = PositionAxis(8, 0.5, IndexOrdering.FFT)
x, t = sympy.symbols("x t")
square = AlgebraicSignal.from_sympy(axis, x**2)

assert (2 * square + 1).expression == 2 * x**2 + 1
assert (square + AlgebraicSignal.from_sympy(axis, t)).expression == x**2 + x
assert (square + AlgebraicSignal(axis, np.cos)).expression is None
assert type(square + Signal(axis, np.ones(8))) is Signal

lens = QuadraticSignal(axis, alpha=0.3)
assert type(-2 * lens) is QuadraticSignal
assert type(lens + 1) is AlgebraicSignal

try:
    square * np.ones(8)
except TypeError:
    pass
else:
    raise AssertionError("arrays are rejected")
assert np.allclose((square * Signal(axis, np.ones(8))).data, axis.values**2)
```

## Pitfalls

- **Mind the ordering when handing samples to other code.** The values of an `FFT`-ordered axis are not ascending, they jump from the largest to the most negative one halfway. `numpy.gradient`, `numpy.trapezoid` and line plots expect ascending positions, so use a `NATURAL` or `CENTERED` axis there, or reorder the samples with `numpy.fft.fftshift`.
- **Axes are plain objects.** Their attributes can be reassigned, but do not: an algebraic signal caches its `data` on first access, and the equality and hash of an axis change with its attributes.
- **Integer data stays integer** in `Signal`, e.g. `Signal(axis, [1, 2, 3])`, until an operation promotes it, as in NumPy; `/` gives floats, while `-` and `*` by integers keep integers.
- **`effective_alpha` depends on the axis**, through its `period`: the same `alpha` on two axes of different spacings gives different coefficients of the indices.
