# User Guide

The package has two modules:

| module                  | contents                                                                        |
| ----------------------- | ------------------------------------------------------------------------------- |
| `assertions`            | the floating-point comparisons of numbers and arrays                            |
| `hypothesis_strategies` | Hypothesis strategies of numbers, axes and signals of [`python-signals`](../../python-signals/) |

Quantum states and operators are compared with Qiskit's equality instead, and the strategies of quantum states and qubit axes live in [`qiskit-pytest-helper`](../../qiskit-pytest-helper/).

## Floating-point comparisons

[`assert_close(actual, expected, scale=1.0)`][python_pytest_helper.assertions.assert_close] asserts that numbers or arrays agree elementwise up to rounding, and [`is_close`][python_pytest_helper.assertions.is_close] is the same comparison as a predicate, e.g. to exclude values in strategies or with `hypothesis.assume`. The tests of the monorepo compare with these instead of tolerances chosen per test. Both are thin wrappers of NumPy's comparisons, `numpy.testing.assert_allclose` and `numpy.allclose`, with the tolerances below.

### Relative tolerance

Values are compared relative to their own magnitude, `|actual - expected| <= RTOL |expected| + atol`, with [`RTOL`][python_pytest_helper.assertions.RTOL] `= 1e-7`, the default of `numpy.testing.assert_allclose`. It covers the rounding errors of the few operations a test compares, at every normal scale: `1e300` and `1e-300` are compared alike, and `1e-300` is not close to `0` or to `2e-300`.

Complex values are compared by the magnitude of their difference, so `1j` is not close to `-1j`. The `expected` values are broadcast to the `actual` ones, e.g. a scalar to an array. Exact types, e.g. integer arrays, are compared as `float64`.

### The underflow floor

Below the smallest normal float, `numpy.finfo(dtype).tiny` (about `2.2e-308` for `float64`), floats lose their relative precision: they underflow to subnormal numbers or to zero. Such values only agree absolutely, at the scale of `tiny`. The absolute tolerance `atol` is therefore [`underflow_atol(dtype, scale)`][python_pytest_helper.assertions.underflow_atol], the `tiny` of the compared type, e.g. about `1.2e-38` for `float32` and `complex64`. So `0.0` is close to the smallest subnormal number `5e-324`, while all normal values are still compared relatively.

### Scaled values

If the compared values were scaled up by a factor after they could underflow, e.g. normalized values multiplied back by their norm, their absolute error is `tiny` times that factor. Pass the factor as `scale`: the absolute tolerance becomes `tiny * max(1, |scale|)`. Scaling down, `|scale| < 1`, keeps the tolerance `tiny`, since the result itself can still underflow.

```python
import numpy as np
from python_pytest_helper.assertions import is_close

x = 1e-300
underflowed = x * 1e-20  # a subnormal number, with only a few significant bits
restored = underflowed * 1e20

assert not is_close(restored, x)
assert is_close(restored, x, scale=1e20)
```

### Quantities that are ideally zero

Rounding errors are relative to the magnitude of the computation, not of its result. A quantity that is ideally 0, e.g. the imaginary part of a real projection or a difference of equal sums, carries an error of the order of the terms it is computed from, which no relative tolerance of the result accepts. Compare the whole quantity instead, e.g. the projection with its expected magnitude, or the two sums with each other.

### Norm-wise comparison

Vectors computed as a whole, e.g. by FFTs, matrix products or unitary evolutions, have rounding errors relative to their norm rather than to each entry, so entries near 0 lose their relative precision. [`assert_close_in_norm(actual, expected)`][python_pytest_helper.assertions.assert_close_in_norm] compares such vectors in the Euclidean norm,

```text
||actual - expected|| <= RTOL ||expected|| + tiny,
```

and requires both to have the same shape. Two zero vectors agree. For example, the FFT round trip of `[1, 1e-20]` loses the second entry entirely, which is an error of `1e-20` relative to the norm, but of 100 % relative to the entry:

```python
import numpy as np
from python_pytest_helper.assertions import assert_close_in_norm, is_close

vector = np.array([1.0, 1e-20])
round_trip = np.fft.ifft(np.fft.fft(vector, norm="ortho"), norm="ortho")

assert not is_close(round_trip, vector)
assert_close_in_norm(round_trip, vector)
```

## Hypothesis strategies

`hypothesis_strategies` holds strategies of the objects the tests of the monorepo are about. They are plain Hypothesis strategies: use them with `hypothesis.given`, draw from them with `st.data()`, or `map` and `filter` them.

### Numbers

Numbers are finite and bounded in magnitude by [`MAX_MAGNITUDE`][python_pytest_helper.hypothesis_strategies.MAX_MAGNITUDE] `= 1e3` by default, far enough from the largest float, about `1e308`, that sums, products and powers up to about 100 of them stay finite: tests never see overflows by accident. They include zero and subnormal numbers, whose loss of precision the comparisons above account for.

- [`reals(max_magnitude)`][python_pytest_helper.hypothesis_strategies.reals]: real numbers in `[-max_magnitude, max_magnitude]`.
- [`complexes(max_magnitude)`][python_pytest_helper.hypothesis_strategies.complexes]: complex numbers of magnitude at most `max_magnitude`.
- [`sample_arrays(size, dtype, max_magnitude)`][python_pytest_helper.hypothesis_strategies.sample_arrays]: one-dimensional arrays of `size` such numbers, complex for a complex `dtype` and real otherwise.

### Axes

The axis strategies take strategies of their parameters, with defaults:

| parameter strategy                                   | generates                                             | default of the axes              |
| ---------------------------------------------------- | ----------------------------------------------------- | -------------------------------- |
| [`axis_sizes(min_size, max_size)`][python_pytest_helper.hypothesis_strategies.axis_sizes] | numbers of samples                          | `axis_sizes()`, 1 to 64           |
| [`power_of_two_sizes(min_exponent, max_exponent)`][python_pytest_helper.hypothesis_strategies.power_of_two_sizes] | sizes `2**n`, e.g. for FFTs or `n` qubits | not used by default; 1 to 64 |
| [`axis_spacings(min_value, max_value)`][python_pytest_helper.hypothesis_strategies.axis_spacings] | positive sampling periods of position axes | `axis_spacings()`, `1e-3` to `10` |
| [`index_orderings`][python_pytest_helper.hypothesis_strategies.index_orderings] | all index orderings                              | all orderings                     |
| [`axis_domains`][python_pytest_helper.hypothesis_strategies.axis_domains] | all axis domains                                       | not used by the axis strategies   |

- [`integer_axes(sizes, orderings)`][python_pytest_helper.hypothesis_strategies.integer_axes]: `IntegerAxis` objects.
- [`position_axes(sizes, spacings, orderings)`][python_pytest_helper.hypothesis_strategies.position_axes]: `PositionAxis` objects.
- [`physical_axes(domain, sizes, spacings, orderings, hbar)`][python_pytest_helper.hypothesis_strategies.physical_axes]: axes of a domain. Position axes are those of `position_axes`. Fourier axes are conjugate to such a position axis, created by its `from_position_axis` with `keep_ordering=True`: the `spacings` are those of the position axis, the spacing of the Fourier axis is its reciprocal period, e.g. `2 pi hbar / (N delta_x)` for momentum with the given `hbar` (default 1), and the ordering is the drawn one, not necessarily `FFT`.

### Signals

The signal strategies take a strategy of their `axes`, by default `physical_axes()`, i.e. position axes:

- [`sampled_signals(axes, dtype, max_magnitude)`][python_pytest_helper.hypothesis_strategies.sampled_signals]: a `Signal` of finite sampled values of bounded magnitude, from `sample_arrays`.
- [`monomial_signals(axes, alphas, powers)`][python_pytest_helper.hypothesis_strategies.monomial_signals]: a `PolynomialSignal` `alpha x^power`, by default with `alpha` in `[-10, 10]` and `power` from 0 to 4.
- [`positive_polynomial_signals(axes, max_degree, coefficients, total)`][python_pytest_helper.hypothesis_strategies.positive_polynomial_signals]: an `AlgebraicSignal` of the absolute value of a polynomial of degree 1 to `max_degree` (default 5), with coefficients in `[0.01, 1]` by default, scaled such that its samples sum up to `total` (default 1). Its samples are non-negative and finite, e.g. for the sample-based phases of `qiskit-phase-propagator`, where the sum is the total phase.

### Composing strategies

The strategies compose, e.g. monomials of power 2 on momentum axes in the `FFT` ordering, whose sizes are powers of 2:

```python
from hypothesis import strategies as st
from python_pytest_helper.hypothesis_strategies import (
    monomial_signals,
    physical_axes,
    power_of_two_sizes,
)
from python_signals.integer_axis import IndexOrdering
from python_signals.physical_axis import AxisDomain

kinetic_energies = monomial_signals(
    physical_axes(
        AxisDomain.MOMENTUM,
        sizes=power_of_two_sizes(),
        orderings=st.just(IndexOrdering.FFT),
    ),
    powers=st.just(2),
)
```

The strategies of `qiskit-pytest-helper` plug in the same way, e.g. `positive_polynomial_signals(qubit_axes(AxisDomain.POSITION), total=0.1)` for signals on axes of `2**n` samples with `n >= 1`.

## Writing tests

- Compare with `assert_close`, `is_close` or `assert_close_in_norm`, and pass `scale` where values were scaled up after they could underflow. A tolerance chosen for a test is a sign that the test compares a quantity that is ideally 0, or entries of a vector computed as a whole.
- Bound the inputs with the strategies' parameters rather than filtering, e.g. `axis_sizes(max_size=4)` or `reals(10.0)`, so that Hypothesis generates few invalid examples.
- Exclude degenerate inputs with `hypothesis.assume`, e.g. an all-zero signal where a normalization is tested.
- Tests that simulate circuits or run long computations set `@settings(max_examples=..., deadline=None)`. On CI, Hypothesis' built-in profile `ci` disables the deadline for all tests.

## Pitfalls

- The default sizes include a single sample, and `power_of_two_sizes()` includes `2**0 = 1`; pass `min_size` or `min_exponent` for axes that need several samples, e.g. `power_of_two_sizes(min_exponent=1)` for qubit registers.
- The default axes of the signal strategies are position axes; pass `physical_axes(domain)` for other domains, and `orderings=st.just(IndexOrdering.FFT)` where the samples must be in the order of `numpy.fft`.
- The numbers include subnormal values. A test that multiplies generated values by large factors after they could underflow must pass `scale` to the comparison, see the [Examples](examples.md#monomials-on-momentum-axes).
- `positive_polynomial_signals` can have samples arbitrarily close to 0, near the roots of the polynomial at negative axis values; only their sum is fixed.
