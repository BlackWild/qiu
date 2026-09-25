# Python Pytest Helper

[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)](https://www.python.org/) [![License: MIT](https://img.shields.io/badge/license-MIT-green)](https://github.com/BlackWild/qiu/blob/master/LICENSE) [![CI](https://github.com/BlackWild/qiu/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/BlackWild/qiu/actions/workflows/ci.yml) [![Docs](https://img.shields.io/badge/docs-mkdocs-blue)](https://blackwild.github.io/qiu/python-pytest-helper/)

Shared helpers for the unit tests of this monorepo that need no quantum computing, not meant to be published. It only depends on NumPy, Hypothesis and [`qiu-signals`](https://github.com/BlackWild/qiu/blob/master/packages/qiu-signals/README.md); the quantum helpers of [`qiskit-pytest-helper`](https://github.com/BlackWild/qiu/blob/master/packages-dev/qiskit-pytest-helper/README.md) build on it.

## Floating-point comparisons

`assertions.assert_close(actual, expected, scale=1.0)` is the comparison of numbers and arrays of the monorepo, with `is_close` as its predicate, and `assert_close_in_norm(actual, expected)` the one of vectors computed as a whole, e.g. by FFTs, whose rounding errors are relative to their norm:

- Values are compared relative to their magnitude, with the default relative tolerance of `numpy.testing.assert_allclose` (`RTOL`).
- Below the smallest normal float, `numpy.finfo(dtype).tiny`, floats lose their relative precision: they underflow to subnormal numbers or zero. Such values agree absolutely at that scale (`underflow_atol`).
- If the values were scaled by a factor after they could underflow, e.g. normalized values multiplied back by the norm, pass it as `scale`.

Tests compare with these instead of tolerances chosen per test. Quantum states and operators are compared with Qiskit's equality, see `qiskit_pytest_helper.assertions`.

## Hypothesis strategies

`hypothesis_strategies` generates:

- Numbers: `reals`, `complexes` and `sample_arrays`, finite and bounded by `MAX_MAGNITUDE` against overflows, including zero and subnormal numbers.
- Axes: `integer_axes`, `position_axes` and `physical_axes(domain)`, whose Fourier axes are conjugate to a position axis and keep its ordering. They take strategies of their `sizes` (`axis_sizes`, `power_of_two_sizes`), `spacings` (`axis_spacings`) and `orderings` (`index_orderings`).
- Signals on a strategy of `axes`: `sampled_signals`, `monomial_signals` and `positive_polynomial_signals`, which sum up to a given `total`.

The strategies compose, e.g. monomials of power 2 on momentum axes in the `FFT` ordering:

```python
from hypothesis import strategies as st
from python_pytest_helper.hypothesis_strategies import monomial_signals, physical_axes
from qiu_signals.integer_axis import IndexOrdering
from qiu_signals.physical_axis import AxisDomain

signals = monomial_signals(
    physical_axes(AxisDomain.MOMENTUM, orderings=st.just(IndexOrdering.FFT)),
    powers=st.just(2),
)
```

## Documentation

The documentation, with the API reference from the docstrings, is built from `docs/` with MkDocs and published at <https://blackwild.github.io/qiu/python-pytest-helper/>. To serve it locally, from the repository root:

```sh
uv run mkdocs serve -f packages-dev/python-pytest-helper/mkdocs.yml
```

## Tests

From the repository root:

```sh
uv run pytest packages-dev/python-pytest-helper
```
