# python-pytest-helper

Shared helpers for the unit tests of this monorepo that need no quantum computing: the floating-point comparisons of numbers and arrays, and the [Hypothesis](https://hypothesis.readthedocs.io/) strategies of numbers, axes and signals. It only depends on NumPy, Hypothesis and [`qiu-signals`](../qiu-signals/).

The comparisons replace tolerances chosen per test by one rule: values agree relative to their magnitude, down to the smallest normal float, below which they only agree absolutely. The strategies generate finite, bounded values that include zero and subnormal numbers, and compose: axis strategies take strategies of their sizes, spacings and orderings, and signal strategies take a strategy of their axes.

The package is used by the tests of `qiu-signals`, `qiu-classical-simulation` and the quantum packages, and [`qiskit-pytest-helper`](../qiskit-pytest-helper/) builds its quantum strategies and assertions on it. It is a development package of the workspace, not meant to be published.

## Installation

The package is not published on PyPI. It is a member of the uv workspace and part of the `test` dependency group of the repository root, so it is installed from the repository root by

```sh
uv sync --all-packages
```

A package of the workspace whose tests use it declares it in its `test` dependency group, taken from the workspace, in its `pyproject.toml`:

```toml
[dependency-groups]
test = [
    "hypothesis>=6.140.3",
    "pytest>=9.0.0",
    "python-pytest-helper",
]

[tool.uv.sources]
python-pytest-helper = { workspace = true }
```

## Quick start

A property-based test of the `FFT` index ordering of `qiu-signals` against NumPy, on integer axes of 1 to 64 samples:

```python
import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from python_pytest_helper.assertions import assert_close
from python_pytest_helper.hypothesis_strategies import integer_axes
from qiu_signals.integer_axis import IndexOrdering


@given(axis=integer_axes(orderings=st.just(IndexOrdering.FFT)))
def test_fft_indices_are_numpy_frequencies(axis):
    """Test that the FFT indices are the frequencies of numpy.fft.fftfreq."""
    assert_close(axis.index, np.fft.fftfreq(axis.size) * axis.size)


test_fft_indices_are_numpy_frequencies()
```

## Where next

- The [User Guide](user-guide.md) explains the comparison rules (the relative tolerance, the underflow floor and `scale`, the norm-wise comparison) and every strategy.
- The [Examples](examples.md) work through property-based tests of axes, signals and FFTs.
- The [API Reference](reference/python_pytest_helper/index.md) documents the modules [`assertions`](reference/python_pytest_helper/assertions.md) and [`hypothesis_strategies`](reference/python_pytest_helper/hypothesis_strategies.md).
