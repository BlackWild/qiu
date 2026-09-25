# qiu-signals

`qiu-signals` provides uniformly sampled axes and signals on them, for any kind of numerical application, classical or quantum. An axis fixes the samples: their number, their spacing, the physical domain they live in (position or one of its Fourier conjugates) and the ordering of their integer indices. A signal is a function on such an axis, given either by its sampled values or by an algebraic expression, and supports elementwise arithmetic. It depends on NumPy and [qiu-python-encore](../qiu-python-encore/index.md), and optionally on SymPy for symbolic expressions.

In this monorepo, the signals are the common language of the classical and the quantum numerics: the quantum packages, e.g. [qiu-quantum-computing](../qiu-quantum-computing/index.md) and [qiu-hamiltonian-simulation](../qiu-hamiltonian-simulation/index.md), encode them in qubit registers, the `2**n` samples of an axis being the basis states of `n` qubits whose bits encode the integer indices, and [qiu-classical-simulation](../qiu-classical-simulation/index.md) builds its optical elements on them.

## Installation

```sh
pip install qiu-signals
pip install "qiu-signals[sympy]"  # with SymPy, for AlgebraicSignal.from_sympy
```

Inside the monorepo, it is installed with all other packages by `uv sync --all-packages`.

## Quick start

```python
import numpy as np
from qiu_signals.algebraic_signal import AlgebraicSignal
from qiu_signals.integer_axis import IndexOrdering
from qiu_signals.physical_axis import PositionAxis, SpatialFrequencyAxis
from qiu_signals.signal import Signal

# 256 positions spaced by 0.1, in the ordering of numpy.fft
x_axis = PositionAxis(size=256, delta_x=0.1, ordering=IndexOrdering.FFT)
gaussian = AlgebraicSignal(x_axis, lambda x: np.exp(-(x**2)))

# its spectrum lives on the conjugate axis, whose values are those of numpy.fft.fftfreq
f_axis = SpatialFrequencyAxis.from_position_axis(x_axis)
spectrum = Signal(f_axis, np.fft.fft(gaussian.data))
assert np.allclose(f_axis.values, np.fft.fftfreq(256, d=0.1))

# elementwise arithmetic with scalars and with signals on an equal axis
beam = 2 * gaussian + 1
assert np.allclose(beam.data, 2 * np.exp(-(x_axis.values**2)) + 1)
assert np.isclose(np.linalg.norm(spectrum.normalized_data), 1)
```

## Where next

- The [User Guide](user-guide.md) explains the axes, their orderings and domains, the two kinds of signals and their arithmetic.
- The [Examples](examples.md) work through Fourier pairs, the free spreading of a wave packet, symbolic signals and signal arithmetic.
- The [API Reference](reference/qiu_signals/index.md) documents every public object, generated from the docstrings.
