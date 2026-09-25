# Examples

Each example defines a property-based test as it would appear in a test module, and calls it directly, which runs Hypothesis on it as pytest would.

## Conjugate axes against NumPy

The momentum axis conjugate to a position axis must sample the momenta of the discrete Fourier transform, `2 pi hbar numpy.fft.fftfreq(N, d=delta_x)`, in the `FFT` ordering. The test draws position axes of any ordering and powers of 2 as sizes, and `hbar` from a bounded range:

```python
import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from python_pytest_helper.assertions import assert_close
from python_pytest_helper.hypothesis_strategies import position_axes, power_of_two_sizes
from python_signals.integer_axis import IndexOrdering
from python_signals.physical_axis import MomentumAxis, PositionAxis


@given(
    x_axis=position_axes(sizes=power_of_two_sizes()),
    hbar=st.floats(min_value=0.1, max_value=10.0),
)
def test_momentum_axis_samples_the_dft(x_axis: PositionAxis, hbar: float):
    """Test that the conjugate momenta are those of numpy.fft.fftfreq."""
    p_axis = MomentumAxis.from_position_axis(x_axis, hbar=hbar)

    assert p_axis.ordering is IndexOrdering.FFT
    assert_close(
        p_axis.values, 2 * np.pi * hbar * np.fft.fftfreq(x_axis.size, d=x_axis.period)
    )


test_momentum_axis_samples_the_dft()
```

The momenta agree with NumPy's up to rounding, relative to their magnitude, for every size, spacing and ordering Hypothesis draws, with no tolerance chosen for the test.

## Monomials on momentum axes

`PolynomialSignal.effective_alpha` is the coefficient in terms of the integer indices, `data == effective_alpha * index**power`. The composed strategy draws monomials on `FFT`-ordered momentum axes. Since `alpha` may be subnormal, `effective_alpha = alpha * period**power` may underflow before it is multiplied by `index**power`, so that factor is passed as `scale`:

```python
import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from python_pytest_helper.assertions import assert_close
from python_pytest_helper.hypothesis_strategies import (
    monomial_signals,
    physical_axes,
    power_of_two_sizes,
)
from python_signals.algebraic_signal import PolynomialSignal
from python_signals.integer_axis import IndexOrdering
from python_signals.physical_axis import AxisDomain

momentum_monomials = monomial_signals(
    physical_axes(
        AxisDomain.MOMENTUM,
        sizes=power_of_two_sizes(),
        orderings=st.just(IndexOrdering.FFT),
    ),
    powers=st.integers(min_value=0, max_value=4),
)


@given(signal=momentum_monomials)
def test_effective_alpha(signal: PolynomialSignal):
    """Test that the monomial is effective_alpha times the powers of the indices."""
    index = signal.axis.index
    scale = float(np.max(np.abs(index))) ** signal.power

    assert signal.axis.is_fourier_domain
    assert_close(signal.data, signal.effective_alpha * index**signal.power, scale=scale)


test_effective_alpha()
```

The comparison is relative for all normal values, and absolute at the scale `tiny * max|index|^power` only where the coefficient could have underflowed.

## Normalizing signals with zero and subnormal samples

`Signal.normalized_data` scales the samples to unit Euclidean norm, first by a power of 2 so that the sum of squares neither underflows nor overflows. The generated samples include zeros and subnormal numbers, the hard cases for a naive `data / numpy.linalg.norm(data)`; the all-zero signal, which is returned unchanged, is excluded with `assume`:

```python
import numpy as np
from hypothesis import assume, given
from python_pytest_helper.assertions import assert_close
from python_pytest_helper.hypothesis_strategies import sampled_signals
from python_signals.signal import Signal


@given(signal=sampled_signals(dtype=np.complex128))
def test_normalized_data_has_unit_norm(signal: Signal):
    """Test that the normalized samples have unit norm and the original directions."""
    assume(np.any(signal.data != 0))
    normalized = signal.normalized_data

    assert_close(np.linalg.norm(normalized), 1.0)
    largest = np.argmax(np.abs(signal.data))
    assert_close(
        np.exp(1j * np.angle(normalized[largest])),
        np.exp(1j * np.angle(signal.data[largest])),
    )


test_normalized_data_has_unit_norm()
```

The norm is 1 up to rounding for all non-zero signals, and the largest sample keeps its complex phase; comparing the phase of the largest sample rather than of every sample avoids the entries that underflowed in the normalization.

## FFT round trips, compared in norm

The orthonormal FFT and its inverse are a round trip, but they compute every entry from all of them, so an entry much smaller than the others carries an error relative to the norm of the vector. The test compares the norms elementwise, as scalars, and the round trip with `assert_close_in_norm`:

```python
import numpy as np
from hypothesis import given
from python_pytest_helper.assertions import assert_close, assert_close_in_norm
from python_pytest_helper.hypothesis_strategies import (
    physical_axes,
    power_of_two_sizes,
    sampled_signals,
)
from python_signals.signal import Signal

signals = sampled_signals(
    physical_axes(sizes=power_of_two_sizes()), dtype=np.complex128
)


@given(signal=signals)
def test_orthonormal_fft(signal: Signal):
    """Test Parseval's theorem and the round trip of the orthonormal FFT."""
    spectrum = np.fft.fft(signal.data, norm="ortho")

    assert_close(np.linalg.norm(spectrum), np.linalg.norm(signal.data))
    assert_close_in_norm(np.fft.ifft(spectrum, norm="ortho"), signal.data)


test_orthonormal_fft()
```

The test passes for all generated signals. With `assert_close` on the round trip instead, Hypothesis finds counterexamples such as `[1.6e-46j, 1j]`, whose first entry comes back as 0: an absolute error far below the rounding relative to the norm, but far above the underflow floor, and a relative error of 1 for that entry.
