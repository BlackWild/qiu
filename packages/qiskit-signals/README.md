# Qiskit Signals

> **Legacy.** No package of this monorepo depends on `qiskit-signals` anymore: `qiskit-phase-propagator`, `qiskit-hamiltonian-simulation` and the test helpers work with [`python-signals`](../../shareable-packages/python-signals/README.md) directly. It is kept, unchanged in behavior, only because the applications in `apps/` (including the code of the first submission and its cluster runs) build their axes and signals with it. Its classes subclass the `python-signals` ones, so they can be passed to the refactored packages as they are. Delete it once the applications are migrated. The mapping to migrate is:
>
> | `qiskit-signals`                                      | `python-signals`                                             |
> | ----------------------------------------------------- | ------------------------------------------------------------ |
> | `PositionAxis(num_qubits, delta_x, encoding)`         | `PositionAxis(2**num_qubits, delta_x, encoding.index_ordering)` |
> | `EncodingType.UNSIGNED` / `TWOS_COMPLEMENT` / `..._MIRRORED` | `IndexOrdering.NATURAL` / `FFT` / `CENTERED`           |
> | `GenericQuantumSignal(axis, signal_function)`          | `AlgebraicSignal(axis, function)`                            |
> | `QuadraticQuantumSignal(axis, alpha)`                  | `QuadraticSignal(axis, alpha)`                               |
> | `ArbitrarySignalForSampleBasedProtocol(...).alpha`, `.statevector` | `qiskit_phase_propagator.sample_based.sample_based_decomposition(signal)` |
> | `signal.num_qubits`                                    | `qiskit_phase_propagator.qubit_encoding.num_qubits_of(signal.axis)` |

Signals for quantum applications: the axes and signals of [`python-signals`](../../shareable-packages/python-signals/README.md), sampled on `2**num_qubits` points and encoded in the computational basis states of a qubit register.

## Concepts

### Encodings

An `EncodingType` fixes which integer index each computational basis state represents, and thus the index ordering of the underlying `python-signals` axis:

| encoding                   | indices for 2 qubits | index ordering |
| -------------------------- | -------------------- | -------------- |
| `UNSIGNED`                 | `0, 1, 2, 3`         | `NATURAL`      |
| `TWOS_COMPLEMENT`          | `0, 1, -2, -1`       | `FFT`          |
| `TWOS_COMPLEMENT_MIRRORED` | `-2, -1, 0, 1`       | `CENTERED`     |

### Quantum axes

`quantum_axis.GenericAxis` is a `python_signals.physical_axis.PhysicalAxis` of `2**num_qubits` samples, constructed from the number of qubits and an encoding instead of a size and an ordering. The concrete axes `PositionAxis`, `MomentumAxis`, `AngularWavenumberAxis` and `SpatialFrequencyAxis` mirror the ones of `python-signals`. Next to the generic attributes, quantum axes provide `num_qubits`, `encoding`, `dimension`, `axis_values`, `axis_type` and `is_fourier_domain_axis`. The axis types (`helper_types.AxisType`) are the domains of `python-signals` (`AxisDomain`).

### Quantum signals

- `quantum_signal.GenericQuantumSignal`, `PolynomialQuantumSignal` and `QuadraticQuantumSignal` are the `AlgebraicSignal`, `PolynomialSignal` and `QuadraticSignal` of `python-signals` on quantum axes, providing `num_qubits` and the `encoding` in addition.
- `sample_based_signal.ArbitrarySignalForSampleBasedProtocol` is a non-negative (or non-positive) signal prepared as the statevector of its normalized square root, for sample-based Hamiltonian simulation.

## Usage

```python
from qiskit_signals.helper_types import EncodingType
from qiskit_signals.quantum_axis import PositionAxis
from qiskit_signals.quantum_signal import QuadraticQuantumSignal

x_axis = PositionAxis(num_qubits=4, delta_x=0.1, encoding=EncodingType.TWOS_COMPLEMENT)
lens = QuadraticQuantumSignal(axis=x_axis, alpha=-0.5)

# the coefficient of the phase circuit acting on the integers of the basis states
lens.effective_alpha
```

## Tests

From the repository root:

```sh
uv run pytest packages/qiskit-signals
```
