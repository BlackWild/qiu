# User Guide

`qiu-qiskit-encore` improves native types of Qiskit, in two modules:

| module                                                   | contents                                                      |
| -------------------------------------------------------- | ------------------------------------------------------------- |
| [`statevector`][qiu_qiskit_encore.statevector]           | statevectors validated as normalized states of qubits         |
| [`synthesis_method`][qiu_qiskit_encore.synthesis_method] | the ways a circuit building block is represented in a circuit |

The circuit building blocks using them, e.g. the state preparation, the QFT and the phase operators, are in [qiu-quantum-computing](../../qiu-quantum-computing/).

## Validated statevectors

A Qiskit `Statevector` accepts amplitudes of any dimension, normalized or not, and leaves it to `Statevector.is_valid` to check them. [`validated_statevector(state)`][qiu_qiskit_encore.statevector.validated_statevector] accepts a Qiskit `Statevector` or anything NumPy converts to an array of amplitudes, returns a copy as a complex `Statevector`, and raises a `ValueError` unless:

- the dimension is a power of 2 of at least one qubit, i.e. `2, 4, 8, ...`, and
- the state is normalized, as checked by `Statevector.is_valid`, i.e. up to Qiskit's tolerances `Statevector.atol` and `Statevector.rtol`.

```python
import numpy as np
from qiskit.quantum_info import Statevector
from qiu_qiskit_encore.statevector import validated_statevector

amplitudes = np.array([0.6, 0.8j])
statevector = validated_statevector(amplitudes)
assert isinstance(statevector, Statevector)
assert statevector.data is not amplitudes  # a copy

# Qiskit accepts both, the validation rejects them
for invalid in ([1.0, 1.0], [1.0, 0.0, 0.0]):
    Statevector(invalid)  # no error
    try:
        validated_statevector(invalid)
    except ValueError as error:
        print(error)  # not normalized, or not a state of qubits
    else:
        raise AssertionError("expected a ValueError")
```

States are never normalized silently: normalize them yourself, e.g. with `state / numpy.linalg.norm(state)`.

## Synthesis methods

[`SynthesisMethod`][qiu_qiskit_encore.synthesis_method.SynthesisMethod] chooses how a circuit building block, e.g. the state preparation or the QFT of [qiu-quantum-computing](../../qiu-quantum-computing/), is represented in the circuit it creates, by the `method` parameter of the function creating it:

| method       | the circuit contains                                                    | use it for                                                                     |
| ------------ | ----------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| `DENSE`      | a single unitary gate defined by its dense matrix                       | exact reference results on few qubits (the matrix has `4**n` entries)         |
| `GATE`       | a single high-level Qiskit gate, synthesized by Qiskit when transpiling | the default: leaving the synthesis to Qiskit, e.g. for hardware-aware transpiling |
| `DECOMPOSED` | an already decomposed circuit of elementary gates                       | a synthesis we control, where Qiskit's is not suitable                         |

Some guidelines for the choice:

- `DENSE` is the most direct representation: simulators apply the matrix as it is, which makes it the reference for tests and small simulations. Its memory grows as `4**n`, and transpiling it into elementary gates falls back on Qiskit's generic unitary synthesis, which yields far more CNOT gates than the other methods.
- `GATE` keeps the circuit abstract until it is transpiled, so Qiskit can choose a synthesis for the target, and the circuit stays small and readable. Its quality is the quality of Qiskit's synthesis, including its bugs.
- `DECOMPOSED` gives a circuit of elementary gates of known structure and gate counts. For state preparation, it is synthesized by qiu-quantum-computing, independently of the Qiskit version, and is the choice wherever Qiskit's synthesis fails; for the QFT, it is Qiskit's standard textbook circuit.

All three methods are exact: they implement the same unitary, up to floating-point rounding, and differ only in their representation. For approximate state preparation with shallower circuits, see [qiu-mps-initializer](../../qiu-mps-initializer/).

`SynthesisMethod` is an `ExtendedEnum` of [qiu-python-encore](../../qiu-python-encore/), whose members compare equal to their raw values. Every function of the monorepo taking a `method` also accepts the raw values `"dense"`, `"gate"` and `"decomposed"`, and converts them to the members:

```python
from qiu_qiskit_encore.synthesis_method import SynthesisMethod

assert SynthesisMethod("dense") is SynthesisMethod.DENSE
assert SynthesisMethod.GATE == "gate"
assert SynthesisMethod.list() == ["dense", "gate", "decomposed"]
```

A method a function does not implement raises a `NotImplementedError`, so a method added to the enum in the future fails loudly until it is implemented everywhere.

## Summary of the pitfalls

- States must be normalized and of a power-of-2 dimension; `validated_statevector` does not normalize them for you.
- A validated statevector is a copy: changing the given array afterwards does not change it.
- `SynthesisMethod` members compare equal to their raw values, so `method == "gate"` holds for `SynthesisMethod.GATE`; convert unknown input with `SynthesisMethod(method)`, which raises a `ValueError` for values that are no method.
