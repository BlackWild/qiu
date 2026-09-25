"""The ways a circuit building block can be represented in a quantum circuit."""

from qiu_python_encore.enum import ExtendedEnum


class SynthesisMethod(ExtendedEnum):
    """How a circuit building block is represented in the circuit it creates.

    - `DENSE`: a single unitary gate defined by its dense matrix. Exact and
      convenient for small numbers of qubits, but the matrix has `4**n` entries.
    - `GATE`: a single high-level Qiskit gate describing the operation, leaving its
      synthesis into elementary gates to Qiskit, e.g. when transpiling.
    - `DECOMPOSED`: an already decomposed circuit of elementary gates, synthesized
      by this package where Qiskit's synthesis is not suitable.
    """

    DENSE = "dense"
    GATE = "gate"
    DECOMPOSED = "decomposed"
