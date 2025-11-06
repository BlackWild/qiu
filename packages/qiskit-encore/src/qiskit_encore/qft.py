from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library.basis_change import QFTGate
from qiskit.quantum_info import Operator
from qiskit.synthesis.qft import synth_qft_full


def generate_whole_qft_circuit(n, inverse: bool = False) -> QuantumCircuit:
    return synth_qft_full(n, inverse=inverse)


def generate_big_matrix_qft_circuit(n, inverse: bool = False) -> QuantumCircuit:
    circ = QuantumCircuit(n)
    qft_gate = QFTGate(n)
    qft_op = Operator(qft_gate)
    if inverse:
        circ.append(qft_op.adjoint(), circ.qubits)
    else:
        circ.append(qft_op, circ.qubits)
    return circ
