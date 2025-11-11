from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Operator


def big_matrix_circ(circuit: QuantumCircuit):
    op = Operator(circuit)
    circ = QuantumCircuit(op.num_qubits)
    circ.append(op, circ.qubits)
    return circ
