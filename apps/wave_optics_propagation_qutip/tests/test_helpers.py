import numpy as np
import qutip as qt
from hypothesis import given
from qiskit.quantum_info import Statevector
from qiskit_pytest_helper.constants import FIDELITY_TOLERANCE
from qiskit_pytest_helper.hypothesis_strategies import valid_qiskit_statevector
from wave_optics_propagation_qutip.helpers import householder_unitary


class TestHouseholderUnitary:
    @given(valid_qiskit_statevector())
    def test_householder_based_initializer(self, statevector: Statevector) -> None:
        """Test the householder_unitary function."""

        N = statevector.dim
        ket0 = qt.basis(N, 0)

        target_state = qt.Qobj(statevector.data)

        U = householder_unitary(statevector)
        generated_state = U @ ket0

        fidelity = qt.fidelity(generated_state, target_state)
        assert fidelity >= 1.0 - FIDELITY_TOLERANCE
