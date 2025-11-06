"""Unit tests for time_independent.sample_based."""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from qiskit import transpile
from qiskit.circuit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Statevector, partial_trace, state_fidelity
from qiskit_aer_encore.simulator import generate_aer_simulator
from qiskit_hamiltonian_simulation.time_independent.sample_based import (
    KineticEvolutionSampleBased,
    PotentialEvolutionSampleBased,
)
from qiskit_pytest_helper.constants import (
    FIDELITY_TOLERANCE,
    MAX_NUM_OF_CYCLES,
    REDUCED_FIDELITY_TOLERANCE,
)
from qiskit_pytest_helper.hypothesis_strategies import (
    random_positive_signal,
)
from qiskit_signals.helper_types import AxisType, EncodingType
from qiskit_signals.sample_based_signal import ArbitrarySignalForSampleBasedProtocol

# TODO: very important, check how many times the initializers are called and optimize


class TestPotentialEvolutionSampleBased:
    """Tests for the PotentialEvolutionSampleBased circuit."""

    @settings(max_examples=50, deadline=None)
    @given(
        positive_potential=random_positive_signal(axis_type=AxisType.POSITION),
        coef=st.floats(0.1, 0.3),
        max_delta=st.floats(min_value=0.01, max_value=0.1),
    )
    def test_reasonable_num_of_iterations(
        self, positive_potential, coef: float, max_delta: float
    ):
        """Tests that the number of iterations is reasonable."""

        positive_potential = ArbitrarySignalForSampleBasedProtocol.from_generic_signal(
            positive_potential
        )
        hbar = 1.0
        t = coef

        propagator = PotentialEvolutionSampleBased(
            V=positive_potential,
            t=t,
            hbar=hbar,
            max_delta=max_delta,
        )

        assert propagator.num_of_cycles < MAX_NUM_OF_CYCLES

    @settings(max_examples=10, deadline=None)
    @given(
        positive_potential=random_positive_signal(axis_type=AxisType.POSITION),
        coef=st.floats(0.1, 0.3),
        max_delta=st.floats(min_value=0.01, max_value=0.1),
    )
    def test_correct_phase_application(
        self, positive_potential, coef: float, max_delta: float
    ):
        """Tests the PotentialEvolutionSampleBased circuit."""

        positive_potential = ArbitrarySignalForSampleBasedProtocol.from_generic_signal(
            positive_potential
        )
        num_qubits = positive_potential.num_qubits
        hbar = 1.0
        t = coef

        propagator = PotentialEvolutionSampleBased(
            V=positive_potential,
            t=t,
            hbar=hbar,
            max_delta=max_delta,
        )

        psi_reg = QuantumRegister(num_qubits, name=r"\psi")
        phi_reg = QuantumRegister(num_qubits, name=r"\phi")
        success_flag = ClassicalRegister(num_qubits, name="success_flag")
        circuit = QuantumCircuit(psi_reg, phi_reg, success_flag)

        psi = Statevector.from_label("+" * num_qubits)

        circuit.initialize(psi.data.tolist(), psi_reg)
        circuit.compose(propagator, circuit.qubits, circuit.clbits, inplace=True)
        circuit.save_statevector()  # type: ignore

        simulator = generate_aer_simulator()
        transpiled = transpile(circuit)
        job = simulator.run(transpiled, shots=1)
        result = job.result()
        counts: dict[int, int] = result.get_counts(circuit).int_outcomes()
        output_state_full = result.get_statevector(circuit)

        # Check that all measured qubits are 0
        assert counts[0] == 1

        # Check the output state
        traced = partial_trace(
            output_state_full,
            np.arange(num_qubits, 2 * num_qubits).tolist(),
        )
        output_state = traced.to_statevector()

        expected_output = np.exp(1.0j * -t / hbar * positive_potential.data) * psi.data

        assert (
            state_fidelity(output_state, Statevector(expected_output))
            >= 1.0 - FIDELITY_TOLERANCE
        )


class TestKineticEvolutionSampleBased:
    """Tests for the KineticEvolutionSampleBased circuit."""

    @settings(max_examples=50, deadline=None)
    @given(
        positive_kinetic=random_positive_signal(
            axis_type=AxisType.MOMENTUM, forced_encoding=EncodingType.TWOS_COMPLEMENT
        ),
        coef=st.floats(0.1, 0.3),
        max_delta=st.floats(min_value=0.01, max_value=0.1),
    )
    def test_reasonable_num_of_iterations(
        self, positive_kinetic, coef: float, max_delta: float
    ):
        """Tests that the number of iterations is reasonable."""

        positive_kinetic = ArbitrarySignalForSampleBasedProtocol.from_generic_signal(
            positive_kinetic
        )
        hbar = 1.0
        t = coef

        propagator = KineticEvolutionSampleBased(
            T=positive_kinetic,
            t=t,
            hbar=hbar,
            max_delta=max_delta,
        )

        assert propagator.num_of_cycles < MAX_NUM_OF_CYCLES

    @settings(max_examples=10, deadline=None)
    @given(
        positive_kinetic=random_positive_signal(
            axis_type=AxisType.MOMENTUM, forced_encoding=EncodingType.TWOS_COMPLEMENT
        ),
        coef=st.floats(0.1, 0.3),
        max_delta=st.floats(min_value=0.01, max_value=0.1),
    )
    def test_correct_phase_application(
        self, positive_kinetic, coef: float, max_delta: float
    ):
        """Tests the KineticEvolutionSampleBased circuit."""

        positive_kinetic = ArbitrarySignalForSampleBasedProtocol.from_generic_signal(
            positive_kinetic
        )
        hbar = 1.0
        t = coef

        propagator = KineticEvolutionSampleBased(
            T=positive_kinetic,
            t=t,
            hbar=hbar,
            max_delta=max_delta,
        )

        num_qubits = positive_kinetic.num_qubits
        psi_reg = QuantumRegister(num_qubits, name=r"\psi")
        phi_reg = QuantumRegister(num_qubits, name=r"\phi")
        success_flag = ClassicalRegister(num_qubits, name="success_flag")
        circuit = QuantumCircuit(psi_reg, phi_reg, success_flag)

        psi = Statevector.from_label("0" * num_qubits)

        circuit.initialize(psi.data.tolist(), psi_reg)
        circuit.compose(propagator, circuit.qubits, circuit.clbits, inplace=True)
        circuit.save_statevector()  # type: ignore

        simulator = generate_aer_simulator()
        transpiled = transpile(circuit)
        job = simulator.run(transpiled, shots=1)
        result = job.result()
        counts: dict[int, int] = result.get_counts(circuit).int_outcomes()
        output_state_full = result.get_statevector(circuit)

        # Check that all measured qubits are 0
        assert counts[0] == 1

        # Check the output state
        traced = partial_trace(
            output_state_full,
            np.arange(num_qubits, 2 * num_qubits).tolist(),
        )
        output_state = traced.to_statevector()

        psi_fourier = np.fft.fft(psi.data, norm="ortho")
        output_fourier = np.fft.fft(output_state.data, norm="ortho")

        expected_output_fourier = (
            np.exp(1j * -t / hbar * positive_kinetic.data) * psi_fourier
        )

        state1 = Statevector(output_fourier)
        state2 = Statevector(expected_output_fourier)

        # TODO: remove these when you systematically took care of fourier transformations and ensured `norm="ortho"` is always used everywhere in the codebase. these assertions probably move to the corresponding unit test files.
        assert state1.is_valid()
        assert state2.is_valid()

        assert (
            state_fidelity(
                Statevector(output_fourier), Statevector(expected_output_fourier)
            )
            >= 1.0 - REDUCED_FIDELITY_TOLERANCE
        )
