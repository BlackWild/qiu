"""Unit tests for mps.py."""

import numpy as np
import pytest
import quimb.tensor as qtn
from hypothesis import given, settings
from hypothesis import strategies as st
from qiskit.quantum_info import Statevector
from qiskit_pytest_helper.assertions import assert_equal_states, assert_unitary
from qiu_mps_initializer.mps import (
    bond2_mps_approximation,
    disentangler_matrices,
    mps_layer,
)

sites = st.integers(min_value=2, max_value=7)


def random_bond2_mps(num_sites: int, seed: int) -> qtn.MatrixProductState:
    """A random normalized MPS of bond dimension 2."""
    return qtn.MPS_rand_state(
        L=num_sites, bond_dim=2, dtype="complex128", normalize=True, seed=seed
    )


class TestBond2MPSApproximation:
    """Test bond2_mps_approximation."""

    @settings(deadline=None)
    @given(num_sites=sites, seed=st.integers(0, 2**16))
    def test_exact_for_bond2_states(self, num_sites: int, seed: int):
        """Test that states of bond dimension 2 are represented exactly."""
        state = random_bond2_mps(num_sites, seed).to_dense().reshape(-1)
        mps = bond2_mps_approximation(state)
        max_bond = mps.max_bond()
        assert max_bond is not None and max_bond <= 2
        assert_equal_states(mps.to_dense().reshape(-1), state)

    def test_rejects_single_qubits(self):
        """Test that a single qubit has no MPS of 2 sites or more."""
        with pytest.raises(ValueError, match="2 qubits"):
            bond2_mps_approximation([0.6, 0.8])

    def test_rejects_unnormalized_states(self):
        """Test that states must be normalized."""
        with pytest.raises(ValueError, match="normalized"):
            bond2_mps_approximation([1.0, 1.0, 1.0, 1.0])


class TestDisentanglerMatrices:
    """Test disentangler_matrices and mps_layer."""

    @settings(deadline=None)
    @given(num_sites=sites, seed=st.integers(0, 2**16))
    def test_unitaries(self, num_sites: int, seed: int):
        """Test one unitary per site, of two qubits but for the last site."""
        mps = bond2_mps_approximation(random_bond2_mps(num_sites, seed).to_dense())
        matrices = disentangler_matrices(mps)

        assert len(matrices) == num_sites
        assert [matrix.shape for matrix in matrices] == [(4, 4)] * (num_sites - 1) + [
            (2, 2)
        ]
        for matrix in matrices:
            assert_unitary(matrix)

    @settings(deadline=None)
    @given(num_sites=sites, seed=st.integers(0, 2**16))
    def test_layer_prepares_the_mps(self, num_sites: int, seed: int):
        """Test that the layer prepares the state of the MPS from |0...0>."""
        state = random_bond2_mps(num_sites, seed).to_dense().reshape(-1)
        assert_equal_states(mps_layer(bond2_mps_approximation(state)), state)

    @pytest.mark.parametrize(
        "state",
        [
            Statevector.from_label("000"),
            Statevector.from_label("101"),
            Statevector.from_label("+-0"),
            Statevector([1, 0, 0, 0, 0, 0, 0, 1] / np.sqrt(2)),
            Statevector([0, 1, 1, 0, 1, 0, 0, 0] / np.sqrt(3)),
        ],
        ids=["zero", "basis", "product", "GHZ", "W"],
    )
    def test_layer_of_structured_states(self, state: Statevector):
        """Test the bond dimensions 1 and zero tensors, whose columns are completed."""
        assert_equal_states(mps_layer(bond2_mps_approximation(state)), state)

    def test_rejects_larger_bond_dimensions(self):
        """Test that an MPS of bond dimension 4 has no layer."""
        with pytest.raises(ValueError, match="at most 2"):
            disentangler_matrices(random_bond2_mps(4, 0).expand_bond_dimension(4))
