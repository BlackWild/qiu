"""Unit tests for uniformly_controlled_rotation.py."""

import numpy as np
import numpy.typing as npt
import pytest
import scipy.linalg
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from qiskit_pytest_helper.assertions import assert_equal_operators
from qiskit_pytest_helper.circuits import gate_counts
from qiu_quantum_computing.uniformly_controlled_rotation import (
    RotationAxis,
    uniformly_controlled_rotation,
)

axes = st.sampled_from(["y", "z"])


@st.composite
def angle_arrays(draw, max_controls: int = 4) -> npt.NDArray[np.float64]:
    """A strategy for arrays of 2**k rotation angles, including repeated angles."""
    num_controls = draw(st.integers(min_value=0, max_value=max_controls))
    elements = st.one_of(
        st.floats(min_value=-10.0, max_value=10.0),
        st.sampled_from([0.0, np.pi, -np.pi, np.pi / 2]),
    )
    return draw(arrays(np.float64, 2**num_controls, elements=elements))


def rotation(axis: RotationAxis, angle: float) -> npt.NDArray[np.complex128]:
    """The matrix of the rotation about the axis by the angle."""
    c, s = np.cos(angle / 2), np.sin(angle / 2)
    if axis == "y":
        return np.array([[c, -s], [s, c]], dtype=np.complex128)
    return np.diag([np.exp(-1j * angle / 2), np.exp(1j * angle / 2)])


def multiplexer(axis: RotationAxis, angles: npt.NDArray) -> npt.NDArray:
    """The block diagonal unitary rotating the target by angles[c] for controls c."""
    return scipy.linalg.block_diag(*(rotation(axis, angle) for angle in angles))


class TestUniformlyControlledRotation:
    """Test uniformly_controlled_rotation."""

    @given(axis=axes, angles=angle_arrays())
    def test_implements_the_multiplexer(self, axis: RotationAxis, angles):
        """Test that the circuit is exactly the block diagonal multiplexer."""
        circuit = uniformly_controlled_rotation(axis, angles)

        assert circuit.num_qubits == int(np.log2(angles.size)) + 1
        assert_equal_operators(circuit, multiplexer(axis, angles))

    @given(axis=axes, angles=angle_arrays())
    def test_elementary_gates(self, axis: RotationAxis, angles):
        """Test that only rotations about the axis and CNOT gates are used."""
        circuit = uniformly_controlled_rotation(axis, angles)
        assert set(gate_counts(circuit)) <= {f"r{axis}", "cx"}

    @pytest.mark.parametrize("num_controls", range(1, 6))
    def test_cnot_count(self, num_controls: int):
        """Test the 2**k CNOT gates of a generic multiplexer with k controls."""
        angles = np.random.default_rng(num_controls).uniform(-3, 3, 2**num_controls)
        circuit = uniformly_controlled_rotation("y", angles)
        assert gate_counts(circuit)["cx"] == 2**num_controls

    @pytest.mark.parametrize("axis", ["y", "z"])
    def test_uncontrolled_rotation(self, axis: RotationAxis):
        """Test that angles independent of the controls need no CNOT gates."""
        circuit = uniformly_controlled_rotation(axis, np.full(8, 0.7))
        assert gate_counts(circuit) == {f"r{axis}": 1}

    def test_zero_angles(self):
        """Test that zero angles give the empty circuit."""
        circuit = uniformly_controlled_rotation("z", np.zeros(4))
        assert circuit.size() == 0
        assert circuit.num_qubits == 3

    def test_invalid_axis(self):
        """Test that only the y and z axes are supported."""
        with pytest.raises(ValueError, match="axis"):
            uniformly_controlled_rotation("x", [0.1, 0.2])  # type: ignore[arg-type]

    @pytest.mark.parametrize("angles", [[], [0.1, 0.2, 0.3], [[0.1, 0.2]]])
    def test_invalid_number_of_angles(self, angles):
        """Test that the number of angles must be a power of 2."""
        with pytest.raises(ValueError, match="power of 2"):
            uniformly_controlled_rotation("y", angles)
