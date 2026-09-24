"""Unit tests for synthesis_method.py."""

from qiskit_encore.synthesis_method import SynthesisMethod


def test_values():
    """Test the raw values of the synthesis methods."""
    assert SynthesisMethod.list() == ["dense", "gate", "decomposed"]


def test_construction_from_raw_values():
    """Test that methods can be given as raw values."""
    assert SynthesisMethod("gate") is SynthesisMethod.GATE
    assert SynthesisMethod.DENSE == "dense"
