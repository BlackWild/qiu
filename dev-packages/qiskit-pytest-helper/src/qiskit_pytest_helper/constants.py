"""Bounds of the quantum strategies and tolerances of approximate protocols in tests."""

FIDELITY_TOLERANCE = 0.0001
"""The infidelity allowed of a sample-based phase to the exact one, for small phases."""

REDUCED_FIDELITY_TOLERANCE = 0.01
"""The infidelity allowed of a sample-based evolution to the exact one."""

MIN_QUBITS = 2
"""The default minimum number of qubits of the quantum strategies."""

MAX_QUBITS = 4
"""The default maximum number of qubits of the quantum strategies."""

MIN_MAGNITUDE = 0.01
"""The default minimum magnitude of generated amplitudes and coefficients."""

MAX_MAGNITUDE = 1.0
"""The default maximum magnitude of generated amplitudes and coefficients."""

MAX_NUM_OF_CYCLES = 200  # TODO: derive the bound from the parameters of the tests
"""The maximum number of cycles of a sample-based propagator in the tests."""
