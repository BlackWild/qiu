"""Circuits applying the phase `e^(i f(x))` of a sampled signal to a qubit register.

`qubit_encoding` fixes how an axis is represented by qubits, `direct` applies polynomial
phases exactly, `sample_based` arbitrary phases of one sign with the sample-based phase
protocol, and `sample_based_manual` simulates the protocol on statevectors.
"""
