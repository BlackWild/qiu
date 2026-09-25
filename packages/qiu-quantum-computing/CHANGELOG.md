# Changelog

All notable changes to `qiu-quantum-computing` are documented in this file. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the package adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html); before 1.0.0, minor versions may break the API.

## [Unreleased]

## [0.1.0]

### Added

- State preparation (`state_preparation_circuit`, `PreparableState`), the QFT with NumPy's convention (`qft_circuit`) and uniformly controlled rotations, each available as a dense unitary, a Qiskit gate or a decomposed circuit as chosen by the `SynthesisMethod` of `qiu-qiskit-encore`.
- The diagonal phase operators of the subpackage `phase_propagator`: the qubit encodings of the index orderings, direct polynomial phase circuits up to power 3, the sample-based phase propagators and their statevector simulation.

[Unreleased]: https://github.com/BlackWild/qiu/commits/master/packages/qiu-quantum-computing
[0.1.0]: https://github.com/BlackWild/qiu/tree/master/packages/qiu-quantum-computing
