# Changelog

All notable changes to `qiskit-mps-initializer` are documented in this file. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the package adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html); before 1.0.0, minor versions may break the API.

## [Unreleased]

## [0.3.0]

The package moved into the [qiu](https://github.com/BlackWild/qiu) monorepo, onto its packages.

### Changed

- `mps_state_preparation(state, max_layers, tolerance=None)` replaces `QuantumState.generate_mps_initializer_circuit` and `multi_layered_circuit_for_non_approximated`, and returns the circuit, its layers, its error and whether it converged.
- `max_layers` is required and bounds the number of layers; by default, the preparation stops once the prepared state equals the target up to Qiskit's tolerances.
- `QuantumState` is replaced by Qiskit's `Statevector`, validated by `qiskit-encore`; `QuantumIntensity` by `qiskit_phase_propagator.sample_based.sample_based_decomposition`; the sampling helpers by `python-signals`.
- `G_matrices` is renamed `disentangler_matrices`, in the new module `mps` with `bond2_mps_approximation` and `mps_layer`.
- Requires Python 3.11 or newer.

### Added

- Single-qubit states.

### Removed

- The dependencies `pydantic`, `pydantic-numpy` and `qiskit-aer`, and the `utils` simulation helpers.

### Fixed

- At most `max_layers` layers are built; formerly, one more layer than asked for could be added, and without a maximum the construction could run forever.

## [0.2.4] - 2025-08-13

### Removed

- Debug prints.

## [0.2.3] - 2025-08-06

### Added

- The tolerance `atol` of the multi-layered MPS initializer.

### Fixed

- The construction of the MPS layers, with tests of more edge cases.

## [0.2.2] - 2025-06-17

### Changed

- The usage example of the README.

## [0.2.1] - 2025-06-16

### Changed

- Links to the documentation.

## [0.2.0] - 2025-06-16

### Added

- The documentation, with the API reference.

### Changed

- The datatypes `QuantumState` and `QuantumIntensity`, as pydantic models supporting NumPy arrays, and the modules of the package.
- Requires Python 3.10 or newer.

## [0.1.1] - 2025-06-03

### Fixed

- A typo.

## [0.1.0] - 2025-06-03

### Added

- MPS-based initializers of wavefunctions for Qiskit.

[Unreleased]: https://github.com/BlackWild/qiu/commits/master/packages/qiskit-mps-initializer
[0.3.0]: https://github.com/BlackWild/qiu/tree/master/packages/qiskit-mps-initializer
[0.2.4]: https://github.com/BlackWild/qiskit-mps-initializer/releases/tag/v0.2.4
[0.2.3]: https://github.com/BlackWild/qiskit-mps-initializer/releases/tag/v0.2.3
[0.2.2]: https://pypi.org/project/qiskit-mps-initializer/0.2.2/
[0.2.1]: https://pypi.org/project/qiskit-mps-initializer/0.2.1/
[0.2.0]: https://pypi.org/project/qiskit-mps-initializer/0.2.0/
[0.1.1]: https://pypi.org/project/qiskit-mps-initializer/0.1.1/
[0.1.0]: https://pypi.org/project/qiskit-mps-initializer/0.1.0/
