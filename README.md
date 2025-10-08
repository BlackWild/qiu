# Qiu

Qiu is a monorepo holding various packages and projects related to quantum computing and quantum programming. The main focus is to provide better tools on top of the Qiskit framework.

## Structure

The repository is structured as follows:

- `packages/`: Contains reusable libraries and tools for quantum computing.
  - `qiskit-encore`: A package for enhancing base Qiskit functionalities.
  - `qiskit-aer-encore`: A package for extending Qiskit's Aer simulator capabilities.
  - `qiskit-signals`: A package for handling signals for quantum applications.
  - `qiskit-phase-propagator`: A package for simulating phase propagation in quantum circuits.
  - `qiskit-hamiltonian-simulation`: A package for simulating Hamiltonian dynamics.
  - `qiskit-mps-initializer`: A package for initializing quantum states using Matrix Product States (MPS).
- `projects/`: Contains standalone applications and projects that utilize the packages.
