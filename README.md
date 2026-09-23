# Qiu

Qiu is a monorepo holding various packages and projects related to quantum computing and quantum programming. The main focus is to provide better tools on top of the Qiskit framework.

## Structure

The repository is structured as follows:

- `shareable-packages/`: Contains reusable libraries and tools for things not necessarily related to scientific computation or quantum computing. Think of them as packages that enhance standard python.
  - `python-encore`: A package for enhancing standard Python functionalities.
  - `python-signals`: A package for uniformly sampled axes and signals on them, for classical and quantum numerics alike.

- `dev-packages/`: Contains packages which are not meant to be published for others but just to be used inside this monorepo for development purposes.
  - `qiskit-pytest-helper`: A package that implements custom strategies for `hypothesis` and other stuff to be used in the unit tests of the other packages.

- `packages/`: Contains reusable libraries and tools for quantum computing.
  - `qiskit-encore`: A package for enhancing base Qiskit functionalities.
  - `qiskit-aer-encore`: A package for extending Qiskit's Aer simulator capabilities.
  - `qiskit-signals`: A package for handling signals for quantum applications, i.e. the signals of `python-signals` encoded in qubit registers.
  - `qiskit-phase-propagator`: A package for simulating phase propagation in quantum circuits.
  - `qiskit-hamiltonian-simulation`: A package for simulating Hamiltonian dynamics.

- `apps/`: Contains standalone applications and projects that utilize the other packages.

There are then other directories which are very specific to some task and are not used very frequently:
- `temp/`
- `scripts/`
- `cluster/`
- and the other dot folders `.*/`
