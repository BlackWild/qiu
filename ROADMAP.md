# Roadmap

The packages planned for the monorepo, and how they fit with the existing ones. A planned package gets its directory in `packages/` once work on it starts, following [CONTRIBUTING.md](CONTRIBUTING.md#adding-a-package).

## Planned packages

### `qiu-quantum-simulation`

The complete simulation of physical dynamical systems on a quantum computer: a physical system and an experiment on it, from the preparation of its initial state through its time evolution to the quantities it is observed by.

It is to build on the existing quantum packages, each of which stays below it in scope:

| Package                      | Scope                                                                                                                  |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `qiu-quantum-computing`      | generic computations with the amplitudes of a quantum computer, without reference to physical dynamics                 |
| `qiu-hamiltonian-simulation` | the unitary time evolution under a given Hamiltonian, without simulating an actual physical experiment                 |
| `qiu-mps-initializer`        | state preparation with matrix product states                                                                           |
| `qiu-quantum-simulation`     | the simulation of a physical system as a whole, using the packages above for its states and its time evolution         |

Like the other quantum packages, it is named after what it does, and names the frameworks it builds on, e.g. Qiskit, only in its description and documentation.
