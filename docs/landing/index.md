# Qiu

Qiu is a monorepo of tools for quantum computing and quantum simulation, whose packages build on [Qiskit](https://www.ibm.com/quantum/qiskit) and [QuTiP](https://qutip.org/), and on NumPy for everything framework-independent. Each package has its own documentation, with a user guide, worked examples and its API reference:

| Package | | Description |
| --- | --- | --- |
| **Packages** (`packages/`): published on PyPI | | |
| [`qiu-python-encore`](qiu-python-encore/) | [![PyPI](https://img.shields.io/pypi/v/qiu-python-encore)](https://pypi.org/project/qiu-python-encore/) | Small enhancements of the Python standard library. |
| [`qiu-qiskit-encore`](qiu-qiskit-encore/) | [![PyPI](https://img.shields.io/pypi/v/qiu-qiskit-encore)](https://pypi.org/project/qiu-qiskit-encore/) | Improvements of the native types of Qiskit: validated statevectors and the synthesis methods of circuit building blocks. |
| [`qiu-qiskit-aer-encore`](qiu-qiskit-aer-encore/) | [![PyPI](https://img.shields.io/pypi/v/qiu-qiskit-aer-encore)](https://pypi.org/project/qiu-qiskit-aer-encore/) | Qiskit Aer simulators configured for the available hardware, CPU or GPU. |
| [`qiu-signals`](qiu-signals/) | [![PyPI](https://img.shields.io/pypi/v/qiu-signals)](https://pypi.org/project/qiu-signals/) | Uniformly sampled axes and signals on them, for classical and quantum numerics alike. |
| [`qiu-quantum-computing`](qiu-quantum-computing/) | [![PyPI](https://img.shields.io/pypi/v/qiu-quantum-computing)](https://pypi.org/project/qiu-quantum-computing/) | Generic computations with the amplitudes of a quantum computer, as Qiskit circuits: state preparation, QFT, uniformly controlled rotations and diagonal phase operators `e^(i f(x))` of sampled signals. |
| [`qiu-hamiltonian-simulation`](qiu-hamiltonian-simulation/) | [![PyPI](https://img.shields.io/pypi/v/qiu-hamiltonian-simulation)](https://pypi.org/project/qiu-hamiltonian-simulation/) | Unitary time evolution under a given Hamiltonian: Qiskit circuits of potentials and kinetic energies given as sampled signals. |
| [`qiu-mps-initializer`](qiu-mps-initializer/) | [![PyPI](https://img.shields.io/pypi/v/qiu-mps-initializer)](https://pypi.org/project/qiu-mps-initializer/) | Quantum state preparation for Qiskit with layers of one- and two-qubit gates from matrix product states. |
| [`qiu-classical-simulation`](qiu-classical-simulation/) | [![PyPI](https://img.shields.io/pypi/v/qiu-classical-simulation)](https://pypi.org/project/qiu-classical-simulation/) | Tools for classical simulations: paraxial wave optics of Gaussian beams through sliced lenses, with a simulation loop for any backend and classical references. |
| **Development packages** (`packages-dev/`): test helpers of this monorepo, not published | | |
| [`python-pytest-helper`](python-pytest-helper/) | | Floating-point comparisons and Hypothesis strategies of numbers, axes and signals. |
| [`qiskit-pytest-helper`](qiskit-pytest-helper/) | | Strategies of quantum states and qubit axes, assertions with Qiskit's equality, and circuit helpers. |
| **Applications** (`apps/`): projects built on the packages, not published | | |
| [`wave_optics_propagation`](wave_optics_propagation/) | | The lens experiments of the paper simulated with Qiskit: the simulation run on the cluster and the analyses of the paper. |
| [`wave_optics_propagation_qutip`](wave_optics_propagation_qutip/) | | The same lens experiments simulated with QuTiP. |

The sources are on [GitHub](https://github.com/BlackWild/qiu), where [CONTRIBUTING.md](https://github.com/BlackWild/qiu/blob/master/CONTRIBUTING.md) describes how to contribute.
