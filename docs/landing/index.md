# Qiu

Qiu is a monorepo of tools for quantum computing and quantum simulation, whose packages build on [Qiskit](https://www.ibm.com/quantum/qiskit) and [QuTiP](https://qutip.org/), and on NumPy for everything framework-independent. Each package has its own section of this site, listed in the navigation, with a user guide, worked examples and its API reference, generated from its docstrings. The packages are ordered from the foundations to the applications:

| Package | PyPI | Description |
| --- | --- | --- |
| **Packages** | | |
| [`qiu-signals`](qiu-signals/index.md) | [![PyPI](https://img.shields.io/pypi/v/qiu-signals?style=for-the-badge&label=PyPI)](https://pypi.org/project/qiu-signals/) | Sampled axes and signals, for classical and quantum numerics alike |
| [`qiu-quantum-computing`](qiu-quantum-computing/index.md) | [![PyPI](https://img.shields.io/pypi/v/qiu-quantum-computing?style=for-the-badge&label=PyPI)](https://pypi.org/project/qiu-quantum-computing/) | Computations on the amplitudes of a quantum computer: state preparation, QFT, controlled rotations and diagonal phase operators |
| [`qiu-mps-initializer`](qiu-mps-initializer/index.md) | [![PyPI](https://img.shields.io/pypi/v/qiu-mps-initializer?style=for-the-badge&label=PyPI)](https://pypi.org/project/qiu-mps-initializer/) | State preparation with layers of gates from matrix product states |
| [`qiu-hamiltonian-simulation`](qiu-hamiltonian-simulation/index.md) | [![PyPI](https://img.shields.io/pypi/v/qiu-hamiltonian-simulation?style=for-the-badge&label=PyPI)](https://pypi.org/project/qiu-hamiltonian-simulation/) | The unitary time evolution under a given Hamiltonian |
| [`qiu-classical-simulation`](qiu-classical-simulation/index.md) | [![PyPI](https://img.shields.io/pypi/v/qiu-classical-simulation?style=for-the-badge&label=PyPI)](https://pypi.org/project/qiu-classical-simulation/) | Classical simulations: paraxial wave optics of Gaussian beams through lenses |
| **Applications, not published** | | |
| [`wave_optics_propagation`](wave_optics_propagation/index.md) |  | The lens experiments of the paper, simulated with Qiskit, and their analyses |
| [`wave_optics_propagation_qutip`](wave_optics_propagation_qutip/index.md) |  | The same lens experiments, simulated with QuTiP |
| **Encore packages: improvements of Python, Qiskit and Qiskit Aer** | | |
| [`qiu-python-encore`](qiu-python-encore/index.md) | [![PyPI](https://img.shields.io/pypi/v/qiu-python-encore?style=for-the-badge&label=PyPI)](https://pypi.org/project/qiu-python-encore/) | Small enhancements of the Python standard library |
| [`qiu-qiskit-encore`](qiu-qiskit-encore/index.md) | [![PyPI](https://img.shields.io/pypi/v/qiu-qiskit-encore?style=for-the-badge&label=PyPI)](https://pypi.org/project/qiu-qiskit-encore/) | Validated statevectors and the synthesis methods of circuit building blocks |
| [`qiu-qiskit-aer-encore`](qiu-qiskit-aer-encore/index.md) | [![PyPI](https://img.shields.io/pypi/v/qiu-qiskit-aer-encore?style=for-the-badge&label=PyPI)](https://pypi.org/project/qiu-qiskit-aer-encore/) | Aer simulators configured for the available hardware, CPU or GPU |
| **Development packages: test helpers, not published** | | |
| [`python-pytest-helper`](python-pytest-helper/index.md) |  | Floating-point assertions and Hypothesis strategies of numbers, axes and signals |
| [`qiskit-pytest-helper`](qiskit-pytest-helper/index.md) |  | Strategies of quantum states, Qiskit assertions and circuit helpers |

The sources are on [GitHub](https://github.com/BlackWild/qiu), where [CONTRIBUTING.md](https://github.com/BlackWild/qiu/blob/master/CONTRIBUTING.md) describes how to contribute.
