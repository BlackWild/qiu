# Qiu

Qiu is a monorepo of tools for quantum computing and quantum simulation, whose packages build on [Qiskit](https://www.ibm.com/quantum/qiskit) and [QuTiP](https://qutip.org/), and on NumPy for everything framework-independent. Each package has its own documentation, with a user guide, worked examples and its API reference:

| Package | | Description |
| --- | --- | --- |
| **Shareable packages** (`shareable-packages/`): useful beyond quantum computing | | |
| [`python-encore`](python-encore/) | [![PyPI](https://img.shields.io/pypi/v/python-encore)](https://pypi.org/project/python-encore/) | Small enhancements of the Python standard library. |
| [`python-signals`](python-signals/) | [![PyPI](https://img.shields.io/pypi/v/python-signals)](https://pypi.org/project/python-signals/) | Uniformly sampled axes and signals on them, for classical and quantum numerics alike. |
| [`python-wave-optics`](python-wave-optics/) | [![PyPI](https://img.shields.io/pypi/v/python-wave-optics)](https://pypi.org/project/python-wave-optics/) | Paraxial wave optics of Gaussian beams through sliced lenses, with a simulation loop for any backend and classical references. |
| **Packages** (`packages/`): quantum computing, with circuits of Qiskit | | |
| [`qiskit-encore`](qiskit-encore/) | [![PyPI](https://img.shields.io/pypi/v/qiskit-encore)](https://pypi.org/project/qiskit-encore/) | Robust state preparation, QFT and uniformly controlled rotations, each as a dense unitary, a Qiskit gate or a decomposed circuit. |
| [`qiskit-aer-encore`](qiskit-aer-encore/) | [![PyPI](https://img.shields.io/pypi/v/qiskit-aer-encore)](https://pypi.org/project/qiskit-aer-encore/) | Aer simulators configured for the available hardware, CPU or GPU. |
| [`qiskit-phase-propagator`](qiskit-phase-propagator/) | [![PyPI](https://img.shields.io/pypi/v/qiskit-phase-propagator)](https://pypi.org/project/qiskit-phase-propagator/) | Circuits applying phases `e^(i f(x))` of sampled signals to qubit registers, directly or sample-based. |
| [`qiskit-hamiltonian-simulation`](qiskit-hamiltonian-simulation/) | [![PyPI](https://img.shields.io/pypi/v/qiskit-hamiltonian-simulation)](https://pypi.org/project/qiskit-hamiltonian-simulation/) | Time evolution under potentials and kinetic energies given as sampled signals. |
| [`qiskit-mps-initializer`](qiskit-mps-initializer/) | [![PyPI](https://img.shields.io/pypi/v/qiskit-mps-initializer)](https://pypi.org/project/qiskit-mps-initializer/) | Approximate state preparation with layers of one- and two-qubit gates from matrix product states. |
| **Development packages** (`dev-packages/`): test helpers of this monorepo, not published | | |
| [`python-pytest-helper`](python-pytest-helper/) | | Floating-point comparisons and Hypothesis strategies of numbers, axes and signals. |
| [`qiskit-pytest-helper`](qiskit-pytest-helper/) | | Strategies of quantum states and qubit axes, assertions with Qiskit's equality, and circuit helpers. |
| **Applications** (`apps/`): projects built on the packages, not published | | |
| [`wave_optics_propagation`](wave_optics_propagation/) | | The lens experiments of the paper simulated with Qiskit: the simulation run on the cluster and the analyses of the paper. |
| [`wave_optics_propagation_qutip`](wave_optics_propagation_qutip/) | | The same lens experiments simulated with QuTiP. |

The sources are on [GitHub](https://github.com/BlackWild/qiu), where [CONTRIBUTING.md](https://github.com/BlackWild/qiu/blob/master/CONTRIBUTING.md) describes how to contribute.
