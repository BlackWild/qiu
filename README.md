# Qiu

[![CI](https://github.com/BlackWild/qiu/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/BlackWild/qiu/actions/workflows/ci.yml) [![Docs](https://github.com/BlackWild/qiu/actions/workflows/docs.yml/badge.svg?branch=master)](https://blackwild.github.io/qiu/) [![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE) [![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)](https://www.python.org/) [![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv) [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff) [![Checked with pyright](https://microsoft.github.io/pyright/img/pyright_badge.svg)](https://microsoft.github.io/pyright/)

Qiu is a monorepo of tools for quantum computing and quantum simulation: building blocks of quantum circuits, the phases of sampled signals as circuits, Hamiltonian simulation and state preparation with matrix product states, the numerics of sampled signals and of paraxial wave optics they share, and applications built on them, such as the simulation of wave optics with quantum circuits. Its packages build on [Qiskit](https://www.ibm.com/quantum/qiskit) and [QuTiP](https://qutip.org/), and on NumPy for everything framework-independent.

The documentation of all packages, with their guides, examples and API references, is at <https://blackwild.github.io/qiu/>.

## Packages

| Package | | Description |
| --- | --- | --- |
| **Packages** (`packages/`): published on PyPI | | |
| [`qiu-python-encore`](packages/qiu-python-encore/) | [![PyPI](https://img.shields.io/pypi/v/qiu-python-encore)](https://pypi.org/project/qiu-python-encore/) | Small enhancements of the Python standard library. |
| [`qiu-qiskit-encore`](packages/qiu-qiskit-encore/) | [![PyPI](https://img.shields.io/pypi/v/qiu-qiskit-encore)](https://pypi.org/project/qiu-qiskit-encore/) | Improvements of the native types of Qiskit: validated statevectors and the synthesis methods of circuit building blocks. |
| [`qiu-qiskit-aer-encore`](packages/qiu-qiskit-aer-encore/) | [![PyPI](https://img.shields.io/pypi/v/qiu-qiskit-aer-encore)](https://pypi.org/project/qiu-qiskit-aer-encore/) | Qiskit Aer simulators configured for the available hardware, CPU or GPU. |
| [`qiu-signals`](packages/qiu-signals/) | [![PyPI](https://img.shields.io/pypi/v/qiu-signals)](https://pypi.org/project/qiu-signals/) | Uniformly sampled axes and signals on them, for classical and quantum numerics alike. |
| [`qiu-quantum-computing`](packages/qiu-quantum-computing/) | [![PyPI](https://img.shields.io/pypi/v/qiu-quantum-computing)](https://pypi.org/project/qiu-quantum-computing/) | Generic computations with the amplitudes of a quantum computer, as Qiskit circuits: state preparation, QFT, uniformly controlled rotations and diagonal phase operators `e^(i f(x))` of sampled signals. |
| [`qiu-hamiltonian-simulation`](packages/qiu-hamiltonian-simulation/) | [![PyPI](https://img.shields.io/pypi/v/qiu-hamiltonian-simulation)](https://pypi.org/project/qiu-hamiltonian-simulation/) | Unitary time evolution under a given Hamiltonian: Qiskit circuits of potentials and kinetic energies given as sampled signals. |
| [`qiu-mps-initializer`](packages/qiu-mps-initializer/) | [![PyPI](https://img.shields.io/pypi/v/qiu-mps-initializer)](https://pypi.org/project/qiu-mps-initializer/) | Quantum state preparation for Qiskit with layers of one- and two-qubit gates from matrix product states. |
| [`qiu-classical-simulation`](packages/qiu-classical-simulation/) | [![PyPI](https://img.shields.io/pypi/v/qiu-classical-simulation)](https://pypi.org/project/qiu-classical-simulation/) | Tools for classical simulations: paraxial wave optics of Gaussian beams through sliced lenses, with a simulation loop for any backend and classical references. |
| **Development packages** (`packages-dev/`): test helpers of this monorepo, not published | | |
| [`python-pytest-helper`](packages-dev/python-pytest-helper/) | | Floating-point comparisons and Hypothesis strategies of numbers, axes and signals. |
| [`qiskit-pytest-helper`](packages-dev/qiskit-pytest-helper/) | | Strategies of quantum states and qubit axes, assertions with Qiskit's equality, and circuit helpers. |
| **Applications** (`apps/`): projects built on the packages, not published | | |
| [`wave_optics_propagation`](apps/wave_optics_propagation/) | | The lens experiments of the paper simulated with Qiskit: the simulation run on the cluster and the analyses of the paper. |
| [`wave_optics_propagation_qutip`](apps/wave_optics_propagation_qutip/) | | The same lens experiments simulated with QuTiP. |

The other directories:

- `docs/`: The shared configuration, the landing page and the build of the documentation.
- `scripts/`: The scripts of the repository: in `scripts/cluster/`, the SLURM jobs running the simulations of `apps/wave_optics_propagation`.

## Installation

The published packages install from PyPI, e.g.

```sh
pip install qiu-quantum-computing
```

For development, clone the repository and install all packages of the [uv](https://docs.astral.sh/uv/) workspace, editable:

```sh
git clone https://github.com/BlackWild/qiu.git
cd qiu
uv sync --all-packages
uv run pytest -n auto
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the conventions of the monorepo and how to contribute, and [ROADMAP.md](ROADMAP.md) for the packages planned next.

## Documentation

Every package has its documentation in `docs/`: a home page, a user guide, worked examples and the API reference, generated from the docstrings of its `src` by `docs/gen_reference.py`. Its `mkdocs.yml` inherits the shared configuration `docs/mkdocs.base.yml`, and `docs/build_all.py` builds all packages, with the landing page `mkdocs.yml`, into one site:

```sh
uv run python docs/build_all.py  # into site/
```

The Python examples of the documentation and the READMEs run as part of the tests (`docs/tests/test_examples.py`).

## Continuous integration

The GitHub Actions workflows in `.github/workflows/`:

- `ci.yml`: lints (`ruff`) and type checks (`pyright`) the monorepo, tests all packages on Python 3.11 to 3.14, builds and checks all packages (`twine check`), and builds the documentation, on pushes to `master` and on pull requests.
- `docs.yml`: builds the documentation and deploys it to GitHub Pages on pushes to `master`.
- `publish-<package>.yml`: publishes a package to PyPI when it changes on `master`, or when run by hand, for the libraries in `packages/`. They share the composite action `.github/actions/publish-package`: it tests the package with only the dependencies it declares, builds it, checks that its release is installable from PyPI, i.e. that its dependencies of this monorepo are published, and publishes it as a trusted publisher unless its version is already on PyPI.

The tests on GitHub Actions use the built-in Hypothesis profile `ci`, derandomized and without deadlines per example. Dependabot (`.github/dependabot.yml`) proposes updates of the actions and the locked dependencies quarterly.

## Citation

If you use this software in your research, please cite it with the metadata of [CITATION.cff](CITATION.cff).

## License

The monorepo and all its packages are licensed under the [MIT License](LICENSE).
