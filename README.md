# Qiu

[![CI](https://github.com/BlackWild/qiu/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/BlackWild/qiu/actions/workflows/ci.yml) [![Docs](https://github.com/BlackWild/qiu/actions/workflows/docs.yml/badge.svg?branch=master)](https://blackwild.github.io/qiu/) [![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE) [![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)](https://www.python.org/) [![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv) [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff) [![Checked with pyright](https://microsoft.github.io/pyright/img/pyright_badge.svg)](https://microsoft.github.io/pyright/)

Qiu is a monorepo of tools for quantum computing and quantum simulation: building blocks of quantum circuits, the phases of sampled signals as circuits, Hamiltonian simulation and state preparation with matrix product states, the numerics of sampled signals and of paraxial wave optics they share, and applications built on them, such as the simulation of wave optics with quantum circuits. Its packages build on [Qiskit](https://www.ibm.com/quantum/qiskit) and [QuTiP](https://qutip.org/), and on NumPy for everything framework-independent.

The documentation of all packages, with their guides, examples and API references, is at <https://blackwild.github.io/qiu/>.

## Packages

| Package | PyPI | Description |
| --- | --- | --- |
| **Packages** | | |
| [`qiu-signals`](packages/qiu-signals/) | [![PyPI](https://img.shields.io/pypi/v/qiu-signals?style=for-the-badge&label=PyPI)](https://pypi.org/project/qiu-signals/) | Sampled axes and signals, for classical and quantum numerics alike |
| [`qiu-quantum-computing`](packages/qiu-quantum-computing/) | [![PyPI](https://img.shields.io/pypi/v/qiu-quantum-computing?style=for-the-badge&label=PyPI)](https://pypi.org/project/qiu-quantum-computing/) | Computations on the amplitudes of a quantum computer: state preparation, QFT, controlled rotations and diagonal phase operators |
| [`qiu-mps-initializer`](packages/qiu-mps-initializer/) | [![PyPI](https://img.shields.io/pypi/v/qiu-mps-initializer?style=for-the-badge&label=PyPI)](https://pypi.org/project/qiu-mps-initializer/) | State preparation with layers of gates from matrix product states |
| [`qiu-hamiltonian-simulation`](packages/qiu-hamiltonian-simulation/) | [![PyPI](https://img.shields.io/pypi/v/qiu-hamiltonian-simulation?style=for-the-badge&label=PyPI)](https://pypi.org/project/qiu-hamiltonian-simulation/) | The unitary time evolution under a given Hamiltonian |
| [`qiu-classical-simulation`](packages/qiu-classical-simulation/) | [![PyPI](https://img.shields.io/pypi/v/qiu-classical-simulation?style=for-the-badge&label=PyPI)](https://pypi.org/project/qiu-classical-simulation/) | Classical simulations: paraxial wave optics of Gaussian beams through lenses |
| **Applications, not published** | | |
| [`wave_optics_propagation`](apps/wave_optics_propagation/) |  | The lens experiments of the paper, simulated with Qiskit, and their analyses |
| [`wave_optics_propagation_qutip`](apps/wave_optics_propagation_qutip/) |  | The same lens experiments, simulated with QuTiP |
| **Encore packages: improvements of Python, Qiskit and Qiskit Aer** | | |
| [`qiu-python-encore`](packages/qiu-python-encore/) | [![PyPI](https://img.shields.io/pypi/v/qiu-python-encore?style=for-the-badge&label=PyPI)](https://pypi.org/project/qiu-python-encore/) | Small enhancements of the Python standard library |
| [`qiu-qiskit-encore`](packages/qiu-qiskit-encore/) | [![PyPI](https://img.shields.io/pypi/v/qiu-qiskit-encore?style=for-the-badge&label=PyPI)](https://pypi.org/project/qiu-qiskit-encore/) | Validated statevectors and the synthesis methods of circuit building blocks |
| [`qiu-qiskit-aer-encore`](packages/qiu-qiskit-aer-encore/) | [![PyPI](https://img.shields.io/pypi/v/qiu-qiskit-aer-encore?style=for-the-badge&label=PyPI)](https://pypi.org/project/qiu-qiskit-aer-encore/) | Aer simulators configured for the available hardware, CPU or GPU |
| **Development packages: test helpers, not published** | | |
| [`python-pytest-helper`](packages-dev/python-pytest-helper/) |  | Floating-point assertions and Hypothesis strategies of numbers, axes and signals |
| [`qiskit-pytest-helper`](packages-dev/qiskit-pytest-helper/) |  | Strategies of quantum states, Qiskit assertions and circuit helpers |

The other directories:

- `docs/`: The configuration shared by the documentation of all packages, the landing page, and the generation and build of the documentation site.
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

The documentation is one [MkDocs](https://www.mkdocs.org/) site, configured by the root `mkdocs.yml`: the landing page, and a section per package with its home page, user guide and worked examples from its `docs/`, and its API reference, generated from the docstrings of its `src` by [mkdocstrings](https://mkdocstrings.github.io/). `docs/gen_pages.py` collects the pages and writes the navigation, which lists every package on every page. Build or serve it with:

```sh
uv run python docs/build_all.py  # into site/, strictly, and checks its links
uv run mkdocs serve              # at http://127.0.0.1:8000/qiu/
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
