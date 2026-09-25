# Contributing to Qiu

Thank you for considering a contribution. This guide covers how to set up the monorepo, the conventions its packages follow, and how changes get tested, documented and released. Participation follows the [Code of Conduct](CODE_OF_CONDUCT.md).

## Setting up

The monorepo is a [uv](https://docs.astral.sh/uv/) workspace. With uv installed, from the repository root:

```sh
uv sync --all-packages   # all packages, editable, with all dependency groups
uvx pre-commit install   # lint, format and type check each commit
```

The Python version is the one of `.python-version`; the packages support Python 3.11 to 3.14. Alternatively, open the repository in its dev container (`.devcontainer/`), which does the above. In VS Code, install the recommended extensions of `.vscode/extensions.json`; the workspace settings `.vscode/settings.json` select the environment `.venv`, the tests and Ruff as the formatter.

The dependency groups of the root `pyproject.toml`, all installed by default:

| Group  | Contains                                                          |
| ------ | ----------------------------------------------------------------- |
| `dev`  | The tools of development: Ruff and pyright                        |
| `test` | pytest with its plugins, Hypothesis and the test helpers          |
| `docs` | MkDocs with its plugins                                           |
| `demo` | Running the `# %%` scripts cell by cell (`ipykernel`), matplotlib |

Install only some with `uv sync --all-packages --no-default-groups --group <group>`, as CI does.

The tools are configured in their own files at the repository root, so that `pyproject.toml` describes only the workspace: `ruff.toml`, `pyrightconfig.json`, `pytest.toml`, `.coveragerc.toml` and `taplo.toml` (TOML formatting).

## Layout

| Directory       | Contains                                                             | May depend on    |
| --------------- | -------------------------------------------------------------------- | ---------------- |
| `packages/`     | The published libraries, e.g. `qiu-signals` and `qiu-quantum-computing` | each other       |
| `packages-dev/` | Test helpers, never published                                        | packages         |
| `apps/`         | Applications, e.g. the simulations of a paper, never published       | everything above |

Dependencies point down this table only, and no library imports an app. Among the packages, they point from the foundations to their uses:

- `qiu-python-encore`, `qiu-qiskit-encore` and `qiu-qiskit-aer-encore` add to Python, Qiskit and Qiskit Aer only what improves them and their native types; `qiu-signals` treats signals in general.
- `qiu-quantum-computing` computes with the amplitudes of a quantum computer, without reference to the physical dynamics they may simulate, e.g. the diagonal phase operators of its `phase_propagator`.
- `qiu-hamiltonian-simulation` builds on it to implement the unitary time evolution under a given Hamiltonian, without simulating an actual physical experiment, and `qiu-mps-initializer` prepares states with matrix product states.
- `qiu-quantum-simulation`, planned in the [roadmap](ROADMAP.md), is to build on them to simulate complete physical dynamical systems.
- `qiu-classical-simulation` holds the tools of classical simulations, e.g. its `wave_optics`.

`qiu-python-encore`, `qiu-signals` and `qiu-classical-simulation` never import a quantum computing framework such as Qiskit or QuTiP. Before writing new functionality, check whether a package already provides it; reuse it, or extend the package it belongs in.

The published packages share the namespace `qiu`, and are imported as their name with underscores, e.g. `qiu_quantum_computing`. A package improving a library is named `qiu-<library>-encore`, e.g. `qiu-qiskit-encore`; the others are named after their field, e.g. `qiu-quantum-computing` or `qiu-classical-simulation`, and hold a subpackage per topic, e.g. `qiu_quantum_computing.phase_propagator` or `qiu_classical_simulation.wave_optics`. The packages of `packages-dev/` and `apps/`, never published, are named freely.

Each package has the same layout:

```text
<group>/<package>/
    pyproject.toml     metadata and dependencies (see "Adding a package")
    README.md          overview, installation and usage, with badges
    CHANGELOG.md       for published packages, in the Keep a Changelog format
    LICENSE            a copy of the root LICENSE
    mkdocs.yml         INHERIT: ../../docs/mkdocs.base.yml
    docs/              index.md, user-guide.md and examples.md
    src/<import_name>/ the code, with py.typed and an __init__.py in every directory
    tests/<import_name>/
```

## Code

- Format and lint with `ruff` (`uv run ruff format .` and `uv run ruff check --fix .`), configured in `ruff.toml`, and type check with `pyright` (`uv run pyright`), configured in `pyrightconfig.json`. Pylance, the language server of Python in VS Code, is pyright and reads the same configuration, so the editor shows the errors CI reports.
- Every module, class and public function has a Google-style docstring: a summary line, then `Args:`, `Returns:` and `Raises:` where applicable; a one-line docstring, e.g. `"""Return the norm of the state."""`, needs no sections. Ruff checks them (the rules `D` of pydocstyle and `DOC` of pydoclint), and the API reference of the documentation is generated from them.
- Annotate the types of all arguments and results: accept `npt.ArrayLike` (or `Statevector | npt.ArrayLike`) as input where any array-like works, and return the most precise type, e.g. `npt.NDArray[np.complex128]`.
- Name things by what they are, in full words; physical conventions (units, index orderings, signs, global phases) are stated in the docstrings.
- Choices between implementations are enums checked exhaustively: every member has its own branch, and unknown ones raise `NotImplementedError` (see `qiu_qiskit_encore.synthesis_method.SynthesisMethod`).
- No hidden constants: tolerances and similar parameters are arguments, or follow the standard of the library at hand, e.g. Qiskit's `Statevector` equality.
- Scripts of the apps are section-based (`# %%`) Python files, not notebooks.

## Tests

- Tests live in `tests/<import_name>/test_<module>.py`, one test class per unit, with a docstring saying what is tested. Run them with `uv run pytest -n auto`, or for one package, e.g. `uv run pytest packages/qiu-quantum-computing`.
- Prefer property-based tests with [Hypothesis](https://hypothesis.readthedocs.io/), using the shared strategies of `python-pytest-helper` (numbers, axes, signals) and `qiskit-pytest-helper` (quantum states, qubit axes).
- Compare floating-point results with the shared assertions, never with a tolerance chosen per test: `python_pytest_helper.assertions.assert_close` for numbers and arrays (relative tolerance, with an absolute floor at the underflow), `assert_close_in_norm` for vectors computed as a whole, and `qiskit_pytest_helper.assertions.assert_equal_states` and `assert_equal_operators` for states and operators (Qiskit's equality, global phase included).
- Approximations are tested against bounds derived from the approximation, and exact results exactly.
- A package's tests may only use the dependencies it declares, including its `test` dependency group: the publish workflows test each package in an environment of only those.
- On CI, Hypothesis loads its built-in profile `ci`: derandomized, without deadlines per example or example database, and printing how to reproduce a failure. Run the tests like CI with `CI=true uv run pytest -n auto`.

## Documentation

- Each package's documentation has a home page (`docs/index.md`: overview, installation, quick start), a user guide (`docs/user-guide.md`: concepts, conventions, pitfalls), examples (`docs/examples.md`: worked use cases) and the generated API reference.
- Every `python` code block of the documentation and the READMEs is executed by `docs/tests/test_examples.py`, as part of the tests: make each block self-contained, and assert the results it claims. Mark a block that cannot run, e.g. one needing a cluster, with a preceding `<!-- no-test -->` line.
- Build the documentation of all packages with `uv run python docs/build_all.py`, which builds strictly, so broken links and references fail; serve one package's with `uv run mkdocs serve -f <group>/<package>/mkdocs.yml`.
- Links between packages in their documentation use the paths of their sites: `../<package>/` from a home page `index.md`, and `../../<package>/` from the other pages, which MkDocs serves one directory deeper (`<package>/user-guide/`). The build checks that every link of the site resolves. Links in READMEs are absolute GitHub URLs, so that they also work on PyPI.

## Adding a package

1. Create `<group>/<package>/` with the layout above, a published package named as described there; the workspace picks up every directory of the groups.
2. Write its `pyproject.toml` like the existing ones: the metadata (description, `license = "MIT"`, `license-files`, authors, keywords, classifiers including the supported Python versions, and the `project.urls` of its source, documentation, repository, issues and changelog), `uv_build` as the build backend, its dependencies, a `test` dependency group with what its tests need, and `tool.uv.sources` of the workspace packages it uses. Packages that are not published get the classifier `Private :: Do Not Upload`.
3. Add it to the root `README.md` and to the landing page `docs/landing/index.md`.
4. For a published package, add `.github/workflows/publish-<package>.yml` like the existing ones, and a trusted publisher on PyPI (see "Releasing").
5. Run `uv lock`.

## Pull requests

1. Branch off `master`, and keep a pull request to one topic.
2. Make sure that the checks of the pull request template pass: the tests, the linters and type checks, the documentation build, and a changelog entry under `Unreleased` for each changed published package.
3. CI (`.github/workflows/ci.yml`) runs all of them, on every supported Python version.

## Releasing

Published packages follow [Semantic Versioning](https://semver.org/); before 1.0.0, minor versions may break the API.

1. Bump the `version` in the package's `pyproject.toml`, and move the `Unreleased` entries of its `CHANGELOG.md` under the new version with the date.
2. Merge to `master`. The package's publish workflow tests it with its own dependencies, builds it, checks that it is installable from PyPI, i.e. that the workspace packages it depends on are published, and publishes it. A version already on PyPI is skipped, so changes without a version bump release nothing.
3. A package depending on other packages of the workspace is published after them; run its workflow by hand (Actions, "Run workflow") once they are on PyPI.

Each package publishes as a PyPI trusted publisher: on PyPI, its project (or a pending publisher for a new one) trusts the repository `BlackWild/qiu`, its workflow `publish-<package>.yml` and the environment `pypi`.
