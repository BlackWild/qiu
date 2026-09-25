# Contributing to Qiu

Thank you for considering a contribution. This guide covers how to set up the monorepo, the conventions its packages follow, and how changes get tested, documented and released. Participation follows the [Code of Conduct](CODE_OF_CONDUCT.md).

## Setting up

The monorepo is a [uv](https://docs.astral.sh/uv/) workspace. With uv installed, from the repository root:

```sh
uv sync --all-packages --group docs  # all packages, editable, with the test and documentation tools
uvx pre-commit install               # lint and format each commit
```

The Python version is the one of `.python-version`; the packages support Python 3.11 to 3.14. Alternatively, open the repository in its dev container (`.devcontainer/`), which does the above.

## Layout

| Directory             | Contains                                                           | May depend on                           |
| --------------------- | ------------------------------------------------------------------ | --------------------------------------- |
| `shareable-packages/` | Libraries useful beyond quantum computing, e.g. `python-signals`   | each other                              |
| `packages/`           | Libraries for quantum computing, with circuits of Qiskit           | shareable packages, each other          |
| `dev-packages/`       | Test helpers, never published                                      | shareable packages and packages         |
| `apps/`               | Applications, e.g. the simulations of a paper, never published     | everything above                        |

Dependencies point down this table only: a shareable package never imports a quantum computing framework such as Qiskit or QuTiP, and no library imports an app. Before writing new functionality, check whether a package already provides it; reuse it, or extend the package it belongs in.

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

- Format and lint with `ruff` (`uv run ruff format .` and `uv run ruff check --fix .`), configured in `ruff.toml`, and type check with `pyright` (`uvx pyright --pythonpath .venv/bin/python apps shareable-packages dev-packages packages`).
- Every module, class and public function has a Google-style docstring: a summary line, then `Args:`, `Returns:` and `Raises:` where applicable. The API reference of the documentation is generated from them.
- Annotate the types of all arguments and results: accept `npt.ArrayLike` (or `Statevector | npt.ArrayLike`) as input where any array-like works, and return the most precise type, e.g. `npt.NDArray[np.complex128]`.
- Name things by what they are, in full words; physical conventions (units, index orderings, signs, global phases) are stated in the docstrings.
- Choices between implementations are enums checked exhaustively: every member has its own branch, and unknown ones raise `NotImplementedError` (see `qiskit_encore.synthesis_method.SynthesisMethod`).
- No hidden constants: tolerances and similar parameters are arguments, or follow the standard of the library at hand, e.g. Qiskit's `Statevector` equality.
- Scripts of the apps are section-based (`# %%`) Python files, not notebooks.

## Tests

- Tests live in `tests/<import_name>/test_<module>.py`, one test class per unit, with a docstring saying what is tested. Run them with `uv run pytest -n auto`, or for one package, e.g. `uv run pytest packages/qiskit-encore`.
- Prefer property-based tests with [Hypothesis](https://hypothesis.readthedocs.io/), using the shared strategies of `python-pytest-helper` (numbers, axes, signals) and `qiskit-pytest-helper` (quantum states, qubit axes).
- Compare floating-point results with the shared assertions, never with a tolerance chosen per test: `python_pytest_helper.assertions.assert_close` for numbers and arrays (relative tolerance, with an absolute floor at the underflow), `assert_close_in_norm` for vectors computed as a whole, and `qiskit_pytest_helper.assertions.assert_equal_states` and `assert_equal_operators` for states and operators (Qiskit's equality, global phase included).
- Approximations are tested against bounds derived from the approximation, and exact results exactly.
- A package's tests may only use the dependencies it declares, including its `dev` dependency group: the publish workflows test each package in an environment of only those.
- On GitHub Actions, the tests run with the Hypothesis profile `ci` of `conftest.py`, without deadlines per example.

## Documentation

- Each package's documentation has a home page (`docs/index.md`: overview, installation, quick start), a user guide (`docs/user-guide.md`: concepts, conventions, pitfalls), examples (`docs/examples.md`: worked use cases) and the generated API reference.
- Every `python` code block of the documentation and the READMEs is executed by `docs/tests/test_examples.py`, as part of the tests: make each block self-contained, and assert the results it claims. Mark a block that cannot run, e.g. one needing a cluster, with a preceding `<!-- no-test -->` line.
- Build the documentation of all packages with `uv run --group docs python docs/build_all.py`, which builds strictly, so broken links and references fail; serve one package's with `uv run --group docs mkdocs serve -f <group>/<package>/mkdocs.yml`.
- Links between packages in their documentation use the paths of their sites: `../<package>/` from a home page `index.md`, and `../../<package>/` from the other pages, which MkDocs serves one directory deeper (`<package>/user-guide/`). The build checks that every link of the site resolves. Links in READMEs are absolute GitHub URLs, so that they also work on PyPI.

## Adding a package

1. Create `<group>/<package>/` with the layout above; the workspace picks up every directory of the groups.
2. Write its `pyproject.toml` like the existing ones: the metadata (description, `license = "MIT"`, `license-files`, authors, keywords, classifiers including the supported Python versions, and the `project.urls` of its source, documentation, repository, issues and changelog), `uv_build` as the build backend, its dependencies, a `dev` dependency group with what its tests need, and `tool.uv.sources` of the workspace packages it uses. Packages that are not published get the classifier `Private :: Do Not Upload`.
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
