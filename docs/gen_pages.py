"""Generate the pages and the navigation of the documentation with MkDocs' `gen-files`.

Run for the site of the monorepo, the root `mkdocs.yml`, it collects the pages of every
package, its `docs/`, into a directory of the site named as the package's directory,
e.g. `qiu-signals/`, and writes the navigation of the site, `SUMMARY.md` for the
`literate-nav` plugin: the landing page, then the packages in the sections of
`SECTIONS`. Run for the `mkdocs.yml` of a single package, the preview of its
documentation alone, it writes the navigation of that package.

Either way, the API reference of a package is generated from the modules of its `src`:
one page per module under `reference/`, a package's page being its `__init__` module,
which mkdocstrings renders from the docstrings. Private modules, starting with `_`, are
skipped.
"""

from pathlib import Path

import mkdocs_gen_files
from mkdocs.structure.files import InclusionLevel

REPOSITORY = Path(__file__).resolve().parents[1]
PACKAGE_GROUPS = ["packages", "packages-dev", "apps"]

# The sections of the navigation of the site and their packages, in the order of the
# building blocks, from the foundations to the applications, followed by the packages
# most readers need least: the improvements of other libraries and the test helpers.
SECTIONS = {
    "Packages": [
        "packages/qiu-signals",
        "packages/qiu-quantum-computing",
        "packages/qiu-mps-initializer",
        "packages/qiu-hamiltonian-simulation",
        "packages/qiu-classical-simulation",
    ],
    "Applications": [
        "apps/wave_optics_propagation",
        "apps/wave_optics_propagation_qutip",
    ],
    "Encore packages": [
        "packages/qiu-python-encore",
        "packages/qiu-qiskit-encore",
        "packages/qiu-qiskit-aer-encore",
    ],
    "Development packages": [
        "packages-dev/python-pytest-helper",
        "packages-dev/qiskit-pytest-helper",
    ],
}

# the pages every package has, in the order of its navigation; others follow them
PAGES = {"index.md": "Home", "user-guide.md": "User Guide", "examples.md": "Examples"}


def reference_pages(package: Path) -> list[tuple[tuple[str, ...], str]]:
    """Return the public modules of a package's `src` and the paths of their pages.

    Args:
        package: The directory of the package.

    Returns:
        The parts of each module's name and the path of its page, relative to the
        package's documentation, e.g. `(("qiu_signals", "signal"),
        "reference/qiu_signals/signal.md")`, sorted so that packages precede their
        modules.
    """
    source = package / "src"
    pages = []
    for path in sorted(source.rglob("*.py")):
        parts = path.relative_to(source).with_suffix("").parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
            page = Path("reference", *parts, "index.md")
        else:
            page = Path("reference", *parts).with_suffix(".md")
        if any(part.startswith("_") for part in parts):
            continue
        pages.append((parts, page.as_posix()))
    return pages


def write_reference(package: Path, prefix: str) -> list[str]:
    """Write the API reference pages of a package, and return their navigation.

    Args:
        package: The directory of the package.
        prefix: The directory of the package's documentation in the site, e.g.
            `"qiu-signals/"`, or `""` for the preview of a single package.

    Returns:
        The lines of the navigation, one list item per module, the top-level package
        being the item "API Reference" and its modules nested in it.
    """
    lines = []
    for parts, page in reference_pages(package):
        with mkdocs_gen_files.open(prefix + page, "w") as file:
            file.write(f"::: {'.'.join(parts)}\n")
        title = "API Reference" if len(parts) == 1 else f"`{parts[-1]}`"
        lines.append("    " * (len(parts) - 1) + f"* [{title}]({prefix}{page})")
    return lines


def package_navigation(package: Path, prefix: str) -> list[str]:
    """Return the navigation of a package: its pages, then its API reference.

    Args:
        package: The directory of the package.
        prefix: The directory of the package's documentation in the site, see
            `write_reference`.

    Returns:
        The lines of the navigation, one list item per page.
    """
    pages = sorted(
        (
            path.relative_to(package / "docs").as_posix()
            for path in (package / "docs").rglob("*.md")
        ),
        key=lambda page: (
            list(PAGES).index(page) if page in PAGES else len(PAGES),
            page,
        ),
    )
    lines = [
        f"* [{PAGES.get(page, Path(page).stem)}]({prefix}{page})"
        for page in pages
        if page != "index.md"
    ]
    return lines + write_reference(package, prefix)


def site_navigation() -> list[str]:
    """Collect the documentation of all packages, and return the navigation of the site.

    Returns:
        The lines of the navigation: the landing page, and a section per entry of
        `SECTIONS`, with an item per package whose page is its home page.

    Raises:
        ValueError: If the packages with documentation are not those of `SECTIONS`.
    """
    documented = {
        docs.parent.relative_to(REPOSITORY).as_posix()
        for group in PACKAGE_GROUPS
        for docs in (REPOSITORY / group).glob("*/docs")
    }
    listed = {package for packages in SECTIONS.values() for package in packages}
    if documented != listed:
        raise ValueError(
            "The packages with documentation must be those of SECTIONS in "
            f"docs/gen_pages.py: missing {sorted(documented - listed)}, unknown "
            f"{sorted(listed - documented)}."
        )

    lines = ["* [Home](index.md)"]
    for section, packages in SECTIONS.items():
        lines.append(f"* {section}")
        for name in packages:
            package = REPOSITORY / name
            prefix = f"{package.name}/"
            for path in sorted((package / "docs").rglob("*")):
                if path.is_file():
                    target = prefix + path.relative_to(package / "docs").as_posix()
                    with mkdocs_gen_files.open(target, "wb") as file:
                        file.write(path.read_bytes())
            lines.append(f"    * [{package.name}]({prefix}index.md)")
            lines += ["        " + line for line in package_navigation(package, prefix)]
    return lines


config_directory = Path(mkdocs_gen_files.config["config_file_path"]).resolve().parent
if config_directory == REPOSITORY:
    navigation = site_navigation()
else:
    navigation = ["* [Home](index.md)", *package_navigation(config_directory, "")]

with mkdocs_gen_files.open("SUMMARY.md", "w") as summary:
    summary.write("\n".join(navigation) + "\n")
# the navigation is read by literate-nav, but is no page of the site itself
summary_file = mkdocs_gen_files.files.get_file_from_path("SUMMARY.md")
assert summary_file is not None
summary_file.inclusion = InclusionLevel.EXCLUDED
