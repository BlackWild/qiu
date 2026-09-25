"""Run the Python examples of the documentation and the READMEs of all packages.

Every fenced `python` block of a package's `README.md` and `docs/*.md`, and of the root
`README.md`, is executed on its own, in a fresh namespace and a temporary working
directory, with a non-interactive Matplotlib backend. A block preceded by the line
`<!-- no-test -->` is skipped, e.g. one that needs a cluster or LaTeX.
"""

import re
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
PACKAGE_GROUPS = ["packages", "shareable-packages", "dev-packages", "apps"]
BLOCK = re.compile(r"(<!-- no-test -->\s*\n)?```python\n(.*?)```", re.DOTALL)


def documentation_files() -> list[Path]:
    """Return the Markdown files whose examples are run."""
    files = [REPOSITORY / "README.md"]
    for group in PACKAGE_GROUPS:
        for package in sorted((REPOSITORY / group).iterdir()):
            files += [package / "README.md", *sorted((package / "docs").glob("*.md"))]
    return [file for file in files if file.is_file()]


def examples() -> list:
    """Return the examples as pytest parameters, identified by file and position."""
    parameters = []
    for file in documentation_files():
        for number, match in enumerate(BLOCK.finditer(file.read_text()), start=1):
            name = f"{file.relative_to(REPOSITORY).as_posix()}#{number}"
            marks = (
                [pytest.mark.skip(reason="marked no-test")] if match.group(1) else []
            )
            parameters.append(pytest.param(match.group(2), id=name, marks=marks))
    return parameters


@pytest.mark.parametrize("code", examples())
def test_example(code: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Test that the example runs without errors, including its assertions."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MPLBACKEND", "Agg")
    exec(compile(code, "<example>", "exec"), {"__name__": "__example__"})
