# Python Encore

[![PyPI](https://img.shields.io/pypi/v/python-encore)](https://pypi.org/project/python-encore/) [![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)](https://www.python.org/) [![License: MIT](https://img.shields.io/badge/license-MIT-green)](https://github.com/BlackWild/qiu/blob/master/LICENSE) [![CI](https://github.com/BlackWild/qiu/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/BlackWild/qiu/actions/workflows/ci.yml) [![Docs](https://img.shields.io/badge/docs-mkdocs-blue)](https://blackwild.github.io/qiu/python-encore/)

Small enhancements of the Python standard library, used across this monorepo. It has no dependencies.

## Installation

```sh
pip install python-encore
```

## `python_encore.enum.ExtendedEnum`

An `Enum` whose members also compare equal to their raw values, so that code can accept either a member or its value, e.g. from a configuration file or a `hypothesis` strategy:

```python
from python_encore.enum import ExtendedEnum


class Color(ExtendedEnum):
    RED = "red"
    GREEN = "green"


assert Color.RED == "red"
assert Color("red") is Color.RED  # normalize raw values to members
assert Color.list() == ["red", "green"]  # the raw values, in definition order
assert {Color.RED: 1}["red"] == 1  # members hash like their values
```

Compared to a standard `Enum`:

- A member equals its raw value (`Color.RED == "red"`), but not its name (`Color.RED != "RED"`).
- A member equals a member of any other `Enum` with the same value.
- Members hash like their raw values, consistently with the equality above, so they can be used in sets, as dict keys and with `functools.cache`.
- `list()` returns the raw values of the members.

## Documentation

The documentation, with the API reference from the docstrings, is built from `docs/` with MkDocs and published at <https://blackwild.github.io/qiu/python-encore/>. To serve it locally, from the repository root:

```sh
uv run --group docs mkdocs serve -f shareable-packages/python-encore/mkdocs.yml
```

## Tests

From the repository root:

```sh
uv run pytest shareable-packages/python-encore
```
