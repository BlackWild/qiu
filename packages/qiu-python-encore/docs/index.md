# qiu-python-encore

`qiu-python-encore` holds small enhancements of the Python standard library that the other packages of this monorepo share. It has no dependencies.

Currently it provides [`ExtendedEnum`][qiu_python_encore.enum.ExtendedEnum], an `Enum` whose members also compare equal to, and hash like, their raw values. Code can then accept either a member or its value, e.g. a string read from a configuration file, a command line or a `hypothesis` strategy, without converting it first. The enums of [qiu-signals](../qiu-signals/), e.g. its index orderings and axis domains, are built on it.

## Installation

```sh
pip install qiu-python-encore
```

Inside the monorepo, it is installed with all other packages by `uv sync --all-packages`.

## Quick start

```python
from qiu_python_encore.enum import ExtendedEnum


class Color(ExtendedEnum):
    RED = "red"
    GREEN = "green"


assert Color.RED == "red"  # a member equals its raw value
assert Color("red") is Color.RED  # normalize raw values to members
assert Color.list() == ["red", "green"]  # the raw values, in definition order
assert {Color.RED: 1}["red"] == 1  # members hash like their values
```

## Where next

- The [User Guide](user-guide.md) explains the semantics of `ExtendedEnum` in detail, with its pitfalls.
- The [Examples](examples.md) show typical uses: configuration values, command lines and caches.
- The [API Reference](reference/qiu_python_encore/index.md) documents every public object, generated from the docstrings.
