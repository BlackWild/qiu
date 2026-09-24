# Python Encore

Small enhancements of the Python standard library, used across this monorepo. It has no dependencies.

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

## Tests

From the repository root:

```sh
uv run pytest shareable-packages/python-encore
```
