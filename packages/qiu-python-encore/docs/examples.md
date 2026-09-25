# Examples

## Options from a configuration file

Configuration files store options as plain strings. With an extended enum, the parsed values can be compared with members directly, and normalized to members where the code needs their attributes.

```python
import json
from dataclasses import dataclass

from qiu_python_encore.enum import ExtendedEnum


class Solver(ExtendedEnum):
    EXACT = "exact"
    SAMPLED = "sampled"


@dataclass(frozen=True)
class Settings:
    solver: Solver
    shots: int

    @classmethod
    def from_json(cls, text: str) -> "Settings":
        values = json.loads(text)
        return cls(solver=Solver(values["solver"]), shots=values["shots"])

    def to_json(self) -> str:
        return json.dumps({"solver": self.solver.value, "shots": self.shots})


raw = json.loads('{"solver": "sampled", "shots": 1000}')
assert raw["solver"] == Solver.SAMPLED  # the raw string already compares equal

settings = Settings.from_json('{"solver": "sampled", "shots": 1000}')
assert settings.solver is Solver.SAMPLED
assert Settings.from_json(settings.to_json()) == settings
```

The raw string compares equal to the member before any conversion, and `Solver(...)` turns it into the member itself; `to_json` stores the raw value, since members are not JSON serializable.

## Choices of a command line

`list()` gives the raw values of the members, which are what `argparse` needs for the `choices` of an option; the parsed string is then normalized to a member.

```python
import argparse

from qiu_python_encore.enum import ExtendedEnum


class Backend(ExtendedEnum):
    CPU = "cpu"
    GPU = "gpu"


parser = argparse.ArgumentParser()
parser.add_argument("--backend", choices=Backend.list(), default=Backend.CPU.value)

arguments = parser.parse_args(["--backend", "gpu"])
assert arguments.backend == Backend.GPU
assert Backend(arguments.backend) is Backend.GPU
assert Backend(parser.parse_args([]).backend) is Backend.CPU
assert Backend.list() == ["cpu", "gpu"]
```

The help text of the parser lists the choices `cpu` and `gpu`, and invalid values are rejected by `argparse` before they reach the enum.

## Lookup tables keyed by members

Members hash like their raw values, so a table keyed by members can be indexed with raw values, e.g. those of a data file, and vice versa.

```python
from qiu_python_encore.enum import ExtendedEnum


class Domain(ExtendedEnum):
    POSITION = "position"
    MOMENTUM = "momentum"


units = {Domain.POSITION: "m", Domain.MOMENTUM: "kg m / s"}

records = [("position", 1.5), ("momentum", 2.0e-3)]
labels = [f"{value} {units[domain]}" for domain, value in records]

assert labels == ["1.5 m", "0.002 kg m / s"]
assert "momentum" in units
assert set(units) == {"position", "momentum"}
```

Neither the table nor the records need to be converted: the raw strings of the records find the members' entries.

## Caching functions of members

`functools.cache` keys its cache by the arguments, which must be hashable. A member and its raw value hash and compare equally, so they share one cache entry.

```python
import functools

from qiu_python_encore.enum import ExtendedEnum


class Ordering(ExtendedEnum):
    NATURAL = "natural"
    CENTERED = "centered"


@functools.cache
def indices(ordering: Ordering | str, size: int) -> tuple[int, ...]:
    if Ordering(ordering) is Ordering.NATURAL:
        return tuple(range(size))
    return tuple(range(-(size // 2), size - size // 2))


assert indices(Ordering.CENTERED, 4) == (-2, -1, 0, 1)
assert indices("centered", 4) == (-2, -1, 0, 1)
assert indices.cache_info().hits == 1  # the second call was served from the cache
assert indices.cache_info().currsize == 1
```

The second call, with the raw value, is a cache hit of the first one, with the member, so the function body runs once.
