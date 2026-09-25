# User Guide

`qiu-python-encore` currently consists of a single module, [`qiu_python_encore.enum`][qiu_python_encore.enum], with the class [`ExtendedEnum`][qiu_python_encore.enum.ExtendedEnum]. This guide explains its semantics, the reasons behind them and the cases where they differ from a standard `Enum`.

## Defining an extended enum

An extended enum is defined like any `Enum`, by subclassing [`ExtendedEnum`][qiu_python_encore.enum.ExtendedEnum] instead of `enum.Enum`:

```python
from qiu_python_encore.enum import ExtendedEnum


class IndexOrdering(ExtendedEnum):
    NATURAL = "natural"
    FFT = "fft"
    CENTERED = "centered"


assert IndexOrdering.FFT.name == "FFT"
assert IndexOrdering.FFT.value == "fft"
assert IndexOrdering["FFT"] is IndexOrdering.FFT  # lookup by name, as for any Enum
```

Everything the standard library offers for enums keeps working: iteration in definition order, lookup by name with `IndexOrdering["FFT"]`, lookup by value with `IndexOrdering("fft")`, aliases for repeated values, `name` and `value`, `repr` and `str`. `ExtendedEnum` defines no members itself, so it can be subclassed; an enum with members cannot, as usual.

The values can be of any hashable type. Strings are the typical choice, since they are what configuration files and command lines provide, but integers, floats or tuples work the same way.

## Equality

The equality of a member `member` with an object `other` is decided as follows:

| `other`                                    | `member == other`                                         |
| ------------------------------------------ | --------------------------------------------------------- |
| a member of the same enum                  | `member is other`, i.e. identity, as for a standard `Enum` |
| a member of any other `Enum`               | `member.value == other.value`                              |
| any other object, e.g. a raw value         | `member.value == other`                                    |

Consequently:

- A member equals its raw value, `Color.RED == "red"`, from both sides: `"red" == Color.RED` holds too, since `str.__eq__` gives way to the member's `__eq__`.
- A member does not equal its name: `Color.RED != "RED"`. Use `Color["RED"]` to look members up by name.
- A member equals a member of another enum with the same value, whether that enum is an `ExtendedEnum` or a standard `Enum`, and in both orders of the operands.
- Membership tests in containers of raw values work: `Color.RED in ["red", "blue"]`.
- `!=` is the negation of `==`, as Python derives it.

```python
from enum import Enum

from qiu_python_encore.enum import ExtendedEnum


class Color(ExtendedEnum):
    RED = "red"
    GREEN = "green"


class Shade(ExtendedEnum):
    RED = "red"


class PlainColor(Enum):
    RED = "red"


assert Color.RED == "red" and "red" == Color.RED
assert Color.RED != "RED"
assert Color.RED == Shade.RED and Color.RED is not Shade.RED
assert Color.RED == PlainColor.RED and PlainColor.RED == Color.RED
assert Color.RED in ["red", "blue"]
```

!!! note "Comparisons with arrays"
    If comparing the value with `other` does not give a single truth value, as for a NumPy array of several elements, the member returns `NotImplemented` and lets `other` decide. For an array this gives the elementwise comparison with the raw value: for a member `Number.ONE` of value `1`, `Number.ONE == numpy.array([1, 2])` is `array([True, False])`, like `1 == numpy.array([1, 2])`. An array of a single element has a truth value, so the comparison gives a plain `bool` instead.

Ordering comparisons, `<`, `<=`, `>` and `>=`, are not defined, as for a standard `Enum`.

## Hashing

Members hash like their raw values, `hash(Color.RED) == hash("red")`. Together with the equality, this makes raw values and members interchangeable wherever Python hashes:

- as dict keys: a dict with the key `Color.RED` can be indexed with `"red"`, and vice versa;
- in sets: `"red" in {Color.RED}` holds;
- with `functools.cache` and `functools.lru_cache`, whose caches are dicts keyed by the arguments.

A standard `Enum` hashes its members by name, so they could not be looked up by value, and its equality is identity, so a member and its value would be different keys.

## Normalizing raw values

Calling the enum with a value returns the member of that value, and a member is returned as it is. This is the idiomatic way to normalize an argument that may be either:

```python
from qiu_python_encore.enum import ExtendedEnum


class Color(ExtendedEnum):
    RED = "red"
    GREEN = "green"


class Shade(ExtendedEnum):
    RED = "red"


def describe(color: Color | str) -> str:
    color = Color(color)  # a member, whatever was given
    return f"{color.name.lower()} ({color.value})"


assert describe("red") == describe(Color.RED) == "red (red)"
assert Color(Shade.RED) is Color.RED  # members of other enums map by value, too

try:
    Color("blue")
except ValueError:
    pass  # unknown values are rejected, as for any Enum
else:
    raise AssertionError("expected a ValueError")
```

The classes of [qiu-signals](../../qiu-signals/) normalize their enum arguments this way, so that e.g. an axis can be created with the ordering `"fft"` instead of `IndexOrdering.FFT`.

## Listing the values

The classmethod [`list`][qiu_python_encore.enum.ExtendedEnum.list] returns the raw values of the members, in definition order. Aliases, i.e. members repeating an earlier value, are not listed, as iterating an `Enum` skips them. It is convenient wherever the accepted values are needed as plain data, e.g. as the `choices` of an `argparse` argument or the options of a user interface.

## Pitfalls

The equality with raw values and with other enums is the point of `ExtendedEnum`, but it has consequences to be aware of:

- **Members of different enums with the same value collapse in sets and dicts.** `{Color.RED, Shade.RED}` has a single element, and `{Color.RED: 1, "red": 2}` has a single key, since the keys are equal and hash equally. Use a standard `Enum`, or key by `(type(member), member)`, where the kinds must stay apart.
- **The equality is that of the values.** A member of value `1` equals `1.0` and `True`, since `1 == 1.0 == True` in Python, and a member of value `"1"` does not equal `1`.
- **Members are not instances of the value's type.** Unlike `enum.StrEnum`, an `ExtendedEnum` member of a string value is not a `str`: string methods, concatenation and `json.dumps` do not accept it. Pass `member.value` to such code.
- **Type checkers see the members only.** Annotate parameters accepting both as `Color | str`, and normalize them with `Color(value)` before relying on member attributes.
