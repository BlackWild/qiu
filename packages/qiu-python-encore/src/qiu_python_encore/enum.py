"""Better Enum for python."""

from enum import Enum
from typing import Any


class ExtendedEnum(Enum):
    """Enum whose members also compare equal to their raw values.

    A member equals another member of the same enum by identity, a member of any
    other Enum with the same value, and its raw value itself, e.g.
    `Color.RED == "red"`. Members hash like their values, so the hashing is
    consistent with this equality and members can be used in sets, as dict keys
    and with `functools.cache`.
    """

    @classmethod
    def list(cls) -> list[Any]:
        """Return the raw values of the members, in definition order."""
        return [member.value for member in cls]

    def __eq__(self, other: object) -> bool:
        """Compare against members of any Enum by value, or against raw values."""
        if isinstance(other, self.__class__):
            return self is other
        if isinstance(other, Enum):
            return self.value == other.value

        try:
            return bool(self.value == other)
        except Exception:
            # e.g. arrays, whose comparison has no single truth value
            return NotImplemented

    def __hash__(self) -> int:
        """Hash like the raw value, consistently with the equality."""
        return hash(self.value)
