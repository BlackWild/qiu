"""Better Enum for python."""

from enum import Enum


class ExtendedEnum(Enum):
    """Extended Enum class with additional utility methods."""

    @classmethod
    def list(cls):
        """Returns a list of the enum values."""
        return list(map(lambda c: c.value, cls))

    # GPT-generated
    def __eq__(self, other):
        """Allow comparisons against enum members, other Enums, raw values, or names."""
        # same enum class -> default Enum equality
        if isinstance(other, self.__class__):
            return super().__eq__(other)

        # other is some Enum (different class) -> compare underlying values
        if isinstance(other, Enum):
            return self.value == other.value

        # compare directly to raw value (int, str, ...)
        try:
            if self.value == other:
                return True
        except Exception:
            pass

        # allow comparing to the name string
        # if isinstance(other, str) and self.name == other:
        #     return True

        return NotImplemented
