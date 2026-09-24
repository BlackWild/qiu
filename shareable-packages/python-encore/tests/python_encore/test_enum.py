"""Unit tests for enum.py."""

import functools
from enum import Enum

import pytest
from python_encore.enum import ExtendedEnum


class Color(ExtendedEnum):
    """An extended enum for testing."""

    RED = "red"
    GREEN = "green"


class Shade(ExtendedEnum):
    """Another extended enum sharing a value with Color."""

    RED = "red"
    DARK = "dark"


class PlainColor(Enum):
    """A standard enum sharing a value with Color."""

    RED = "red"


class Number(ExtendedEnum):
    """An extended enum with integer values."""

    ONE = 1
    TWO = 2


class TestList:
    """Test the list classmethod."""

    def test_values_in_definition_order(self):
        """Test that the raw values are listed in definition order."""
        assert Color.list() == ["red", "green"]
        assert Number.list() == [1, 2]


class TestEquality:
    """Test the equality of the members."""

    def test_same_enum(self):
        """Test that members of the same enum compare by identity."""
        assert Color.RED == Color.RED
        assert Color.RED != Color.GREEN

    def test_raw_values(self):
        """Test that members compare equal to their raw values, from both sides."""
        assert Color.RED == "red"
        assert "red" == Color.RED  # noqa: SIM300
        assert Color.RED != "green"
        assert Number.ONE == 1
        assert Number.ONE != 2

    def test_names_are_not_values(self):
        """Test that members do not compare equal to their names."""
        assert Color.RED != "RED"

    def test_other_enums_by_value(self):
        """Test that members compare equal to members of other enums by value."""
        assert Color.RED == Shade.RED
        assert Color.RED == PlainColor.RED
        assert Color.GREEN != Shade.DARK

    def test_membership(self):
        """Test membership in containers of raw values."""
        assert Color.RED in ["red", "blue"]
        assert Color.GREEN not in ["red", "blue"]

    def test_unrelated_objects(self):
        """Test that members do not equal unrelated objects."""
        assert Color.RED != object()  # noqa: SIM300
        assert Number.ONE != "1"

    def test_arrays_compare_like_the_raw_value(self):
        """Test that members defer to arrays, which compare elementwise."""
        np = pytest.importorskip("numpy")
        array = np.array([1, 2])
        assert Number.ONE.__eq__(array) is NotImplemented
        np.testing.assert_array_equal(Number.ONE == array, 1 == array)  # noqa: SIM300

    def test_construction_from_raw_value(self):
        """Test that members are recovered from their raw values."""
        assert Color("red") is Color.RED
        assert Color(Color.RED) is Color.RED
        with pytest.raises(ValueError):
            Color("blue")


class TestHashing:
    """Test the hashing of the members."""

    def test_members_are_hashable(self):
        """Test that members can be used in sets and as dict keys."""
        assert {Color.RED, Color.RED, Color.GREEN} == {Color.RED, Color.GREEN}
        assert {Color.RED: 1}[Color.RED] == 1

    def test_hash_is_consistent_with_equality(self):
        """Test that equal objects hash equally, so lookups by raw value work."""
        assert hash(Color.RED) == hash("red")
        assert {Color.RED: 1}["red"] == 1
        assert "red" in {Color.RED}
        assert hash(Color.RED) == hash(Shade.RED)

    def test_caching(self):
        """Test that members can be passed to cached functions."""

        @functools.cache
        def upper(color: Color) -> str:
            return color.value.upper()

        assert upper(Color.RED) == "RED"
        assert upper.cache_info().currsize == 1
