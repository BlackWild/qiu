"""The encoding of the integer indices of an axis in the basis states of qubits.

An axis of `2**n` samples is represented by `n` qubits, with the sample at array
position `k` in the basis state `|k>`, `k = sum_i 2^i x_i` in little-endian order.
The basis state thus encodes the integer index `axis.index[k]`, which depends on the
index ordering of the axis:

- `NATURAL`: the unsigned integer `sum_i 2^i x_i`.
- `FFT`: the two's complement integer `-2^(n-1) x_(n-1) + sum_(i<n-1) 2^i x_i`.
- `CENTERED`: the two's complement integer of the bits with the most significant one
  flipped, `-2^(n-1) (1 - x_(n-1)) + sum_(i<n-1) 2^i x_i`.
"""

from python_signals.integer_axis import IndexOrdering, IntegerAxis


def num_qubits_of(axis: IntegerAxis) -> int:
    """Return the number of qubits representing the axis.

    Args:
        axis: An axis of `2**n` samples, with `n >= 1`.

    Returns:
        The number of qubits `n`.

    Raises:
        ValueError: If the axis does not have `2**n` samples with `n >= 1`.
    """
    num_qubits = axis.size.bit_length() - 1
    if num_qubits < 1 or axis.size != 2**num_qubits:
        raise ValueError(
            f"An axis represented by qubits must have 2**n >= 2 samples, got "
            f"{axis.size}."
        )
    return num_qubits


def is_msb_flipped(ordering: IndexOrdering) -> bool:
    """Whether the bit weights refer to the flipped most significant bit.

    For such orderings, circuits built from the bit weights must flip the most
    significant qubit before and after applying them.

    Returns:
        True for the `CENTERED` ordering, False otherwise.
    """
    return IndexOrdering(ordering) == IndexOrdering.CENTERED


def bit_weights(num_qubits: int, ordering: IndexOrdering) -> list[int]:
    """Return the weights `w_i` of the bits in the encoded integer indices.

    The integer encoded by the basis state `|x_(n-1) ... x_0>` is `sum_i w_i x'_i`,
    where `x'_i = x_i`, except for the most significant bit, which is flipped for
    the orderings of `is_msb_flipped`.

    Args:
        num_qubits: The number of qubits `n`.
        ordering: The index ordering of the axis.

    Returns:
        The weights `[w_0, ..., w_(n-1)]`.
    """
    ordering = IndexOrdering(ordering)
    weights = [2**i for i in range(num_qubits)]

    if ordering == IndexOrdering.NATURAL:
        return weights
    if ordering in (IndexOrdering.FFT, IndexOrdering.CENTERED):
        weights[-1] = -weights[-1]
        return weights
    raise NotImplementedError(f"Unsupported index ordering: {ordering.value!r}")
