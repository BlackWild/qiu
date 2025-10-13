"""Direct implementation of phase operator circuits up to 3rd order."""

from qiskit.circuit import QuantumCircuit
from qiskit_signals.helper_types import EncodingType


# Binary coefficients for local use in this file
def c_twos_complement(i: int, n: int) -> int:
    """Two's complement binary coefficients.

    Such that any integer can be represented as
        $x = -2^{n-1} + sum_{i=0}^{n-2} 2^i x_i = sum_{i=0}^{n-1} c(i, n) x_i$,
    where $x_i$ are the binary digits.
    """

    if i == n - 1:
        return -(2 ** (i))
    if i < n - 1:
        return 2 ** (i)

    raise ValueError("Out of range!")


def c_unsigned(i: int, n: int) -> int:
    """Unsigned binary coefficients.

    Such that any integer can be represented as
        $x = sum_{i=0}^{n-1} 2^i x_i = sum_{i=0}^{n-1} c(i, n) x_i$,
    where $x_i$ are the binary digits.
    """

    if i < n:
        return 2 ** (i)

    raise ValueError("Out of range!")


def c(i: int, n: int, encoding: EncodingType) -> int:
    """Binary coefficients.

    Such that any integer can be represented as
        $x = sum_{i=0}^{n-1} c(i, n) x_i$, where $x_i$ are the binary digits.
    """

    if encoding == EncodingType.TWOS_COMPLEMENT:
        return c_twos_complement(i, n)
    elif encoding == EncodingType.UNSIGNED:
        return c_unsigned(i, n)
    else:
        raise ValueError(f"Unsupported encoding type: {encoding}")


class Order1DirectPhase(QuantumCircuit):
    """1st-order phase circuit as e^(i * coef * x)."""

    def __init__(self, num_qubits: int, coef: float, encoding: EncodingType) -> None:
        """Initialize the 1st-order phase circuit."""
        super().__init__(num_qubits)
        self.coef = coef
        self.encoding = encoding

        for i in range(num_qubits):
            self.p(coef * c(i, num_qubits, encoding), i)


class Order2DirectPhase(QuantumCircuit):
    """2nd-order phase circuit as e^(i * coef * x^2)."""

    def __init__(self, num_qubits: int, coef: float, encoding: EncodingType) -> None:
        """Initialize the 2nd-order phase circuit."""
        super().__init__(num_qubits)
        self.coef = coef
        self.encoding = encoding

        n = num_qubits
        # the case of k = 1
        for i in range(n):
            self.p(coef * (c(i, n, encoding) ** 2), i)

        # the case of k = 2
        for i in range(n):
            for j in range(i):
                self.cp(coef * 2 * (c(i, n, encoding) * c(j, n, encoding)), i, j)


class Order3DirectPhase(QuantumCircuit):
    """3rd-order phase circuit as e^(i * coef * x^3)."""

    def __init__(self, num_qubits: int, coef: float, encoding: EncodingType) -> None:
        """Initialize the 3rd-order phase circuit."""
        super().__init__(num_qubits)
        self.coef = coef
        self.encoding = encoding

        n = num_qubits

        # the case of k = 1
        for i in range(n):
            self.p(coef * (c(i, n, encoding) ** 3), i)

        # the case of k = 2
        for i in range(n):
            for j in range(i):
                self.cp(
                    coef
                    * 3
                    * (
                        c(i, n, encoding) ** 2 * c(j, n, encoding)
                        + c(i, n, encoding) * c(j, n, encoding) ** 2
                    ),
                    i,
                    j,
                )

        # the case of k = 3
        for i in range(n):
            for j in range(i):
                for k in range(j):
                    self.mcp(
                        coef
                        * 6
                        * (c(i, n, encoding) * c(j, n, encoding) * c(k, n, encoding)),
                        [i, j],
                        k,
                    )
