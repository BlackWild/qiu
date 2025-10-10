"""Direct implementation of phase operator circuits up to 3rd order."""

from qiskit.circuit import QuantumCircuit


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


def c(i: int, n: int, signed: bool) -> int:
    """Binary coefficients.

    Such that any integer can be represented as
        $x = sum_{i=0}^{n-1} c(i, n) x_i$, where $x_i$ are the binary digits.
    """

    if signed:
        return c_twos_complement(i, n)
    else:
        return c_unsigned(i, n)


class Order1DirectPhase(QuantumCircuit):
    """1st-order phase circuit as e^(i * coef * x)."""

    def __init__(self, num_qubits: int, coef: float, signed: bool) -> None:
        """Initialize the 1st-order phase circuit."""
        super().__init__(num_qubits)
        self.coef = coef
        self.signed = signed

        for i in range(num_qubits):
            self.p(coef * c(i, num_qubits, signed), i)


class Order2DirectPhase(QuantumCircuit):
    """2nd-order phase circuit as e^(i * coef * x^2)."""

    def __init__(self, num_qubits: int, coef: float, signed: bool) -> None:
        """Initialize the 2nd-order phase circuit."""
        super().__init__(num_qubits)
        self.coef = coef
        self.signed = signed

        n = num_qubits
        # the case of k = 1
        for i in range(n):
            self.p(coef * (c(i, n, signed) ** 2), i)

        # the case of k = 2
        for i in range(n):
            for j in range(i):
                self.cp(coef * 2 * (c(i, n, signed) * c(j, n, signed)), i, j)


class Order3DirectPhase(QuantumCircuit):
    """3rd-order phase circuit as e^(i * coef * x^3)."""

    def __init__(self, num_qubits: int, coef: float, signed: bool) -> None:
        """Initialize the 3rd-order phase circuit."""
        super().__init__(num_qubits)
        self.coef = coef
        self.signed = signed

        n = num_qubits

        # the case of k = 1
        for i in range(n):
            self.p(coef * (c(i, n, signed) ** 3), i)

        # the case of k = 2
        for i in range(n):
            for j in range(i):
                self.cp(
                    coef
                    * 3
                    * (
                        c(i, n, signed) ** 2 * c(j, n, signed)
                        + c(i, n, signed) * c(j, n, signed) ** 2
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
                        * (c(i, n, signed) * c(j, n, signed) * c(k, n, signed)),
                        [i, j],
                        k,
                    )
