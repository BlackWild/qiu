import matplotlib.pyplot as plt
import numpy as np


def plot_wavefunction(psi, plot_size_scale=1, normalize=True):
    abs_part = np.abs(psi)
    if normalize:
        abs_part = abs_part / np.linalg.norm(abs_part)
    phase_part = np.angle(psi)

    num_of_basis = len(abs_part)
    num_of_qubits = (num_of_basis - 1).bit_length()
    x_axis = np.arange(num_of_basis)

    # ket_labels = [
    #     rf"$|{np.binary_repr(x, width=num_of_qubits)}\rangle$" for x in x_axis
    # ]
    # example:
    # KET_LABELS = [r'$|000\rangle$', r'$|001\rangle$', r'$|010\rangle$', r'$|011\rangle$', r'$|100\rangle$', r'$|101\rangle$', r'$|110\rangle$', r'$|111\rangle$']

    PHASE_LABELS = [
        r"$-\pi$",
        r"$-\frac{3\pi}{4}$",
        r"$-\frac{\pi}{2}$",
        r"$-\frac{\pi}{4}$",
        r"$0$",
        r"$\frac{\pi}{4}$",
        r"$\frac{\pi}{2}$",
        r"$\frac{3\pi}{4}$",
        r"$\pi$",
    ]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(2 * 6.4 * plot_size_scale, 1 * 4.8 * plot_size_scale)
    )
    ax1.plot(abs_part)
    ax2.plot(phase_part)
    ax2.set_ylim(-np.pi - 0.3, np.pi + 0.3)

    # ax1.set_xticks(x_axis, ket_labels, rotation=45)
    # ax2.set_xticks(x_axis, ket_labels, rotation=45)

    ax2.set_yticks(
        [
            -np.pi,
            -np.pi * 3 / 4,
            -np.pi / 2,
            -np.pi / 4,
            0,
            np.pi / 4,
            np.pi / 2,
            np.pi * 3 / 4,
            np.pi,
        ]
    )
    ax2.set_yticklabels(PHASE_LABELS)

    ax1.set_xlabel(r"$|x\rangle$")
    ax2.set_xlabel(r"$|x\rangle$")

    ax1.set_ylabel(r"$|\psi(x)|$")
    ax2.set_ylabel(r"$\angle\psi(x)$")

    return fig, (ax1, ax2)
