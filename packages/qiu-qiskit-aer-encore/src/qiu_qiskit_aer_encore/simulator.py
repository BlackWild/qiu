"""Aer simulators configured for the available hardware."""

from functools import cache
from typing import Any

from qiskit_aer import AerError, AerSimulator
from qiu_python_encore.enum import ExtendedEnum

GPU_OPTIONS: dict[str, Any] = {
    # accelerate with Nvidia's cuStateVec library
    "cuStateVec_enable": True,
    # distribute the simulation over all available GPU and CPU nodes
    "blocking_enable": True,
    # distribute the shots over the available GPUs
    "batched_shots_gpu": True,
}
"""The default options of simulators on GPUs, overridable per simulator."""


class AerDevice(ExtendedEnum):
    """The devices an Aer simulator can run on.

    - `AUTO`: a GPU if one is available, and the CPU otherwise.
    - `CPU`: the CPU, which is always available.
    - `GPU`: a GPU, raising an error if none is available.
    """

    AUTO = "auto"
    CPU = "cpu"
    GPU = "gpu"


@cache
def available_aer_devices() -> tuple[str, ...]:
    """Return the devices Aer can simulate on, e.g. `("CPU", "GPU")`.

    The devices are detected once and cached.
    """
    return tuple(AerSimulator().available_devices() or ())


def aer_simulator(
    device: AerDevice | str = AerDevice.AUTO, **options: Any
) -> AerSimulator:
    """Return an Aer simulator on the requested device.

    Args:
        device: The device to simulate on, see `AerDevice`.
        **options: Options of the `AerSimulator`, e.g. `method="statevector"`. On
            GPUs, they override the defaults of `GPU_OPTIONS`.

    Returns:
        The configured simulator.

    Raises:
        AerError: If a GPU is requested, but none is available.
    """
    device = AerDevice(device)
    gpu_available = "GPU" in available_aer_devices()

    if device == AerDevice.GPU and not gpu_available:
        raise AerError("A GPU simulator was requested, but no GPU is available.")

    if device == AerDevice.GPU or (device == AerDevice.AUTO and gpu_available):
        return AerSimulator(device="GPU", **{**GPU_OPTIONS, **options})
    return AerSimulator(device="CPU", **options)
