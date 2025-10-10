"""Module for Qiskit simulators."""

from qiskit_aer import AerError, AerSimulator


def generate_aer_simulator(force_gpu: bool = False) -> AerSimulator:
    """A function to prepare the Aer simulator with desired configuration. GPU accelerated in case available.

    Args:
        force_gpu (bool): If True, enforces GPU usage which means it will throw an error if no GPU is available. Default is False.

    Returns:
        AerSimulator: The prepared Aer simulator.
    """
    available_devices: tuple[str] = AerSimulator().available_devices()  # type: ignore[]
    if "GPU" in available_devices:
        simulator = AerSimulator(
            device="GPU",
            # enable accelerating using Nvidia's cuStateVec library
            cuStateVec_enable=True,
            # maximize the use of all available GPU and CPU nodes
            blocking_enable=True,
            # distribute shots to different available GPUs
            batched_shots_gpu=True,
        )
    elif force_gpu:
        raise AerError("Asked for GPU-accelerated simulation but no GPU was available.")
    else:
        simulator = AerSimulator(
            device="CPU",
        )

    return simulator
