import numpy as np
from wave_optics_propagation.storage import load_initial_parameters, load_numpy_results


class ExperimentResult:
    snapshots: dict[str, np.ndarray]
    total_lenses_simulated: int

    def __init__(self, snapshots: dict[str, np.ndarray], total_lenses_simulated: int):
        self.snapshots = snapshots
        self.total_lenses_simulated = total_lenses_simulated

    @classmethod
    def from_file(cls, path: str) -> "ExperimentResult":
        snapshots = load_numpy_results(path)

        # TODO: should actually save and load this from the results stored object later
        initial_parameters = load_initial_parameters(path)
        total_lenses_simulated = initial_parameters["total_lenses_simulated"]

        return cls(snapshots=snapshots, total_lenses_simulated=total_lenses_simulated)
