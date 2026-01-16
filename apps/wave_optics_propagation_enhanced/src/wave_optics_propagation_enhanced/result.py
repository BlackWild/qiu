from dataclasses import dataclass

import numpy as np
from wave_optics_propagation.storage import load_initial_parameters, load_numpy_results


@dataclass
class ExperimentResult:
    snapshots: dict[str, np.ndarray]
    total_lenses_simulated: int
    total_probability_of_success: float

    @classmethod
    def from_file(cls, path: str) -> "ExperimentResult":
        snapshots = load_numpy_results(path)

        # TODO: should actually save and load this from the results stored object later
        initial_parameters = load_initial_parameters(path)
        total_lenses_simulated = initial_parameters["total_lenses_simulated"]
        total_probability_of_success = initial_parameters[
            "total_probability_of_success"
        ]

        return cls(
            snapshots=snapshots,
            total_lenses_simulated=total_lenses_simulated,
            total_probability_of_success=total_probability_of_success,
        )
