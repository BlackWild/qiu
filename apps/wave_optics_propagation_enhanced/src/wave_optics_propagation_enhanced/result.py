import numpy as np


class ExperimentResult:
    snapshots: dict[str, np.ndarray]
    total_lenses_simulated: int

    def __init__(self, snapshots: dict[str, np.ndarray], total_lenses_simulated: int):
        self.snapshots = snapshots
        self.total_lenses_simulated = total_lenses_simulated
