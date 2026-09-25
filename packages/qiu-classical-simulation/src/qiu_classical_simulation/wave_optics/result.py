"""The result of a lens experiment: snapshots of the transverse field along the way.

The snapshots are named, in the order the simulation takes them: `step_0` for the
entering beam, `step_lens_{i}` after each lens slice `i`, `after_lens` behind the lens,
`step_after_lens_{j}` after each free propagation step `j = 1, 2, ...`, and `final`.
"""

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt


def lens_snapshot_name(slice_index: int) -> str:
    """Return the name of the snapshot after a lens slice, counted from 0."""
    return f"step_lens_{slice_index}"


def free_space_snapshot_name(step: int) -> str:
    """Return the name of the snapshot after a free propagation step, from 1."""
    return f"step_after_lens_{step}"


@dataclass
class ExperimentResult:
    """The snapshots of an experiment and the success probabilities of its protocol."""

    snapshots: dict[str, npt.NDArray[np.complex128]]
    """The normalized amplitudes of the named snapshots, in the order taken."""
    total_lenses_simulated: int
    """The number of lens slices with a phase, i.e. not skipped as constant."""
    total_probability_of_success: float | None = None
    """The probability that all cycles of the phase protocol succeeded."""
    success_probabilities: list[float] | None = field(default=None)
    """The cumulative success probability at each snapshot, in the same order."""

    def __post_init__(self) -> None:
        """Flatten the snapshots, which older results stored as column vectors."""
        self.snapshots = {
            name: np.asarray(state).reshape(-1)
            for name, state in self.snapshots.items()
        }

    def success_probability(self, snapshot: str) -> float:
        """Return the cumulative success probability up to a named snapshot."""
        if self.success_probabilities is None:
            raise ValueError("The result has no success probabilities.")
        return self.success_probabilities[list(self.snapshots).index(snapshot)]

    def lens_states(self, lens_slices: int) -> list[npt.NDArray[np.complex128]]:
        """Return the snapshots after each lens slice."""
        return [self.snapshots[lens_snapshot_name(i)] for i in range(lens_slices)]

    def free_space_states(self, steps: int) -> list[npt.NDArray[np.complex128]]:
        """Return the snapshots after each free propagation step."""
        return [self.snapshots[free_space_snapshot_name(j + 1)] for j in range(steps)]

    def summary(self) -> dict:
        """Return the scalar results, stored with the parameters."""
        return {
            "total_lenses_simulated": self.total_lenses_simulated,
            "total_probability_of_success": self.total_probability_of_success,
            "success_probabilities": self.success_probabilities,
        }
