"""Storage of lens experiments, one folder per run.

A run folder holds `initial_parameters.json`, the parameters together with the scalar
results, and `results.npz`, the snapshots. Runs are stored as `<results_dir>/<uuid>`;
older ones as `<results_dir>/<timestamp>`, which reads the same way.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np

from python_wave_optics.parameters import ExperimentParameters
from python_wave_optics.result import ExperimentResult

PARAMETERS_FILENAME = "initial_parameters.json"
RESULTS_FILENAME = "results.npz"


def save_experiment(
    results_dir: str | Path, parameters: ExperimentParameters, result: ExperimentResult
) -> Path:
    """Store a run in the folder `<results_dir>/<uuid>`, and return the folder."""
    folder = Path(results_dir) / parameters.uuid
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / PARAMETERS_FILENAME, "w") as file:
        json.dump({**parameters.to_dict(), **result.summary()}, file)
    np.savez(folder / RESULTS_FILENAME, allow_pickle=False, **result.snapshots)
    return folder


def load_stored_values(folder: str | Path) -> dict[str, Any]:
    """Return the stored parameters and scalar results of a run."""
    with open(Path(folder) / PARAMETERS_FILENAME) as file:
        return json.load(file)


def load_parameters(
    folder: str | Path, defaults: dict[str, Any] | None = None
) -> ExperimentParameters:
    """Load the parameters of a run, see `ExperimentParameters.from_dict`."""
    return ExperimentParameters.from_dict(load_stored_values(folder), defaults)


def load_result(folder: str | Path) -> ExperimentResult:
    """Load the result of a run."""
    values = load_stored_values(folder)
    with np.load(Path(folder) / RESULTS_FILENAME, allow_pickle=False) as data:
        snapshots = {name: data[name] for name in data.files}
    return ExperimentResult(
        snapshots=snapshots,
        total_lenses_simulated=values["total_lenses_simulated"],
        total_probability_of_success=values.get("total_probability_of_success"),
        success_probabilities=values.get("success_probabilities"),
    )


def load_experiment(
    folder: str | Path, defaults: dict[str, Any] | None = None
) -> tuple[ExperimentParameters, ExperimentResult]:
    """Load the parameters and the result of a run."""
    return load_parameters(folder, defaults), load_result(folder)


def run_folders(results_dir: str | Path) -> list[Path]:
    """Return the run folders in a results directory, sorted by name."""
    return sorted(
        path
        for path in Path(results_dir).iterdir()
        if (path / PARAMETERS_FILENAME).is_file()
    )
