"""Unit tests for storage.py, result.py and analysis.py."""

import json
from dataclasses import replace

import numpy as np
import pytest
from python_pytest_helper.assertions import assert_close
from python_wave_optics.analysis import (
    beam_waist,
    lens_surface,
    principal_plane_position,
    propagation_distances,
    thin_lens_reference_states,
)
from python_wave_optics.cli import parse_parameters
from python_wave_optics.result import ExperimentResult
from python_wave_optics.simulation import ExactBackend, simulate
from python_wave_optics.storage import (
    PARAMETERS_FILENAME,
    RESULTS_FILENAME,
    load_experiment,
    run_folders,
    save_experiment,
)


class TestStorage:
    """Test storing and loading runs."""

    def test_round_trip(self, small_experiment, tmp_path):
        """Test that a stored run loads as it was."""
        params = small_experiment()
        result = simulate(params, ExactBackend())
        folder = save_experiment(tmp_path, params, result)

        assert folder == tmp_path / params.uuid
        assert run_folders(tmp_path) == [folder]
        loaded_params, loaded_result = load_experiment(folder)
        assert loaded_params == params
        assert loaded_result.success_probabilities == result.success_probabilities
        assert list(loaded_result.snapshots) == list(result.snapshots)
        for name, state in result.snapshots.items():
            np.testing.assert_array_equal(loaded_result.snapshots[name], state)

    def test_never_overwrites_a_run(self, small_experiment, tmp_path):
        """Test that a run of the same uuid, e.g. of copied parameters, is refused."""
        params = small_experiment()
        result = simulate(params, ExactBackend())
        save_experiment(tmp_path, params, result)
        with pytest.raises(FileExistsError):
            save_experiment(tmp_path, replace(params, max_delta=0.1), result)
        save_experiment(tmp_path, replace(params, uuid="another"), result)
        assert len(run_folders(tmp_path)) == 2

    def test_legacy_column_snapshots(self, small_experiment, tmp_path):
        """Test that snapshots stored as column vectors load flattened."""
        params = small_experiment()
        with open(tmp_path / PARAMETERS_FILENAME, "w") as file:
            json.dump({**params.to_dict(), "total_lenses_simulated": 3}, file)
        np.savez(tmp_path / RESULTS_FILENAME, step_0=np.ones((4, 1), dtype=complex))

        _, result = load_experiment(tmp_path)
        assert result.snapshots["step_0"].shape == (4,)
        assert result.success_probabilities is None


class TestResult:
    """Test ExperimentResult."""

    def test_success_probability_by_snapshot(self):
        """Test that the probabilities are looked up by the snapshot names."""
        result = ExperimentResult(
            snapshots={
                "step_0": np.ones(2, complex),
                "after_lens": np.ones(2, complex),
            },
            total_lenses_simulated=0,
            success_probabilities=[1.0, 0.5],
        )
        assert result.success_probability("after_lens") == 0.5

    def test_without_probabilities(self):
        """Test that results of the direct propagator may have no probabilities."""
        result = ExperimentResult({"step_0": np.ones(2, complex)}, 0)
        with pytest.raises(ValueError, match="no success probabilities"):
            result.success_probability("step_0")


class TestAnalysis:
    """Test the analysis quantities."""

    def test_beam_waist_of_a_gaussian(self):
        """Test that the waist of `exp(-x**2 / w**2)` is w."""
        x = np.linspace(-50, 50, 2001)
        assert_close(beam_waist(np.exp(-(x**2) / 4.0**2), x), 4.0)

    def test_principal_plane(self, small_experiment):
        """Test the principal plane: at `t - t / n` forward, at the convex vertex reversed."""
        forward, reverse = small_experiment(), small_experiment(lens_reverse_order=True)
        t, n = forward.lens_thickness, forward.refractive_index
        assert_close(principal_plane_position(forward), t - t / n)
        assert_close(principal_plane_position(reverse), t)

    @pytest.mark.parametrize("fresnel_approximation", [False, True])
    def test_lens_surface(self, small_experiment, fresnel_approximation: bool):
        """Test that the surface spans the lens and is mirrored for the reverse order."""
        forward = small_experiment(fresnel_approximation=fresnel_approximation)
        reverse = small_experiment(
            fresnel_approximation=fresnel_approximation, lens_reverse_order=True
        )
        surface = lens_surface(forward)
        assert np.min(surface) == 0.0  # at the center, a sample of the window
        assert_close(lens_surface(reverse), forward.lens_thickness - surface)

    def test_references_along_the_propagation(self, small_experiment):
        """Test one thin lens reference per slice and step."""
        params = small_experiment()
        distances = propagation_distances(params)
        assert len(distances) == params.lens_slices + params.num_of_steps_after_lens
        assert_close(distances[params.lens_slices - 1], params.lens_thickness)
        assert len(thin_lens_reference_states(params)) == len(distances)


def test_command_line(tmp_path):
    """Test that the command line varies the experiment of the paper."""
    params, results_dir = parse_parameters(
        "", tmp_path, ["--max-delta", "0.1", "--reverse-order", "--num-qubits", "6"]
    )
    assert results_dir == tmp_path
    assert params.max_delta == 0.1
    assert params.lens_reverse_order and not params.direct_propagator
    assert params.num_qubits == 6 and params.lens_slices == 10000
