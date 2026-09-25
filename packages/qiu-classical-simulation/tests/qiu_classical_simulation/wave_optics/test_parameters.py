"""Unit tests for parameters.py."""

from datetime import datetime

import numpy as np
import pytest
from python_pytest_helper.assertions import assert_close
from qiu_classical_simulation.wave_optics.parameters import ExperimentParameters
from qiu_signals.integer_axis import IndexOrdering


class TestDerivedParameters:
    """Test the parameters derived from the given ones."""

    def test_lens_geometry(self, small_experiment):
        """Test the lensmaker's curvature and the thickness of the spherical cap."""
        params = small_experiment()
        radius = params.focal_length * (params.refractive_index - 1)
        assert_close(params.radius_of_curvature, radius)
        assert_close(
            (radius - params.lens_thickness) ** 2 + params.lens_radius**2, radius**2
        )

    def test_grid(self, small_experiment):
        """Test the axes of the 2**n transverse samples filling the window."""
        params = small_experiment()
        assert params.x_axis.size == params.k_axis.size == 2**params.num_qubits
        assert params.x_axis.ordering is IndexOrdering.NATURAL
        assert params.k_axis.ordering is IndexOrdering.FFT
        assert_close(params.x_axis.sampling_window_length, params.transverse_length)

    def test_initial_state(self, small_experiment):
        """Test that the initial state is the normalized Gaussian beam."""
        params = small_experiment()
        assert_close(np.linalg.norm(params.initial_state), 1.0)
        assert np.argmax(np.abs(params.initial_state)) == params.dimension // 2

    def test_lens_slices(self, small_experiment):
        """Test that the slices fill the lens, narrowing towards the vertex."""
        params = small_experiment()
        assert len(params.lens_signals) == params.lens_slices
        assert_close(
            params.lens_slice_thickness * params.lens_slices, params.lens_thickness
        )
        assert np.all(np.diff(params.lens_transverse_radii) > 0)

    def test_reverse_order(self, small_experiment):
        """Test that the reverse order passes the slices from the plane side."""
        forward, reverse = small_experiment(), small_experiment(lens_reverse_order=True)
        assert forward.ordered_lens_signals[0] is forward.lens_signals[0]
        reversed_data = [signal.data for signal in reverse.ordered_lens_signals]
        forward_data = [signal.data for signal in forward.lens_signals]
        for reversed_slice, forward_slice in zip(
            reversed_data, forward_data[::-1], strict=True
        ):
            np.testing.assert_array_equal(reversed_slice, forward_slice)

    def test_validity(self, small_experiment):
        """Test the sampling condition `delta_x <= wavelength`."""
        assert small_experiment(num_qubits=7).is_valid()
        coarse = small_experiment(num_qubits=5)
        assert not coarse.is_valid()
        assert "delta_x" in coarse.validity_problems()[0]


class TestIdentity:
    """Test the time and the ID of experiments."""

    def test_unique_per_instance(self, small_experiment):
        """Test that each experiment gets its own ID and time."""
        first, second = small_experiment(), small_experiment()
        assert first.uuid != second.uuid
        assert second.experiment_datetime >= first.experiment_datetime


class TestStorage:
    """Test to_dict and from_dict."""

    def test_round_trip(self, small_experiment):
        """Test that the stored values recreate the parameters."""
        params = small_experiment(lens_reverse_order=True, direct_propagator=False)
        assert ExperimentParameters.from_dict(params.to_dict()) == params

    def test_legacy_keys(self, small_experiment):
        """Test that the former names of the time and the order are read."""
        values = small_experiment(lens_reverse_order=True).to_dict()
        values["timestamp"] = values.pop("experiment_datetime")
        values["reverse_order"] = values.pop("lens_reverse_order")
        del values["direct_propagator"], values["uuid"]

        params = ExperimentParameters.from_dict(values)
        assert params.lens_reverse_order
        assert params.direct_propagator
        assert isinstance(params.experiment_datetime, datetime)

    def test_missing_parameters(self, small_experiment):
        """Test that missing parameters must be given as defaults."""
        values = small_experiment().to_dict()
        del values["fresnel_approximation"]
        with pytest.raises(KeyError, match="fresnel_approximation"):
            ExperimentParameters.from_dict(values)
        params = ExperimentParameters.from_dict(
            values, defaults={"fresnel_approximation": True}
        )
        assert params.fresnel_approximation
